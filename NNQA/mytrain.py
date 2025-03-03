# Training script
import os,sys

import wandb
import time
from fastprogress import master_bar, progress_bar
import logging

import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer

from csr_utils.arguments import get_train_args
from csr_utils.my_data_prep import build_data_loader
from csr_utils.my_bert_features import *
from csr_utils.training_control import *
from model.mymodel import CSN
from evaluate import evaluate,test

SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIP_DIR,'./..'))
from utils import ConstructSingleQuoteInstance,compute_accuracy

wandb.login(key='b3451a268e7b638ac4d8789aa1e8046da81710c5')



# training log
LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%m:%s %a'
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

def train_batch(args):
    """
    Training script.

    return
        best_dev_acc: the best development accuracy.
        best_test_acc: the accuracy on test instances of the model that has the best performance on development instances.
    """
    if any([dataset_name in args.train_file for dataset_name in ['CSI', 'JY', 'WP2021']]):
        is_chinese = True
        logger.info('training on chinese dataset...')
    else:
        is_chinese = False
        logger.info('training on english dataset...')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir,'arguments.json'),'w') as f:
        json.dump(vars(args),f,indent=2)

    # checkpoint
    model_dir = os.path.join(args.checkpoint_dir, 'best_model')
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)

    train_data = build_data_loader(args.train_file, args.batch_size, args.length_limit, skip_only_one=True,
                                   use_cache=args.use_cache)
    logger.info("The number of training instances: " + str(len(train_data)))
    dev_data = build_data_loader(args.dev_file, args.batch_size, args.length_limit, use_cache=args.use_cache)
    logger.info("The number of development instances: " + str(len(dev_data)))
    with open(args.dev_file,'r') as f:
        raw_dev_data = json.load(f)
        dev_single_quote_insts = ConstructSingleQuoteInstance(raw_dev_data,tokenizer=tokenizer)

    # initialize model

    model = CSN(args)
    model = model.to(device)

    # initialize optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown optimizer type...")

    # loss criterion
    loss_fn = nn.MarginRankingLoss(margin=args.margin)

    # training loop
    logger.info("############################Training Begins...################################")

    # logging best
    best_score = 0

    # control parameters
    patience_counter = 0

    epoch_bar = master_bar(range(args.num_epochs))
    for epoch in epoch_bar:
        model.train()
        optimizer.zero_grad()

        torch.autograd.set_detect_anomaly(True)

        logger.info('Epoch: %d' % (epoch + 1))
        omit_data = 0

        for i, (qids, _, CSSs_batch, CSSs_tokens_batch, sent_char_lens_batch, mention_poses_batch, quote_idxes_batch, one_hot_label_batch, true_index_batch, _,_) \
                in enumerate(progress_bar(train_data, total=len(train_data), parent=epoch_bar)):

            features_batch = convert_examples_to_features(batch=CSSs_tokens_batch, tokenizer=tokenizer)
            removed_ids = []
            for k in range(len(features_batch)):
                skip = False
                for f_id in range(len(features_batch[k])):
                    jieba_tok_num = sum(sent_char_lens_batch[k][f_id])
                    trans_tok_num = len(features_batch[k][f_id].tokens) - 2
                    # deduct two special tokens including [CLS] and [SEP]
                    if jieba_tok_num != trans_tok_num:
                        omit_data += 1
                        print(
                            f'skip current instance due to unequal tokens. jieba:{jieba_tok_num}, transformer:{trans_tok_num}')
                        skip = True
                        break
                if skip or true_index_batch[k] == -1:
                    removed_ids.append(k)
                    continue
            #CSSs_batch = [e for k,e in enumerate(CSSs_batch) if k not in removed_ids]
            sent_char_lens_batch = [e for k,e in enumerate(sent_char_lens_batch) if k not in removed_ids]
            mention_poses_batch = [e for k,e in enumerate(mention_poses_batch) if k not in removed_ids]
            quote_idxes_batch = [e for k,e in enumerate(quote_idxes_batch) if k not in removed_ids]
            #one_hot_label_batch = [e for k,e in enumerate(one_hot_label_batch) if k not in removed_ids]
            true_index_batch = [e for k,e in enumerate(true_index_batch) if k not in removed_ids]
            features_batch = [e for k,e in enumerate(features_batch) if k not in removed_ids]
            # if all instances in a batch are removed, skip the current batch
            if len(features_batch) == 0:
                continue

            try:
                scores_batch, scores_false_batch, scores_true_batch = model(features_batch, sent_char_lens_batch,
                                                                        mention_poses_batch, quote_idxes_batch,
                                                                        true_index_batch, device)
            except Exception as e:
                print(e)
                continue

            loss = 0
            loss_counter = 0
            removed_ids = []
            for sid, (scores_false, scores_true, scores) in enumerate(zip(scores_false_batch, scores_true_batch, scores_batch)):
                if len(scores_false) == 0 or len(scores_true) == 0:
                    removed_ids.append(sid)
                else:
                    #print(f"score false:{torch.cat(scores_false)}, len:{len(scores_false)}")
                    #print(f"score true:{torch.cat(scores_true)}, len:{len(scores_true)}")
                    loss += loss_fn(torch.cat(scores_false), torch.cat(scores_true),
                                    torch.tensor([-1.0] * len(scores_true)).to(device))
                    loss_counter += 1

            loss = loss / loss_counter
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            wandb.log({'loss': loss.item()})


        logger.info(f'# omitted instances:{omit_data}')

        # adjust learning rate after each epoch
        adjust_learning_rate(optimizer, args.lr_decay)

        # Evaluation
        model.eval()

        # development stage
        records,revised_records = evaluate(model,tokenizer,dev_data,dev_single_quote_insts,is_chinese=is_chinese,device=device,use_sap=args.use_sap)
        dev_scores = compute_accuracy(records)
        revised_dev_scores = compute_accuracy(revised_records)
        dev_acc = float(dev_scores['acc'])
        revised_dev_acc= float(revised_dev_scores['acc'])
        wandb.log({'dev_acc':dev_acc})
        wandb.log({'revised_dev_acc':revised_dev_acc})


        # save the model with best performance
        if  dev_acc >= best_score:
            best_score = dev_acc

            patience_counter = 0
            new_best = True
        else:
            patience_counter += 1
            new_best = False

        # only save the model which outperforms the former best on development set
        if new_best:
            try:
                torch.save(model.state_dict(),os.path.join(model_dir,'best_model.pt'))
                with open(os.path.join(model_dir,'dev_score.json'),'w') as f:
                    json.dump(dev_scores,f,indent=2)
            except Exception as e:
                print(e)

        # early stopping
        if patience_counter > args.patience:
            logger.info("Early stopping...")
            break

        logger.info('------------------------------------------------------')

    return best_score
if __name__ == '__main__':
    # run several times and calculate average accuracy and standard deviation
    args = get_train_args()
    data_files = list(filter(lambda x:x.strip(),[args.train_file,args.test_file]))
    assert len(data_files)>0, "no data file is specified"
    dataset_name = \
    list(filter(lambda x: x in data_files[0], ['CSI', 'JY', 'WP2021', 'PAP', 'RIQUA', 'PDNC_genre', 'PDNC_merge', 'SQLITE']))[
        0]
    logger.info('Enter training.....')

    torch.manual_seed(args.seed)
    if not args.eval_only:
        wandb_run = wandb.init(
            project=f"NNSI_{dataset_name}_standard_new",
            name=f'train_{args.bert_pretrained_dir}_new2_lr({args.lr})_bsz({args.batch_size})_ep({args.num_epochs})_sd({args.seed})',
            config=args
        )
        train_batch(args)

    else:

        test(args)

