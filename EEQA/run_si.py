import random
import os
import argparse
import numpy as np
import json
import torch
import wandb
from models.pytorch_modeling import BertConfig, BertForQuestionAnswering
from optimizations.pytorch_optimization import get_optimization, warmup_linear
from evaluate.cmrc2018_output import write_predictions
from evaluate.cmrc2018_evaluate import get_eval
import collections
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from tokenizations import official_tokenization as tokenization
from preprocess.cmrc2018_preprocess import json2features
from preprocess import csi_utils
from transformers import AutoConfig,AutoTokenizer,AutoModelForQuestionAnswering
from utils import compute_accuracy,ConstructSingleQuoteInstance

wandb.login(key='b3451a268e7b638ac4d8789aa1e8046da81710c5')

def generate_result_records(dev_file, predictions):

    data = json.load(open(dev_file, 'r', encoding='utf8'))
    single_quote_insts = ConstructSingleQuoteInstance(data=data,tokenizer=tokenizer)
    records = []
    skip_count = 0
    for instance in single_quote_insts:
        char_dict = instance['character']
        for para in instance["dialogue"]:
            for qindex in range(len(para['utterance'])):
                utterance_id = para['utterance'][qindex]['quote_id']
                if utterance_id not in predictions:
                    print('Unanswered question: {}\n'.format(utterance_id))
                    skip_count += 1
                    continue
                prediction = str(predictions[utterance_id])
                record = {'context':'\n'.join(map(lambda x:x['paragraph'],
                                                 instance['preceding_paragraphs']
                                                 +instance['dialogue']
                                                 +instance['succeeding_paragraphs'])),
                          'quote':para['utterance'][qindex]['quote'],
                          'predict_speaker':prediction,
                          'label':para['utterance'][qindex]['speaker'],
                          'quote_id':utterance_id,
                          'instance_id': instance['id'],
                          'character':char_dict
                          }
                records.append(record)
    test_table = wandb.Table(columns=list(records[0].keys()))
    for record in random.sample(records,k=100):
        test_table.add_data(*[str(record[k]) for k in record.keys()])
    wandb.log({'test_record':test_table})
    return records

def generate_result_records2(dev_file, predictions):

    data = json.load(open(dev_file, 'r', encoding='utf8'))
    single_quote_insts = ConstructSingleQuoteInstance(data=data,tokenizer=tokenizer)
    records = []
    skip_count = 0
    for instance in single_quote_insts:
        char_dict = instance['character']
        for para in instance["dialogue"]:
            for qindex in range(len(para['utterance'])):
                utterance_id = para['utterance'][qindex]['quote_id']
                if utterance_id in predictions:
                    prediction = str(predictions[utterance_id])
                else:
                    prediction = 'None'
                record = {'context':'\n'.join(map(lambda x:x['paragraph'],
                                                 instance['preceding_paragraphs']
                                                 +instance['dialogue']
                                                 +instance['succeeding_paragraphs'])),
                          'quote':para['utterance'][qindex]['quote'],
                          'predict_speaker':prediction,
                          'label':para['utterance'][qindex]['speaker'],
                          'quote_id':utterance_id,
                          'instance_id': instance['id'],
                          'character':char_dict
                          }
                records.append(record)
    test_table = wandb.Table(columns=list(records[0].keys()))
    for record in random.sample(records,k=100):
        test_table.add_data(*[str(record[k]) for k in record.keys()])
    wandb.log({'test_record':test_table})
    return records


def evaluate(model, args, eval_examples, eval_features, device, best_acc):
    print("***** Eval *****")
    if any([name in args.dev_file for name in ['CSI','WP2021','JY']]):
        eval_chinese_model = True
        print('evaluating chinese model...')
    else:
        eval_chinese_model = False
        print('evaluating english model...')
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    #output_prediction_file = os.path.join(args.checkpoint_dir,
    #                                      "predictions_steps" + str(global_steps) + ".json")
    #output_nbest_file = output_prediction_file.replace('predictions', 'nbest')

    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_feature_indices = torch.arange(all_input_ids.size(0), dtype=torch.long)

    all_start_positions = torch.tensor([f['start_position'] for f in eval_features], dtype=torch.long)
    all_end_positions = torch.tensor([f['end_position'] for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_feature_indices,
                               all_start_positions, all_end_positions)

    #eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_feature_indices)
    eval_dataloader = DataLoader(eval_data, batch_size=args.n_batch, shuffle=False)

    model.eval()
    all_results = []
    avg_loss = 0
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, feature_indices, start_positions, end_positions in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_loss,batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
            if n_gpu > 1:
                batch_loss = batch_loss.mean()  # mean() to average on multi-gpu.
            avg_loss += batch_loss.item()/len(eval_dataloader)

        for i, feature_index in enumerate(feature_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[feature_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    #print(f'all results:{len(all_results)}, samples:{all_results[:5]}')

    all_predictions,_=write_predictions(eval_examples, eval_features, all_results,
                                        n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                                        do_lower_case=True,is_chinese=eval_chinese_model)

    #tmp_result = get_eval(args.dev_file, all_predictions)
    result_records = generate_result_records(args.dev_file, all_predictions)
    score = compute_accuracy(result_records)
    tqdm.write(f'score:{score}, avg_loss:{avg_loss}')

    if args.eval_only:
        eval_dir = os.path.join(args.checkpoint_dir,'eval_log')
        os.makedirs(eval_dir,exist_ok=True)
        with open(os.path.join(eval_dir,'test_score.json'),'w') as f:
            score['avg_loss'] = str(avg_loss)
            json.dump(score,f,indent=2)
        with open(os.path.join(eval_dir,'result_records.json'),'w') as f:
            json.dump(result_records,f,indent=2)
        wandb.log({'test_acc':float(score['acc']),'avg_loss':avg_loss})
    else:
        wandb.log({'dev_acc':float(score['acc']), 'avg_loss':avg_loss})


    #if not args.eval_only and avg_loss <= best_acc:
    #    best_acc = avg_loss
    if not args.eval_only and float(score['acc'])>=best_acc:
        best_acc = float(score['acc'])
        model_dir = os.path.join(args.checkpoint_dir,'best_model')
        os.makedirs(model_dir,exist_ok=True)
        csi_utils.torch_save_model(model, model_dir)
        with open(os.path.join(model_dir,'dev_score.json'),'w') as f:
            score['avg_loss'] = str(avg_loss)
            json.dump(score,f,indent=2)

    model.train()

    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7')

    # training parameter
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--seed', nargs='+', type=int, default=[123])
    parser.add_argument('--float16', type=bool, default=False)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=6)
    parser.add_argument('--eval_epochs', type=float, default=0.5)
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--vocab_size', type=int, default=21128)

    # data dir
    parser.add_argument('--train_dir', type=str,
                        default='dataset/cmrc2018/train_features_roberta512.json')
    parser.add_argument('--dev_dir1', type=str,
                        default='dataset/cmrc2018/dev_examples_roberta512.json')
    parser.add_argument('--dev_dir2', type=str,
                        default='dataset/cmrc2018/dev_features_roberta512.json')
    parser.add_argument('--train_file', type=str,
                        default='origin_data/cmrc2018/cmrc2018_train.json')
    parser.add_argument('--dev_file', type=str,
                        default='origin_data/cmrc2018/cmrc2018_dev.json')
    parser.add_argument('--bert_config_file', type=str,
                        default='check_points/pretrain_models/roberta_wwm_ext_base/bert_config.json')
    parser.add_argument('--vocab_file', type=str,
                        default='check_points/pretrain_models/roberta_wwm_ext_base/vocab.txt')
    parser.add_argument('--init_restore_dir', type=str,
                        default='check_points/pretrain_models/roberta_wwm_ext_base/pytorch_model.pth')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='check_points/cmrc2018/roberta_wwm_ext_base/')
    parser.add_argument('--resumepar',
                        default=False,
                        action='store_true',
                        help="Whether to resume the training partially.")

    # use some global vars for convenience
    args = parser.parse_args()
    #args = csi_utils.check_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    dataset_name = list(filter(lambda x:x in args.train_dir,['CSI','JY','WP2021','PAP','RIQUA','PDNC_genre','PDNC_merge','SQLITE']))[0]


    wandb_run = wandb.init(
        project=f"EESI_{dataset_name}_standard_new",
        name=f"{'test' if args.eval_only else 'train'}_bert_{'large' if 'large' in args.init_restore_dir else 'base'}_lr({args.lr})_bs({args.n_batch})_ep({args.train_epochs})",
        config=args
    )

    os.makedirs(args.checkpoint_dir,exist_ok=True)
    with open(os.path.join(args.checkpoint_dir,'arguments.json'),'w') as f:
        json.dump(vars(args),f,indent=2)

    # load the bert setting
    if any([name in args.train_file for name in ['CSI','WP2021','JY']]):
        train_chinese_model = True
    else:
        train_chinese_model = False
    if train_chinese_model:
        print("training on chinese dataset")
        bert_config = BertConfig.from_json_file(args.bert_config_file)
        bert_config.hidden_dropout_prob = args.dropout
        bert_config.attention_probs_dropout_prob = args.dropout
        
        # load data
        print('loading data...')
        tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
        assert args.vocab_size == len(tokenizer.vocab)

    else:
        print("training on english dataset")
        tokenizer = tokenization.BertTokenizer.from_pretrained(args.init_restore_dir)
        bert_config1 = AutoConfig.from_pretrained(args.init_restore_dir)
        bert_config = BertConfig(vocab_size_or_config_json_file=bert_config1.vocab_size)
        bert_config.hidden_dropout_prob = args.dropout
        bert_config.attention_probs_dropout_prob = args.dropout

    
    if not args.eval_only and not os.path.exists(args.train_dir):

        json2features(args.train_file, [args.train_dir.replace('_features_', '_examples_'), args.train_dir],
                      tokenizer, is_training=True,
                      max_seq_length=bert_config.max_position_embeddings)

    if not os.path.exists(args.dev_dir1) or not os.path.exists(args.dev_dir2):
        json2features(args.dev_file, [args.dev_dir1, args.dev_dir2], tokenizer, is_training=False,
                      max_seq_length=bert_config.max_position_embeddings)

    if not args.eval_only:
        train_features = json.load(open(args.train_dir, 'r'))
    dev_examples = json.load(open(args.dev_dir1, 'r'))
    dev_features = json.load(open(args.dev_dir2, 'r'))

    if not args.eval_only:
        steps_per_epoch = len(train_features) // args.n_batch
        eval_steps = int(steps_per_epoch * args.eval_epochs)
    dev_steps_per_epoch = len(dev_features) // args.n_batch
    if not args.eval_only and len(train_features) % args.n_batch != 0:
        steps_per_epoch += 1
    if len(dev_features) % args.n_batch != 0:
        dev_steps_per_epoch += 1


    if not args.eval_only:
        total_steps = steps_per_epoch * args.train_epochs

        tqdm.write(f'steps per epoch: { steps_per_epoch}')
        tqdm.write(f'total steps: {total_steps}')
        tqdm.write(f'warmup steps: {int(args.warmup_rate * total_steps)}' )


        #for seed_ in args.seed:
        for seed_ in args.seed:

            tqdm.write(f'SEED: {seed_}' )

            random.seed(seed_)
            np.random.seed(seed_)
            torch.manual_seed(seed_)
            if n_gpu > 0:
                torch.cuda.manual_seed_all(seed_)

            if train_chinese_model:
                # init model
                tqdm.write('init model...')
                model = BertForQuestionAnswering(bert_config)
                
                csi_utils.torch_show_all_params(model)
                csi_utils.torch_init_model(model, args.init_restore_dir, args.resumepar)
            else:
                model = BertForQuestionAnswering(bert_config)
                model = model.from_pretrained(args.init_restore_dir)

            if args.float16:
                model.half()
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            optimizer = get_optimization(model=model,
                                         float16=args.float16,
                                         learning_rate=args.lr,
                                         total_steps=total_steps,
                                         schedule=args.schedule,
                                         warmup_rate=args.warmup_rate,
                                         max_grad_norm=args.clip_norm,
                                         weight_decay_rate=args.weight_decay_rate)

            all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f['input_mask'] for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features], dtype=torch.long)

            seq_len = all_input_ids.shape[1]

            assert seq_len <= bert_config.max_position_embeddings

            # true label
            all_start_positions = torch.tensor([f['start_position'] for f in train_features], dtype=torch.long)
            all_end_positions = torch.tensor([f['end_position'] for f in train_features], dtype=torch.long)

            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                       all_start_positions, all_end_positions)
            train_dataloader = DataLoader(train_data, batch_size=args.n_batch, shuffle=True)

            tqdm.write('***** Training *****')
            model.train()
            global_steps = 1
            # best_acc=1000000
            best_acc = 0
            for i in range(int(args.train_epochs)):
                tqdm.write(f'Starting epoch {i + 1}')
                total_loss = 0
                iteration = 1
                with tqdm(total=steps_per_epoch, desc='Epoch %d' % (i + 1)) as pbar:
                    for step, batch in enumerate(train_dataloader):
                        batch = tuple(t.to(device) for t in batch)
                        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                        loss,start_logits,end_logits = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                        if n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.
                        wandb.log({'loss':loss.item()})
                        total_loss += loss.item()

                        pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (iteration + 1e-5))})
                        pbar.update(1)

                        if args.float16:
                            optimizer.backward(loss)
                            # modify learning rate with special warm up BERT uses
                            # if args.fp16 is False, BertAdam is used and handles this automatically
                            lr_this_step = args.lr * warmup_linear(global_steps / total_steps, args.warmup_rate)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        else:
                            loss.backward()

                        optimizer.step()
                        model.zero_grad()
                        global_steps += 1
                        iteration += 1

                        #if global_steps % eval_steps == 0:
                best_acc = evaluate(model, args, dev_examples, dev_features, device, best_acc)


            del model
            del optimizer
            torch.cuda.empty_cache()
            wandb.log({'best_dev_loss':best_acc})

    else:
        print('init model...')
        model_path = os.path.join(args.checkpoint_dir,'best_model/best_model.pt')
        if train_chinese_model:
            # init model
            print('init model...')
            model = BertForQuestionAnswering(bert_config)

            csi_utils.torch_show_all_params(model)
            csi_utils.torch_init_model(model, model_path, args.resumepar)
        else:
            model = BertForQuestionAnswering.from_pretrained(args.init_restore_dir)
            csi_utils.torch_init_model(model,model_path, args.resumepar)
        
        if args.float16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        evaluate(model, args, dev_examples, dev_features, device, -1)
        
