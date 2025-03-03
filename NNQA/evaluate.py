import copy
import os,sys
import json,random
import wandb

from transformers import AutoTokenizer
from collections import defaultdict

from csr_utils.my_data_prep import build_data_loader
from csr_utils.my_bert_features import *
from csr_utils.training_control import *
from model.mymodel import CSN
CHARACTER_MASK_ID=10000

SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIP_DIR,'./..'))
from utils import ConstructSingleQuoteInstance,compute_accuracy
from sapr.sap_rev import Prediction,sap_rev


def evaluate(model, tokenizer, eval_data, eval_insts, is_chinese=False, device='cuda:0',use_sap=True):
    model.eval()

    omit_data =0
    predictions = {}
    pred_objs = []
    for i, (qids, segments_batch, CSSs_batch, CSSs_tokens_batch, sent_char_lens_batch, mention_poses_batch, quote_idxes_batch, one_hot_label_batch,
    true_index_batch, _, alias2ids_batch) in enumerate(eval_data):
        #seg_sents: [sent_tokens_1,...,sent_tokens_N]
        #CSSs: [sent_str_m,..,sent_str_n]
        #CSSs_tokens:[sent_tokens_m,...,sent_tokens_n]
        # skip the instance whose candidate does not contain the correct answer
        with torch.no_grad():
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
            qids = [e for k,e in enumerate(qids) if k not in removed_ids]
            segments_batch = [e for k,e in enumerate(segments_batch) if k not in removed_ids]
            #CSSs_batch = [e for k, e in enumerate(CSSs_batch) if k not in removed_ids]
            CSSs_tokens_batch = [e for k,e in enumerate(CSSs_tokens_batch) if k not in removed_ids]
            sent_char_lens_batch = [e for k, e in enumerate(sent_char_lens_batch) if k not in removed_ids]
            mention_poses_batch = [e for k, e in enumerate(mention_poses_batch) if k not in removed_ids]
            quote_idxes_batch = [e for k, e in enumerate(quote_idxes_batch) if k not in removed_ids]
            true_index_batch = [e for k, e in enumerate(true_index_batch) if k not in removed_ids]
            features_batch = [e for k, e in enumerate(features_batch) if k not in removed_ids]
            alias2ids_batch = [e for k,e in enumerate(alias2ids_batch) if k not in removed_ids]
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

            for qid,scores,seg_sents,CSSs_tokens,mention_poses,alias2id in zip(qids,scores_batch,segments_batch,CSSs_tokens_batch,mention_poses_batch,alias2ids_batch):
                candidate_aliases = [''.join(CSS_tokens[cdd_pos[1]:cdd_pos[2]]) if is_chinese else
                                     tokenizer.convert_tokens_to_string(CSS_tokens[cdd_pos[1]:cdd_pos[2]])
                                     for cdd_pos, CSS_tokens in zip(mention_poses, CSSs_tokens)]
                candidate_ids = [alias2id[x] for x in candidate_aliases]
                predict_index = scores.max(0).indices.item()
                predict_speaker = candidate_aliases[predict_index]
                predictions[qid] = predict_speaker
                pred_obj = Prediction(seg_sents,scores.detach().cpu().numpy(),candidate_ids,alias2id,qid)
                pred_objs.append(pred_obj)

    if use_sap:
        revised_predictions =copy.deepcopy(predictions)
        sap_rev_dict = sap_rev(pred_objs, th=2, is_chinese=is_chinese)

        for pred_obj_idx,pred_obj in enumerate(pred_objs):
            if pred_obj_idx in sap_rev_dict:
                id2alias = defaultdict(list)
                for name, name_id in pred_obj.alias2ids.items():
                    id2alias[str(name_id)].append(name)
                try:
                    revised_speaker = id2alias[str(sap_rev_dict[pred_obj_idx])][0]
                except Exception as e:
                    print(f'alias2id:{pred_obj.alias2ids}')
                    print(f'id2alias:{id2alias}')
                    print(f'qid:{pred_obj.quote_id}')
                    raise  e
                revised_predictions[pred_obj.quote_id] = revised_speaker


    records = []
    revised_records = []
    for inst in eval_insts:
        for para in inst['dialogue']:
            for quote_index in range(len(para['utterance'])):
                qid = para['utterance'][quote_index]['quote_id']
                quote = para['utterance'][quote_index]['quote']
                speaker = para['utterance'][quote_index]['speaker']

                if qid not in predictions:
                    print(f'unanswered quote:{qid}')
                    continue

                context = '\n'.join(map(lambda x: x['paragraph'],
                                        inst['preceding_paragraphs'] + inst['dialogue'] + inst[
                                            'succeeding_paragraphs']))
                record = {
                    'context': context,
                    'quote': quote,
                    'quote_id': qid,
                    'instance_id': inst['id'],
                    'predict_speaker': predictions[qid],
                    'label': speaker,
                    'character': inst['character']
                }
                records.append(record)

                if use_sap:
                    revised_record = copy.deepcopy(record)
                    revised_record['predict_speaker'] = revised_predictions[qid]
                    revised_records.append(revised_record)

    test_table = wandb.Table(columns=list(records[0].keys()))
    for record in random.sample(records, k=min(100,len(records))):
        test_table.add_data(*[str(record[k]) for k in record.keys()])
    wandb.log({'test_record': test_table})

    return records,revised_records


def test(args):
    device = torch.device('cuda:0')

    dataset_name = \
        list(filter(lambda x: x in args.test_file,
                    ['CSI', 'JY', 'WP2021', 'PAP', 'RIQUA', 'PDNC_genre', 'PDNC_merge', 'SQLITE']))[
            0]
    wandb_run = wandb.init(
        project=f"NNSI_{dataset_name}_standard_new",
        name=f'test_{args.bert_pretrained_dir}_new2_lr({args.lr})_bsz({args.batch_size})_ep({args.num_epochs})_sd({args.seed})',
        config=args
    )
    if dataset_name in ['CSI', 'JY', 'WP2021']:
        is_chinese = True
        print('evaluating on chinese dataset...')
    else:
        is_chinese = False
        print('evaluating on english dataset...')

    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)

    test_loader = build_data_loader(args.test_file, args.batch_size, args.length_limit)

    with open(args.test_file, 'r') as f:
        raw_test_data = json.load(f)
        single_quote_insts = ConstructSingleQuoteInstance(raw_test_data,tokenizer=tokenizer)


    model = CSN(args)
    model.load_state_dict(
        torch.load(os.path.join(args.checkpoint_dir, 'best_model/best_model.pt'), map_location='cpu'))
    model = model.to(device)

    records,revised_records = evaluate(model, tokenizer, test_loader, single_quote_insts, is_chinese, device, args.use_sap)
    test_scores = compute_accuracy(records)
    revised_test_scores=compute_accuracy(revised_records)
    test_acc = float(test_scores['acc'])
    revised_test_acc = float(revised_test_scores['acc'])
    wandb.log({'test_acc': test_acc})
    wandb.log({'revised_test_acc': revised_test_acc})

    eval_dir = os.path.join(args.checkpoint_dir, 'eval_log')
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, 'result_records.json'), 'w') as f:
        json.dump(records, f, indent=2)

    with open(os.path.join(eval_dir, 'revised_result_records.json'), 'w') as f:
        json.dump(revised_records,f,indent=2)

    with open(os.path.join(eval_dir, 'test_score.json'), 'w') as f:
        json.dump({'test_acc':test_scores,'revised_test_acc':revised_test_scores}, f, indent=2)

    print({'test_acc':test_acc,'revised_test_acc':revised_test_acc})