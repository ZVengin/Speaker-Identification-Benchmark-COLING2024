import os, json, torch, sys, re, random, wandb, logging
from transformers import AutoTokenizer
from MagicTools import MagicModel, MagicParse, Param, get_dataloader
from model import *
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, './../..'))
from utils import ConstructSingleQuoteInstance, compute_accuracy, cut_sentence_with_quotation_marks

wandb.login(key='b3451a268e7b638ac4d8789aa1e8046da81710c5')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def Inference(model, tokenizer, batch, do_sample):
    outs = model(**batch)
    probs = torch.softmax(outs['logits'], dim=-1)
    labels_ids = torch.topk(probs, dim=-1, k=1)[1].squeeze(-1)
    label_nums = (outs['logits_mask'] == 1).sum(dim=1)

    results = []
    for index, labels_id, label_num in zip(batch['index'], labels_ids.tolist(), label_nums.tolist()):
        results.append({
            'index': index,
            'predict_label_id': [c for c in labels_id][:label_num]
        })
    return results


def ProcessOuts(tokenizer, batch_outputs, batch):
    return batch_outputs


def GetTokenType(text, tokenizer, chars_id, char_dict):
    text_ids = tokenizer.encode(text, truncation=True, max_length=max_len, add_special_tokens=False)
    char_pos = defaultdict(list)
    for char_id in chars_id:
        for men in char_dict['id2names'][str(char_id)]:
            tokenized_men_text = tokenizer.decode(tokenizer.encode(men, add_special_tokens=False))
            i = 0
            j = i + 1
            while (i < j and j < len(text_ids)):
                token_i = tokenizer.decode(text_ids[i:j]).strip()
                if token_i in tokenized_men_text:

                    while (j < len(text_ids)):
                        token_j = tokenizer.decode(text_ids[i:j + 1]).strip()
                        if token_j in tokenized_men_text:
                            j += 1
                        else:
                            break
                    if tokenizer.decode(text_ids[i:j]).strip() == tokenized_men_text:
                        # print(f'men:{tokenized_men_text},detect:{tokenizer.decode(text_ids[i:j]).strip()}')
                        overlap = False
                        # print(f'char_pos start:{char_pos}')
                        for subchar_id, pos in char_pos.items():
                            new_pos = []
                            for pi, p in enumerate(pos):
                                if p[0] >= i and p[-1] < j:
                                    # print(f'skip:{p}={tokenizer.decode([text_ids[_] for _ in p])}')
                                    continue
                                if p[0] <= i and p[-1] >= j:
                                    overlap = True
                                new_pos.append(p)
                            char_pos[subchar_id] = new_pos

                        if not overlap:
                            char_pos[char_id].append(list(range(i, j)))
                        # print(f'char_pos end:{char_pos}')
                i = j
                j = i + 1
    # print(f'final:{char_pos}')
    token_type = torch.zeros(len(text_ids)).fill_(CharMaskId)
    for char_id, pos in char_pos.items():
        p = sum(pos, [])
        token_type[p] = char_id
    token_type = token_type.tolist()
    # pp.pprint(list(zip(tokenizer.tokenize(text),token_type)))
    return token_type


def CollateFn(data):
    indexs, input_ids, attention_mask, token_types, labels = [], [], [], [], []
    for e in data:
        indexs.append(e['index'])
        input_ids.append(e['input_ids'])
        attention_mask.append(e['attention_mask'])
        token_types.append(e['token_type'])
        if 'labels' in e:
            labels.append(e['labels'])
    max_text_len = max([len(text_ids) for text_ids in input_ids])
    input_ids = torch.LongTensor([text_ids + [tokenizer.pad_token_id] * (max_text_len - len(text_ids))
                                  for text_ids in input_ids])
    attention_mask = torch.Tensor([e_mask + [0] * (max_text_len - len(e_mask)) for e_mask in attention_mask])
    token_types = torch.Tensor([e_type + [CharMaskId] * (max_text_len - len(e_type))
                                for e_type in token_types])

    if len(labels) > 0 and len(labels) == len(indexs):
        max_utter_num = max([len(e_labels) for e_labels in labels])
        labels = [e_labels + [CharMaskId] * (max_utter_num - len(e_labels))
                  for e_labels in labels]
        labels = torch.LongTensor(labels)

    model_inputs = {
        'index': indexs,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type': token_types
    }
    if len(labels) > 0:
        model_inputs['labels'] = labels
    return model_inputs


def ProcessInputs(tokenizer, instance, is_train):
    index = instance['index']
    text_encodes = tokenizer(instance['text'], truncation=True, max_length=max_len, add_special_tokens=False)
    char_dict = instance['character']
    reindex_dict = instance['reindex_dict']
    if is_train and 'label' in instance:
        # print(f'instance label:{instance["label"]}')
        inst_labels = [char_dict['name2id'][c] for c in instance['label']]
        inst_labels = [reindex_dict['old2new'][str(l)] for l in inst_labels]

        # we count the utterances remained in the truncated text
        label_num = (torch.Tensor(text_encodes['input_ids']) == tokenizer.mask_token_id).sum().item()
        # remove additional utterance labels which exceed the length limitation of the bart encoder
        inst_labels = inst_labels[:label_num]
        instance_inputs = {
            'index': index,
            'input_ids': text_encodes['input_ids'],
            'attention_mask': text_encodes['attention_mask'],
            'token_type': instance['token_type'],
            'labels': inst_labels}
    else:
        instance_inputs = {
            'index': index,
            'input_ids': text_encodes['input_ids'],
            'attention_mask': text_encodes['attention_mask'],
            'token_type': instance['token_type']}

    return instance_inputs


def GetLoss(model, batch):
    loss = model(**batch)['loss']
    return loss


def ConstructInstanceFinetune(data, tokenizer, is_train=True, is_chinese=False):
    single_quote_insts = ConstructSingleQuoteInstance(data,tokenizer=tokenizer)
    print(f'single quote instance number:{len(single_quote_insts)}')
    instances = []
    omit_num = 0
    for inst in single_quote_insts:
        for utter in inst['dialogue']:
            #if utter['utterance'][0]['speaker'] == 'None':
            #    continue
            sents = cut_sentence_with_quotation_marks(utter['paragraph'], is_chinese)
            sents = list(map(lambda x: x['sentence'], sents))
            for quote_index in range(len(utter['utterance'])):
                context = []
                labels = [utter['utterance'][quote_index]['speaker']]
                for p in inst['preceding_paragraphs']:
                    context.append(p['paragraph'])
                quote = utter['utterance'][quote_index]['quote']
                quote_pos = -1
                for sent_index in range(len(sents)):
                    sent = sents[sent_index]
                    if sent.find(quote.strip()) != -1:
                        quote_pos = sent_index
                        break
                if quote_pos == -1:
                    omit_num += 1
                    continue
                quote_start = utter['paragraph'].find(sents[quote_pos])
                utter_text = (utter['paragraph'][:quote_start]
                              + f' {tokenizer.mask_token} '
                              + utter['paragraph'][quote_start: quote_start + len(sents[quote_pos])]
                              + f' {tokenizer.sep_token} '
                              + utter['paragraph'][quote_start + len(sents[quote_pos]):])
                context.append(utter_text)
                for p in inst['succeeding_paragraphs']:
                    context.append(p['paragraph'])

                context = '\n'.join(context)
                mask_pos = context.find(tokenizer.mask_token)
                mask_pos = len(tokenizer.tokenize(context[:mask_pos]))
                if mask_pos >= max_len:
                    continue

                char_dict = inst['character']
                label_ids = [int(char_dict['name2id'][label]) for label in labels]
                label_set = set(label_ids)
                all_chars_id = [int(char_id) for char_id in list(char_dict['id2names'].keys())
                                if int(char_id) != CharMaskId]
                token_type = GetTokenType(context, tokenizer, all_chars_id, char_dict)

                token_type_set = set(filter(lambda x: x != CharMaskId, token_type))
                overlap_set = token_type_set.intersection(label_set)

                if len(token_type_set) == 0 or len(token_type_set) > MaxCharNum or len(overlap_set) < len(label_set):
                    continue

                reindex_dict = {
                    'new2old': dict(
                        [(str(nid), int(oid)) for nid, oid in enumerate(sorted(token_type_set))] + [
                            (CharMaskId, CharMaskId)]),
                    'old2new': dict(
                        [(str(int(oid)), nid) for nid, oid in enumerate(sorted(token_type_set))] + [
                            (CharMaskId, CharMaskId)])}

                construct_inst = {
                    'index': utter['utterance'][quote_index]['quote_id'],
                    'text': context,
                    'label': labels,
                    'quote': quote,
                    'character': char_dict,
                    'reindex_dict': reindex_dict,
                    'token_type': token_type
                }

                instances.append(construct_inst)

    print(f'the constructed instance: {len(instances)}')
    print(f'the omitted instance: {omit_num}')

    feature_table = wandb.Table(columns=[k for k in instances[0].keys()])
    for inst in random.sample(instances, k=20):
        feature_table.add_data(*[str(inst[k]) for k in inst.keys()])
    wandb.log({'feature': feature_table})
    return instances



def ConstructRecords(eval_data, results):
    results = {r['index']: r for r in results}
    records = []
    for inst in eval_data:
        context = '\n'.join(map(lambda x: x['paragraph'],
                                inst['preceding_paragraphs']
                                + inst['dialogue']
                                + inst['succeeding_paragraphs']))
        char_dict = inst['character']
        for para in inst['dialogue']:
            for quote_index in range(len(para['utterance'])):
                qid = para['utterance'][quote_index]['quote_id']
                quote = para['utterance'][quote_index]['quote']
                if qid not in results:
                    continue
                result = results[qid]
                reindex_dict = result['reindex_dict']
                predict_labels_id = [
                    int(reindex_dict['new2old'][str(c)]) if str(c) in reindex_dict['new2old'] else CharMaskId
                    for c in result['predict_label_id']]
                assert len(predict_labels_id) == 1, 'invalid predicted speaker number!'
                predict_speaker = char_dict['id2names'][str(predict_labels_id[0])][0]
                assert len(result['label']) == 1, 'invalid label number!'
                label = result['label'][0]
                record = {
                    'context': context,
                    'quote_id': qid,
                    'quote': quote,
                    'instance_id': inst['id'],
                    'predict_speaker': predict_speaker,
                    'label': label,
                    'character': char_dict
                }
                records.append(record)
    return records




def main(config):
    # create the checkpoint dir to save the best model checkpoint
    best_model_dir = os.path.join(config.checkpoint_dir, 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)

    # initialize the model
    model = SpeakerModel(config.layer_num, model_mapping[config.model_name], 8,
                         hidden_num=hidden_size_mapping[config.model_name], tokenizer=tokenizer)

    model.to_device({'bart_encoder': 'cuda:0', 'char_encoder': 'cuda:0'})

    model = MagicModel(
        model,
        tokenizer,
        cache_dir=config.cache_dir,
        loss_function=GetLoss,
        inference=Inference,
        compute_score=compute_accuracy,
        init_eval_score=1e5 if config.optimize_direction == 'min' else 0,
        optimize_direction=config.optimize_direction)

    with open(os.path.join(config.checkpoint_dir, "arguments.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    train_loader = get_dataloader(
        config.train_file,
        format='json',
        tokenizer=model._tokenizer,
        construct_instance=ConstructInstanceFinetune,
        process_inputs=ProcessInputs,
        sample_weight=None,
        is_train=True,
        is_chinese=is_chinese,
        use_cache=config.use_cache,
        cache_dir=config.cache_dir,
        batch_size=config.batch_size,
        collate_fn=CollateFn,
        num_workers=2)

    eval_loader = get_dataloader(
        config.validate_file,
        format='json',
        tokenizer=model._tokenizer,
        construct_instance=ConstructInstanceFinetune,
        process_inputs=ProcessInputs,
        sample_weight=None,
        is_train=False,
        is_chinese=is_chinese,
        use_cache=config.use_cache,
        cache_dir=config.cache_dir,
        batch_size=config.batch_size,
        collate_fn=CollateFn,
        num_workers=2)

    with open(config.validate_file, 'r') as f:
        dev_data = json.load(f)
        dev_data = ConstructSingleQuoteInstance(dev_data,tokenizer=tokenizer)

    model.load_data("train", train_loader)
    model.load_data("test", eval_loader)

    train_steps = len(train_loader) * config.epoch
    warmup_steps = int(train_steps * config.warmup_proportion)

    model.get_optimizer(config.lr,
                        train_steps,
                        warmup_steps,
                        config.weight_decay,
                        config.epsilon)

    logger.info(f'train steps in each epoch:{len(model._dataset["train"])},'
                + f'total train steps:{train_steps}, warmup steps:{warmup_steps}')

    for epoch in range(config.epoch):
        logger.info(f"==>>> starting epoch [{epoch}]/[{config.epoch}]...")
        model.train_epoch(epoch=epoch, no_tqdm=True, accumulated_size=config.accumulated_size)
        results = model.test()
        records = ConstructRecords(eval_data=dev_data, results=results)
        scores = model.compute_score(records)

        wandb.log({'dev_acc': float(scores['acc'])})

        test_table = wandb.Table(columns=list(records[0].keys()))
        for record in random.sample(records, k=min(100, len(records))):
            test_table.add_data(*[str(record[k]) for k in record.keys()])
        wandb.log({'test_record': test_table})

        if (model._optimize_direction == 'min' and float(scores['acc']) <= model._best_eval_score) or (
                model._optimize_direction == 'max' and float(scores['acc']) >= model._best_eval_score):
            model._best_eval_score = float(scores['acc'])
            model.save_model(os.path.join(best_model_dir, "best_model.pt"))
            logger.info("==>>>Best model updated.")

    logger.info('==>>> finished training procedures...')


def test(config):
    assert config.checkpoint_dir != '', 'checkpoint directory should not be None.'
    assert os.path.exists(os.path.join(config.checkpoint_dir, 'best_model/best_model.pt')), 'checkpoint does not exist.'
    model = SpeakerModel(config.layer_num, model_mapping[config.model_name], 8,
                         hidden_num=hidden_size_mapping[config.model_name], tokenizer=tokenizer)
    if any([dataset_name in config.test_file for dataset_name in ['CSI', 'JY', 'WP2021']]):
        print('testing on Chinese data...')
    else:
        print('testing on English data...')
    model.to_device({'bart_encoder': 'cuda:0', 'char_encoder': 'cuda:0'})
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, 'best_model/best_model.pt')))

    model = MagicModel(
        model,
        tokenizer,
        cache_dir=config.cache_dir,
        loss_function=None,
        inference=Inference,
        compute_score=compute_accuracy,
        init_eval_score=None,
        optimize_direction=None)

    test_loader = get_dataloader(
        config.test_file,
        format='json',
        tokenizer=model._tokenizer,
        construct_instance=ConstructInstanceFinetune,
        process_inputs=ProcessInputs,
        sample_weight=None,
        is_train=False,
        is_chinese=is_chinese,
        use_cache=config.use_cache,
        cache_dir=config.cache_dir,
        batch_size=config.batch_size,
        collate_fn=CollateFn,
        num_workers=2)

    model.load_data("test", test_loader)

    with open(config.test_file, 'r') as f:
        test_data = json.load(f)
        test_data = ConstructSingleQuoteInstance(test_data,tokenizer=tokenizer)

    results = model.test(no_tqdm=False, inference_with_sampling=False)
    records = ConstructRecords(eval_data=test_data, results=results)
    scores = model.compute_score(records)

    wandb.log({'test_acc': float(scores['acc'])})

    test_table = wandb.Table(columns=list(records[0].keys()))
    for record in random.sample(records, k=min(100, len(records))):
        test_table.add_data(*[str(record[k]) for k in record.keys()])
    wandb.log({'test_record': test_table})

    eval_dir = os.path.join(config.checkpoint_dir, 'eval_log')
    os.makedirs(eval_dir, exist_ok=True)

    with open(os.path.join(eval_dir, 'result_records.json'), 'w') as f:
        json.dump(records, f, indent=2)

    with open(os.path.join(eval_dir, 'test_scores.json'), 'w') as f:
        json.dump(scores, f, indent=2)


model_mapping = {
    'bart-base-zh': 'fnlp/bart-base-chinese',
    'bart-base-en': 'facebook/bart-base',
    'bart-large-zh': 'fnlp/bart-large-chinese',
    'bart-large-en': 'facebook/bart-large',
    'bert-base-zh': 'bert-base-chinese',
    'bert-base-en': 'bert-base-cased',
    'bert-large-zh': 'yechen/bert-large-chinese',
    'bert-large-en': 'bert-large-cased',
    'roberta-base-en': 'roberta-base',
    'roberta-base-zh': 'hfl/chinese-roberta-wwm-ext',
    'roberta-large-en': 'roberta-large',
    'roberta-large-zh': 'hfl/chinese-roberta-wwm-ext-large',
    'gpt2-base-en': 'gpt2',
    'gpt2-base-zh': 'ckiplab/gpt2-base-chinese',
    'gpt2-large-en': 'gpt2-large',
    't5-base-en': 't5-base',
    't5-base-zh': 'lemon234071/t5-base-Chinese',
    't5-large-en': 't5-large'
}

hidden_size_mapping = {'bert-base-en': 768, 'gpt2-base-en': 768, 'bart-base-en': 768, 'bart-large-en': 1024,
                       'bert-base-zh': 768, 'gpt2-base-zh': 768, 'bart-base-zh': 768, 'bart-large-zh': 1024,
                       't5-base-en': 768, 'roberta-base-en': 768,
                       't5-base-zh': 768, 'roberta-base-zh': 768}
max_len_mapping = {'bert-base-en': 500, 'gpt2-base-en': 1000, 'bart-base-en': 1000, 'bart-large-en': 1000,
                   'bert-base-zh': 500, 'gpt2-base-zh': 1000, 'bart-base-zh': 1000, 'bart-large-zh': 1000,
                   't5-base-en': 1000, 'roberta-base-en': 500, 't5-base-zh': 1000, 'roberta-base-zh': 500}

if __name__ == '__main__':
    params = [
        Param(name='model_name', type=str, choices=list(model_mapping.keys())),
        Param(name='--optimize_direction', choices=['max', 'min'], default='max', help='max,min'),
        Param(name='--train_file', type=str, default=''),
        Param(name='--validate_file', type=str, default=''),
        Param(name='--test_file', type=str, default=''),
        Param(name='--checkpoint_dir', type=str, default=''),
        Param(name='--mode', choices=['train', 'test'], default='train', help='train,test'),
        Param(name='--layer_num', type=int, default=1),
        Param(name='--batch_size', type=int, default=16),
        Param(name='--accumulated_size',type=int,default=1),
        Param(name='--epoch', type=int, default=10),
        Param(name='--warmup_proportion', type=float, default=0.1),
        Param(name='--lr', type=float, default=5e-5),
        Param(name='--weight_decay', type=float, default=0),
        Param(name='--epsilon', type=float, default=1e-8),
        Param(name='--seed', type=int, default=43),
        Param(name='--use_cache', action='store_true', default=False),
        Param(name='--cache_dir', type=str, default='./cache')
    ]
    config = MagicParse(params)
    torch.manual_seed(config.seed)

    max_len = max_len_mapping[config.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_mapping[config.model_name],
                                              mask_token='[MASK]',
                                              pad_token='[PAD]',
                                              sep_token='[SEP]')
    if config.mode == 'train':
        assert config.train_file.strip(), 'In train mode, the train file should not be empty!'
        assert config.validate_file.strip(), 'In train mode, the dev file should not be empty!'
        dataset_names = list(filter(
            lambda x: x in config.train_file,
            ['CSI', 'JY', 'WP2021', 'PAP', 'RIQUA', 'PDNC_genre', 'PDNC_merge','SQLITE']))

    else:
        assert config.test_file.strip(), 'In test mode, the test file should not be empty!'
        dataset_names = list(filter(
            lambda x: x in config.test_file,
            ['CSI', 'JY', 'WP2021', 'PAP', 'RIQUA', 'PDNC_genre', 'PDNC_merge','SQLITE']))
    assert len(dataset_names) > 0, 'Dataset is invalid!'

    if dataset_names[0] in ['CSI', 'JY', 'WP2021']:
        is_chinese = True
        logger.info('running on Chinese data...')
    else:
        is_chinese=False
        logger.info('running on English data...')

    wandb_run = wandb.init(
        project=f"CBSI_{dataset_names[0]}_standard_new",
        name=f"{config.mode}_{config.model_name}_new2_lr({config.lr})_bs({config.batch_size*config.accumulated_size})_ep({config.epoch})_sd({config.seed})",
        config=config
    )
    if config.mode == 'test':
        test(config)
    else:
        main(config)
