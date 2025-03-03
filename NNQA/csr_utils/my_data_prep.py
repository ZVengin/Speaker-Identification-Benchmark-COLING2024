import random
from urllib.parse import quote
# Data pre-processing and data loader generation.

import copy
import os,sys
import re
import logging
import jieba
import json
import wandb
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from collections import defaultdict

from spacy.lang.en import English
from spacy.lang.zh import Chinese


SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIP_DIR,'./../..'))
from utils import ConstructSingleQuoteInstance,cut_sentence_with_quotation_marks

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

sent_tokenizer_en = English()
sent_tokenizer_zh = Chinese()
sent_tokenizer_en.add_pipe("sentencizer")
sent_tokenizer_zh.add_pipe("sentencizer")

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
zh_tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')




def clean_quote(text):
    p = re.compile('(“.*?”)|(".*?")')
    for i in p.finditer(text):
        start = i.start()
        end = i.end()
        temp = ''
        for k in range(start, end):
            temp += text[k]
        if temp.strip() != '':
            quote = temp
            return quote
    return None

def NML_zh(seg_sents, mention_positions, ws):
    """
    Nearest Mention Location

    params
        seg_sents: segmented sentences of an instance in a list.
            [[word 1,...] of sentence 1,...].
        mention_positions: the positions of mentions of a candidate.
            [[sentence-level index, word-level index] of mention 1,...].
        ws: single-sided context window size.

    return
        The position of the mention which is the nearest to the quote.
    """
    def word_dist(pos):
        """
        The word level distance between quote and the mention position

        param
            pos: [sentence-level index, word-level index] of the character mention.

        return
            w_d: word-level distance between the mention and the quote.
        """
        if pos[0] == ws:
            w_d = ws * 2
        elif pos[0] < ws:
            w_d = sum(len(sent) for sent in seg_sents[pos[0] + 1:ws]) + len(seg_sents[pos[0]][pos[1] + 1:])
        else:
            w_d = sum(len(sent) for sent in seg_sents[ws + 1:pos[0]]) + len(seg_sents[pos[0]][:pos[1]])
        return w_d

    sorted_positions = sorted(mention_positions, key=lambda x: word_dist(x))

    # trick
    if len(seg_sents[ws-1])>0 and seg_sents[ws - 1][-1] == '：':
        # if the preceding sentence ends with '：'
        for pos in sorted_positions:
            # search candidate mention from left-side context
            if pos[0] < ws:
                return pos

    return sorted_positions[0]



def seg_and_mention_location_zh(raw_sents_in_list, alias2id):
    """
    Chinese word segmentation and candidate mention location.

    params
        raw_sents_in_list: unsegmented sentences of an instance in a list.
        alias2id: a dict mapping character alias to its ID.

    return
        seg_sents: segmented sentences of the input instance.
        character_mention_poses: a dict mapping the index of a candidate to its mention positions.
            {character index: [[sentence index, word index in sentence] of mention 1,...]...}.
    """
    character_mention_poses = {}
    seg_sents = []
    #print("find the mention location:")
    for sent_idx, sent in enumerate(raw_sents_in_list):
        seg_sent = list(jieba.cut(sent, cut_all=False))
        seg_sent = [tok for tok in seg_sent if tok.strip()]
        for word_idx, word in enumerate(seg_sent):
            if word in alias2id:
                if alias2id[word] in character_mention_poses:
                    character_mention_poses[alias2id[word]].append([sent_idx, word_idx])
                else:
                    character_mention_poses[alias2id[word]] = [[sent_idx, word_idx]]

        seg_sents.append(seg_sent)
    id2alias = defaultdict(list)
    for k,v in alias2id.items():
        id2alias[v].append(k)

    #print(f"character mention positions:{[(id2alias[k],v) for k,v in character_mention_poses.items()]}")
    return seg_sents, character_mention_poses



def create_CSS_zh(seg_sents, candidate_mention_poses, ws, max_len):
    """
    Create candidate-specific segments for each candidate in an instance.

    params
        seg_sents: 2ws + 1 segmented sentences in a list.
        candidate_mention_poses: a dict which contains the position of candiate mentions,
            with format {character index: [[sentence index, word index in sentence] of mention 1,...]...}.
        ws: single-sided context window size.
        max_len: maximum length limit.

    return
        Returned contents are in lists, in which each element corresponds to a candidate.
        The order of candidate is consistent with that in list(candidate_mention_poses.keys()).
        many_CSS: candidate-specific segments.
        many_sent_char_len: segmentation information of candidate-specific segments.
            [[character-level length of sentence 1,...] of the CSS of candidate 1,...].
        many_mention_pos: the position of the nearest mention in CSS.
            [(sentence-level index of nearest mention in CSS,
             character-level index of the leftmost character of nearest mention in CSS,
             character-level index of the rightmost character + 1) of candidate 1,...].
        many_quote_idx: the sentence-level index of quote sentence in CSS.

    """

    assert len(seg_sents) == ws * 2 + 1

    def max_len_cut(seg_sents, mention_pos):
        """
        Cut the CSS of each candidate to fit the maximum length limitation.

        params
            seg_sents: the segmented sentences involved in the CSS of a candidate.
            mention_pos: the position of the mention of the candidate in the CSS.

        return
            seg_sents: ... after truncated.
            mention_pos: ... after truncated.
        """
        sent_char_lens = [sum(len(word) for word in sent) for sent in seg_sents]
        sum_char_len = sum(sent_char_lens)

        running_cut_idx = [len(sent) - 1 for sent in seg_sents]

        while sum_char_len > max_len:
            max_len_sent_idx = max(list(enumerate(sent_char_lens)), key=lambda x: x[1])[0]

            if max_len_sent_idx == mention_pos[0] and running_cut_idx[max_len_sent_idx] == mention_pos[1]:
                running_cut_idx[max_len_sent_idx] -= 1

            if max_len_sent_idx == mention_pos[0] and running_cut_idx[max_len_sent_idx] < mention_pos[1]:
                mention_pos[1] -= 1

            reduced_char_len = len(seg_sents[max_len_sent_idx][running_cut_idx[max_len_sent_idx]])
            sent_char_lens[max_len_sent_idx] -= reduced_char_len
            sum_char_len -= reduced_char_len

            del seg_sents[max_len_sent_idx][running_cut_idx[max_len_sent_idx]]

            running_cut_idx[max_len_sent_idx] -= 1

        return seg_sents, mention_pos

    many_CSSs = []
    many_CSSs_Tokens = []
    many_sent_char_lens = []
    many_mention_poses = []
    many_quote_idxes = []

    for candidate_idx in candidate_mention_poses.keys():

        nearest_pos = NML_zh(seg_sents, candidate_mention_poses[candidate_idx], ws)
        if nearest_pos[0] <= ws:
            CSS = copy.deepcopy(seg_sents[nearest_pos[0]:ws + 1])
            mention_pos = [0, nearest_pos[1]]
            quote_idx = ws - nearest_pos[0]
        else:
            CSS = copy.deepcopy(seg_sents[ws:nearest_pos[0] + 1])
            mention_pos = [nearest_pos[0] - ws, nearest_pos[1]]
            quote_idx = 0
            
        #logger.info(f'smention_pos:{mention_pos}')
        cut_CSS, mention_pos = max_len_cut(CSS, mention_pos)

        sent_char_lens = [sum(len(word) for word in sent) for sent in cut_CSS]

        mention_pos_left = sum(sent_char_lens[:mention_pos[0]]) + sum(len(x) for x in cut_CSS[mention_pos[0]][:mention_pos[1]])
        mention_pos_right = mention_pos_left + len(cut_CSS[mention_pos[0]][mention_pos[1]])
        mention_pos = (mention_pos[0], mention_pos_left, mention_pos_right)
        #if sum(sent_char_lens)==mention_pos_left:
        #    logger.info(f'mention_pos:{mention_pos}')
        #    logger.info(f'sgement_length:{sum(sent_char_lens)}')
        cat_CSS = ''.join([''.join(sent) for sent in cut_CSS])

        many_CSSs.append(cat_CSS)
        many_CSSs_Tokens.append(zh_tokenizer.tokenize(cat_CSS))
        many_sent_char_lens.append(sent_char_lens)
        many_mention_poses.append(mention_pos)
        many_quote_idxes.append(quote_idx)

    #print(f"CSSs:{list([(css[pos[1]:pos[2]],css) for pos,css in zip(many_mention_poses,many_CSSs)])}")

    return many_CSSs,many_CSSs_Tokens, many_sent_char_lens, many_mention_poses, many_quote_idxes






def NML_en(seg_sents, mention_positions, ws):
    """
    Nearest Mention Location

    params
        seg_sents: segmented sentences of an instance in a list.
            [[word 1,...] of sentence 1,...].
        mention_positions: the positions of mentions of a candidate.
            [[sentence-level index, word-level index] of mention 1,...].
        ws: single-sided context window size.

    return
        The position of the mention which is the nearest to the quote.
    """
    def word_dist(pos):
        """
        The word level distance between quote and the mention position

        param
            pos: [sentence-level index, word-level index] of the character mention.

        return
            w_d: word-level distance between the mention and the quote.
        """
        if pos[0] == ws:
            w_d = ws * 2
        elif pos[0] < ws:
            w_d = sum(len(sent) for sent in seg_sents[pos[0] + 1:ws]) + len(seg_sents[pos[0]][pos[2]+1:])
        else:
            w_d = sum(len(sent) for sent in seg_sents[ws + 1:pos[0]]) + len(seg_sents[pos[0]][:pos[1]])
        return w_d

    sorted_positions = sorted(mention_positions, key=lambda x: word_dist(x))

    #logger.info(f'seg_sents:{seg_sents}')
    # trick
    if len(seg_sents[ws - 1])>0 and seg_sents[ws - 1][-1] == ':':
        # if the preceding sentence ends with '：'
        for pos in sorted_positions:
            # search candidate mention from left-side context
            if pos[0] < ws:
                return pos

    return sorted_positions[0]



def IsAlias(tokens, alias2id, start_index):
    mentions = []
    for alias in alias2id.keys():
        i = start_index
        while (i<len(tokens)):
            substr = tokenizer.convert_tokens_to_string(tokens[start_index:i+1])
            if alias.startswith(substr):
                i +=1
            else:
                break
        if i!=start_index and alias == tokenizer.convert_tokens_to_string(tokens[start_index:i]):
            mentions.append([alias,start_index,i-1])

    if len(mentions)>0:
        mentions = sorted(mentions,key=lambda x:len(x[0]),reverse=True)
        return mentions[0]
    else:
        return [None]*3



def seg_and_mention_location_en(raw_sents_in_list, alias2id):
    """
    Chinese word segmentation and candidate mention location.

    params
        raw_sents_in_list: unsegmented sentences of an instance in a list.
        alias2id: a dict mapping character alias to its ID.

    return
        seg_sents: segmented sentences of the input instance. e.g. [['I','like','it'],['how','about','you']]
        character_mention_poses: a dict mapping the index of a candidate to its mention positions.
            {character index: [[sentence index, word index in sentence] of mention 1,...]...}.
    """
    character_mention_poses = {}
    seg_sents = []
    for sent_idx, sent in enumerate(raw_sents_in_list):
        seg_sent =  tokenizer.tokenize(sent)
        word_idx = 0
        while (word_idx<len(seg_sent)):
            alias, start_idx, end_idx = IsAlias(seg_sent,alias2id,word_idx)
            if alias != None:
                if alias2id[alias] in character_mention_poses:
                    character_mention_poses[alias2id[alias]].append([sent_idx, start_idx,end_idx])
                else:
                    character_mention_poses[alias2id[alias]] = [[sent_idx, start_idx, end_idx]]
                word_idx = end_idx+1
            else:
                word_idx +=1
        seg_sents.append(seg_sent)
    return seg_sents, character_mention_poses


def create_CSS_en(seg_sents, candidate_mention_poses, ws, max_len):
    """
    Create candidate-specific segments for each candidate in an instance.

    params
        seg_sents: 2ws + 1 segmented sentences in a list.
        candidate_mention_poses: a dict which contains the position of candiate mentions,
            with format {character index: [[sentence index, word index in sentence] of mention 1,...]...}.
        ws: single-sided context window size.
        max_len: maximum length limit.

    return
        Returned contents are in lists, in which each element corresponds to a candidate.
        The order of candidate is consistent with that in list(candidate_mention_poses.keys()).
        many_CSS: candidate-specific segments.
        many_sent_char_len: segmentation information of candidate-specific segments.
            [[character-level length of sentence 1,...] of the CSS of candidate 1,...].
        many_mention_pos: the position of the nearest mention in CSS.
            [(sentence-level index of nearest mention in CSS,
             character-level index of the leftmost character of nearest mention in CSS,
             character-level index of the rightmost character + 1) of candidate 1,...].
        many_quote_idx: the sentence-level index of quote sentence in CSS.

    """
    
    #logger.info(f'# sent number: {len(seg_sents)}, # window size:{ws}')
    assert len(seg_sents) == ws * 2 + 1

    def max_len_cut(seg_sents, mention_pos):
        """
        Cut the CSS of each candidate to fit the maximum length limitation.

        params
            seg_sents: the segmented sentences involved in the CSS of a candidate.
            mention_pos: the position of the mention of the candidate in the CSS.

        return
            seg_sents: ... after truncated.
            mention_pos: ... after truncated.
        """
        sent_char_lens = [len(sent) for sent in seg_sents]
        sum_char_len = sum(sent_char_lens)

        running_cut_idx = [len(sent) - 1 for sent in seg_sents]

        while sum_char_len > max_len:
            max_len_sent_idx = max(list(enumerate(sent_char_lens)), key=lambda x: x[1])[0]

            if max_len_sent_idx == mention_pos[0] and running_cut_idx[max_len_sent_idx] == mention_pos[2]:
                running_cut_idx[max_len_sent_idx] = mention_pos[1]-1

            if max_len_sent_idx == mention_pos[0] and running_cut_idx[max_len_sent_idx] < mention_pos[1]:
                mention_pos[1] -= 1
                mention_pos[2] -= 1

            #reduced_char_len = len(seg_sents[max_len_sent_idx][running_cut_idx[max_len_sent_idx]])
            sent_char_lens[max_len_sent_idx] -= 1
            sum_char_len -= 1

            del seg_sents[max_len_sent_idx][running_cut_idx[max_len_sent_idx]]

            running_cut_idx[max_len_sent_idx] -= 1

        return seg_sents, mention_pos

    many_CSSs = []
    many_CSSs_Tokens = []
    many_sent_char_lens = []
    many_mention_poses = []
    many_quote_idxes = []

    for candidate_idx in candidate_mention_poses.keys():

        nearest_pos = NML_en(seg_sents, candidate_mention_poses[candidate_idx], ws)
        if nearest_pos[0] <= ws:
            CSS = copy.deepcopy(seg_sents[nearest_pos[0]:ws + 1])
            mention_pos = [0, nearest_pos[1],nearest_pos[2]]
            quote_idx = ws - nearest_pos[0]
        else:
            CSS = copy.deepcopy(seg_sents[ws:nearest_pos[0] + 1])
            mention_pos = [nearest_pos[0] - ws, nearest_pos[1],nearest_pos[2]]
            quote_idx = 0

        cut_CSS, mention_pos = max_len_cut(CSS, mention_pos)

        sent_char_lens = [len(sent) for sent in cut_CSS]

        mention_pos_left = sum(sent_char_lens[:mention_pos[0]]) +  len(cut_CSS[mention_pos[0]][:mention_pos[1]])
        mention_pos_right = mention_pos_left + len(cut_CSS[mention_pos[0]][mention_pos[1]:mention_pos[2]+1])
        mention_pos = (mention_pos[0], mention_pos_left, mention_pos_right)
        #cat_CSS = ' '.join([tokenizer.convert_tokens_to_string(sent) for sent in cut_CSS])
        cat_CSS = tokenizer.convert_tokens_to_string(sum(cut_CSS,[]))

        many_CSSs.append(cat_CSS)
        many_CSSs_Tokens.append(sum(cut_CSS,[]))
        many_sent_char_lens.append(sent_char_lens)
        many_mention_poses.append(mention_pos)
        many_quote_idxes.append(quote_idx)

    return many_CSSs, many_CSSs_Tokens,many_sent_char_lens, many_mention_poses, many_quote_idxes


class ISDataset(Dataset):
    """
    Dataset subclass for Identifying speaker.
    """
    def __init__(self, data_list):
        super(ISDataset, self).__init__()
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_data_loader(data_file, batch_size, max_len, skip_only_one=False, use_cache=False):
    """
    Build the dataloader for training.

    Input:
        data_file: labelled training data as in https://github.com/YueChenkkk/Chinese-Dataset-Speaker-Identification.
        name_list_path: the path of the name list which contains the aliases of characters.
        args: parsed arguments.
        skip_only_one: a flag for filtering out the instances that have only one candidate, such
            instances have no effect while training.

    Output:
        A torch.csr_utils.data.DataLoader object which generates:
            raw_sents_in_list: the raw (unsegmented) sentences of the instance.
                [sentence -ws, ..., qs, ..., sentence ws].
            CSSs: candidate-specific segments for candidates.
                [CSS of candidate 1,...].
            sent_char_lens: the character length of each sentence in the instance.
                [[character-level length of sentence 1,...] in the CSS of candidate 1,...].
            mention_poses: positions of mentions in the concatenated sentences.
                [(sentence-level index of nearest mention in CSS,
                 character-level index of the leftmost character of nearest mention in CSS,
                 character-level index of the rightmost character + 1) of candidate 1,...]
            quote_idxes: quote index in CSS of candidates in list.
            one_hot_label: one-hot label of the true speaker on list(mention_poses.keys()).
            true_index: index of the speaker on list(mention_poses.keys()).
    """
    if use_cache and os.path.exists(f'{data_file}.cache'):
        logger.info('Using the cache data...')
        with open(f'{data_file}.cache','r') as f:
            data_list = json.load(f)
        data_list = batchfy_dataset(data_list,batch_size=batch_size)
        return data_list



    is_chinese = False
    if any([dataset_name in data_file for dataset_name in ['CSI','WP2021','JY']]):
        logger.info("process chinese dataset...")
        is_chinese = True

    with open(data_file,'r') as f:
        data = json.load(f)


    data = sorted(data,key=lambda x:x['id'])
    single_quote_insts = ConstructSingleQuoteInstance(data, tokenizer=zh_tokenizer if is_chinese else tokenizer)
    if is_chinese:
        names = []
        for inst in single_quote_insts:
            char_dict = inst['character']
            names += list(char_dict['name2id'].keys())
        names = list(set(names))
        for name in names:
            jieba.add_word(name)

    logger.info(f'# of instances: {len(single_quote_insts)}')

    data_list=[]
    for inst in single_quote_insts:
        alias2id = inst['character']['name2id']
        for quote_index in range(len(inst['dialogue'][0]['utterance'])):
            quote = inst['dialogue'][0]['utterance'][quote_index]['quote']
            qid = inst['dialogue'][0]['utterance'][quote_index]['quote_id']
            speaker_name = inst['dialogue'][0]['utterance'][quote_index]['speaker']
            if not quote.strip():
                continue
            if speaker_name not in alias2id:
                print(f'unknown speaker:{speaker_name}')
                continue
            preceding_context = ' '.join(map(lambda x:x['paragraph'],inst['preceding_paragraphs']))
            succeeding_context = ' '.join(map(lambda x:x['paragraph'],inst['succeeding_paragraphs']))
            dialogue_text = inst['dialogue'][0]['paragraph']

            preceding_sents = cut_sentence_with_quotation_marks(preceding_context,is_chinese)
            succeeding_sents = cut_sentence_with_quotation_marks(succeeding_context,is_chinese)
            dialogue_sents = cut_sentence_with_quotation_marks(dialogue_text,is_chinese)

            preceding_sents = list(map(lambda x:x['sentence'],preceding_sents))
            succeeding_sents = list(map(lambda x:x['sentence'],succeeding_sents))
            dialogue_sents = list(map(lambda x:x['sentence'],dialogue_sents))

            for sent_index,sent in enumerate(dialogue_sents):
                quote_pos = -1
                if sent.find(quote.strip()) != -1:
                    quote_pos = sent_index
                    break
            if quote_pos == -1:
                logger.info(f'unable to find the quote in the dialogue sents:\nquote:{quote}\n{dialogue_sents}')
                continue
            preceding_sents = preceding_sents + dialogue_sents[:quote_pos]
            succeeding_sents = dialogue_sents[quote_pos+1:] + succeeding_sents

            max_sent_num = max(len(preceding_sents),len(succeeding_sents))
            preceding_sents = ['']*(max_sent_num-len(preceding_sents)) + preceding_sents
            succeeding_sents = succeeding_sents + ['']*(max_sent_num-len(succeeding_sents))

            raw_sents = preceding_sents + dialogue_sents[quote_pos:quote_pos+1] + succeeding_sents

            quote_sent_idx = len(raw_sents) // 2

            if is_chinese:
                seg_sents, candidate_mention_poses = seg_and_mention_location_zh(
                    raw_sents,
                    alias2id)
            else:
                seg_sents, candidate_mention_poses = seg_and_mention_location_en(
                    raw_sents,
                    alias2id)
            if (skip_only_one and len(candidate_mention_poses) == 1) or len(candidate_mention_poses) == 0:
                # logger.info(f'candidate_mention_poses:{candidate_mention_poses}')
                continue

            if is_chinese:
                CSSs, CSSs_tokens, sent_char_lens, mention_poses, quote_idxes = create_CSS_zh(
                    seg_sents,
                    candidate_mention_poses,
                    quote_sent_idx,
                    max_len)
            else:
                CSSs, CSSs_tokens, sent_char_lens, mention_poses, quote_idxes = create_CSS_en(
                    seg_sents,
                    candidate_mention_poses,
                    quote_sent_idx,
                    max_len)
            one_hot_label = [0 if character_idx != alias2id[speaker_name] else 1
                             for character_idx in candidate_mention_poses.keys()]
            true_index = one_hot_label.index(1) if 1 in one_hot_label else -1
            category = 'None'
            data_list.append([qid, seg_sents, CSSs, CSSs_tokens, sent_char_lens, mention_poses,
                              quote_idxes, one_hot_label, true_index, category,alias2id])

    with open(f'{data_file}.cache', 'w') as f:
        json.dump(data_list, f, indent=2)
    logger.info(f'# of constructed instances:{len(data_list)}')

    samples = random.sample(data_list, k=min(20,len(data_list)))
    sample_table = wandb.Table(
        columns=['qid', 'seg_sents', 'CSSs', 'CSSs_tokens', 'mention_poses', 'quote_idxes', 'true_index'])
    for sample in samples:
        sample_table.add_data(
            *[str(sample[0]),
              str(sample[1]),
              str(sample[2]),
              str(sample[3]),
              str(sample[5]),
              str(sample[6]),
              str(sample[8])]
        )
    wandb.log({'features': sample_table})

    data_list = batchfy_dataset(data_list, batch_size)

    return data_list


def collate_fn(batch):
    batch = list(zip(*batch))
    batch = [list(e) for e in batch]
    #print(f"batch:{batch}")
    return batch


def batchfy_dataset(examples, batch_size):
    batches = []
    max_size = batch_size
    start,end =0,1
    while  end<=len(examples):
        batch = examples[start:end]
        css_counter = sum([len(e[1]) for e in batch])
        if css_counter <= max_size:
            end += 1
        else:
            #print(f"css_counter:{css_counter}")
            end -= 1
            if end >start:
                batch = examples[start:end]
            else:
                end = start + 1
                batch = examples[start:end]
            batch = [list(e) for e in zip(*batch)]
            if len(batch) > 0:
                batches.append(batch)
            start = end
            end = start + 1
    if end>start:
        batch = examples[start:end]
        batch = [list(e) for e in zip(*batch)]
        if len(batch)>0:
            batches.append(batch)
    return batches