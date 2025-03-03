# Speaker-alternation-pattern-based revision
import nltk
import copy,re
import numpy as np
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


class Prediction:
    def __init__(self, seg_sents, scores, candidate_ids, alias2ids, quote_id):
        self.seg_sents = seg_sents
        self.pred_speaker_id = candidate_ids[int(np.argmax(scores))]
        self.cdd_scores = {k: v for k, v in zip(candidate_ids, scores)}
        self.alias2ids = alias2ids
        self.quote_id = quote_id


p = re.compile('(“.*?”)|(".*?")|(``.*?\'\')')
def is_quote(sentence):
    return p.search(sentence) is not None


def cut_text(text):
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)

    # Combine consecutive proper nouns into a single token
    result_tokens = []
    current_name = ""
    for token, pos_tag in tagged_tokens:
        if pos_tag == 'NNP':
            current_name += token + " "
        else:
            if current_name:
                result_tokens.append(current_name.strip())
                current_name = ""
            result_tokens.append(token)

    if current_name:
        result_tokens.append(current_name.strip())

    return result_tokens

def match_sents(left_sents,left_center, right_sents,right_center):
    """Match two groups of sentences, return their distance
    if they have sentences in common.
    pos_relation: the relative positional relation between left_sents and right_sents
                  and has three values including "preceding", "same", "subsequent".
                  "preceding" refers to that the left sents are in the front of right sents,
                  "subsequent" refers to that the left sents are in the back of right sents,
                  "same" indicates that the left sents and the right sents are the same
    """
    rightmost = len(left_sents)

    for idx1, sent1 in enumerate(left_sents):
        for idx2, sent2 in enumerate(right_sents):
            if sent1 != sent2 or not sent1.strip() or not sent2.strip():
                continue
            #else:
            #    print(f'sent1:{sent1},idx1:{idx1},idx2:{idx2}')

            dist = idx1 + (right_center-idx2) - left_center
            if dist<0:
                continue

            bad = False
            for i in range(idx1, rightmost):
                if idx2+i-idx1>=len(right_sents) or left_sents[i] != right_sents[idx2+i-idx1]:
                    bad = True
                    break

            if bad:
                continue

            #print(f'left:{left_sents[idx1:rightmost]}, right:{right_sents[idx2:idx2+rightmost-idx1]}')
            #print(f'left center:{left_center},{len(left_sents)}, right center:{right_center},{len(right_sents)}')
            #print(f'left center:{left_sents[left_center]},right center:{right_sents[right_center]}')

            return dist

    return None



def retrieve_continuous_quotes(pred_list, is_chinese):
    """
    cluster quotes into groups. The quotes within each group are connected by their context window
    ctn_quotes_list = [[[q1,0],[q2,dis_12],[q3,dis_13],...],[[qi,0],[qi+1,dis_ii+1],...]]
    starting from a quote qi, if group is empty and then add itself to group=[[qi,0]]. first check whether the quote of the next instance is connected
    with the quote of the current instance by a sequence of consecutive quotes.
    if connected, compute the distance between the two quotes and then add the new quote to group=[[qi,0],[qi+1,dis_ii+1]]
    if not connected, if the group contains more than one quote and then add the group to ctn_quotes_list. finally start a new group=[]
    Then starting from the next quote qi+1, and repeat the above process again.
	"""
    def get_start_end(sents):
        start_idx = 0
        end_idx = len(sents)-1
        for sent_idx in range(len(sents)):
            if sents[sent_idx].strip():
                break
        start_idx = sent_idx
        for sent_idx in range(len(sents)):
            if sents[end_idx-sent_idx].strip():
                break
        end_idx = end_idx-sent_idx
        return start_idx,end_idx

    #ws = (len(pred_list[0].seg_sents) - 1) // 2

    connecting = False
    ctn_quotes_list = []
    for i, pred in enumerate(pred_list[:-1]):
        ws = (len(pred.seg_sents) - 1) // 2
        this_inst = [''.join(x) if is_chinese else tokenizer.convert_tokens_to_string(x) for x in pred.seg_sents]
        next_inst = [''.join(x) if is_chinese else tokenizer.convert_tokens_to_string(x) for x in pred_list[i + 1].seg_sents]
        this_start, this_end = get_start_end(this_inst)
        next_start, next_end = get_start_end(next_inst)
        this_center = (len(this_inst) - 1) // 2 - this_start
        next_center = (len(next_inst)-1)//2 - next_start
        #print(f'this start:{(this_start, this_end,len(this_inst))}, next_start:{(next_start,next_end,len(next_inst))}')



        # not in connecting mode, start a new connection
        if not connecting:
            # each element in inst_id_quote_id: (the index of instance in the dataset, the index of the center quote of the instance within this group of continuous quotes)
            inst_id_quote_id = [(i, 0)]
            connecting = True

        # compute the number of continuous quotes on the right
        n_ctn_quotes = 0
        for sent in this_inst[ws + 1:]:
            if is_quote(sent):
                n_ctn_quotes += 1
                #print(f'sent:{sent} is quote')
            else:
                #print(f'sent:{sent} is not quote')
                break

        # no more adjacent quotes
        if n_ctn_quotes == 0:
            if len(inst_id_quote_id) > 1:
                ctn_quotes_list.append(inst_id_quote_id)
            connecting = False
            continue

        # compute the distance between this_inst and next_inst
        #distance = ws + 1
        dist = match_sents(this_inst[this_start:this_end+1],this_center,next_inst[next_start:next_end+1],next_center)

        if dist is not None and dist <= n_ctn_quotes:
            inst_id_quote_id.append((i + 1, inst_id_quote_id[-1][1] + dist))
        else:
            if len(inst_id_quote_id) > 1:
                ctn_quotes_list.append(inst_id_quote_id)
            connecting = False

    if len(inst_id_quote_id) > 1:
        ctn_quotes_list.append(inst_id_quote_id)
    #print(f'ctn_quotes_list:{ctn_quotes_list}')

    return ctn_quotes_list


def dominant_speakers(seg_ctx, alias2id, th, is_chinese):
    """
    find out the most frequent two mentions within the context
    first count the frequency of each mention, and rank the mentions.
    Finally select the top two mentions
	"""
    char_freq = {}
    if not is_chinese:
        seg_text = tokenizer.convert_tokens_to_string(seg_ctx)
        seg_ctx = cut_text(seg_text)

    for word in seg_ctx:
        if word not in alias2id:
            continue

        if alias2id[word] in char_freq:
            char_freq[alias2id[word]] += 1
        else:
            char_freq[alias2id[word]] = 1

    sorted_char_freq = sorted(char_freq.items(), key=lambda x: -x[1])
    if len(sorted_char_freq) < 2:
        return None

    c1 = sorted_char_freq[0][0]
    c2 = sorted_char_freq[1][0]

    if len(sorted_char_freq) == 2:
        return (c1, c2)
    else:
        fc2 = sorted_char_freq[1][1]
        fc3 = sorted_char_freq[2][1]
        if fc2 >= fc3 + th:
            return (c1, c2)
        else:
            return None


def pred_cfd(pred, speakers):
    """
    infer the speaker according to the prediction scores
    """
    c1, c2 = speakers
    cdd2score = pred.cdd_scores
    c1_score = -1 if c1 not in cdd2score else cdd2score[c1]
    c2_score = -1 if c2 not in cdd2score else cdd2score[c2]
    pred_spk = c1 if c1_score > c2_score else c2
    score_diff = c1_score - c2_score
    pred_cfd = score_diff if score_diff >= 0 else -score_diff
    return pred_spk, pred_cfd


def sap_figure_out_speaker(ctn_quotes, side, spk_idx):
    if side == 'left':
        # sap-revision start from the left
        prev_spk_idx = spk_idx
    else:
        prev_spk_idx = spk_idx if ctn_quotes[-1][1] % 2 == 0 else (1 - spk_idx)

    inst2spk_idx = {ctn_quotes[0][0]: prev_spk_idx}
    prev_quote_idx = 0
    for inst_idx, quote_idx_in_cvs in ctn_quotes[1:]:
        # sap-revision start from the right
        if (quote_idx_in_cvs - prev_quote_idx) % 2 == 0:
            inst2spk_idx[inst_idx] = prev_spk_idx
        else:
            inst2spk_idx[inst_idx] = 1 - prev_spk_idx
            prev_spk_idx = 1 - prev_spk_idx
        prev_quote_idx = quote_idx_in_cvs

    return inst2spk_idx


def merge_alias2ids(alias2ids_batch):
    # make sure that the alias2ids comes from the original instance
    new_alias2ids = {}
    for alias2ids in alias2ids_batch:
        for name,name_id in alias2ids.items():
            if name not in new_alias2ids:
                new_alias2ids[name] = name_id
            else:
                assert new_alias2ids[name] == name_id, "unequal name id"
    #print(f'alias2id_batch:{alias2ids_batch}')
    #print(f'merged_alias2id:{new_alias2ids}')
    return new_alias2ids

def sap_rev(pred_list, th, is_chinese):
    """
	Speaker-alternation-pattern-based revision (SAPR)

	params
		pred_list: a list of Prediction object.

	return
		rev_dict: {instance index: revised speaker id}
	"""
    #ws = (len(pred_list[0].seg_sents) - 1) // 2

    ctn_quotes_list = retrieve_continuous_quotes(pred_list, is_chinese=is_chinese)

    #name_list_path = '/home/ychen/183/codes_and_scripts/IdentifySpeaker/CSN_SAPR/data/name_list.txt'
    #with open(name_list_path, 'r', encoding='utf-8') as fin:
    #    name_lines = fin.readlines()
    #id2alias = []
    #for i, line in enumerate(name_lines):
    #    id2alias.append(line.strip().split()[1])


    rev_dict = {}
    for ctn_quotes in ctn_quotes_list:
        # run sap-rev for each conversation

        left_inst_idx, _ = ctn_quotes[0]
        right_inst_idx, quote_idx = ctn_quotes[-1]

        left_pred = pred_list[left_inst_idx]
        right_pred = pred_list[right_inst_idx]

        left_ws = (len(left_pred.seg_sents) - 1) // 2
        right_ws = (len(right_pred.seg_sents) - 1) // 2

        seg_left_ctx_sents = left_pred.seg_sents[:left_ws]
        seg_right_ctx_sents = right_pred.seg_sents[-right_ws:]

        seg_ctx = [y for x in seg_left_ctx_sents + seg_right_ctx_sents for y in x]

        alias2ids = merge_alias2ids(alias2ids_batch=list(map(lambda x:x.alias2ids,pred_list[left_inst_idx:right_inst_idx+1])))

        # two dominant speaker ids
        speakers = dominant_speakers(seg_ctx,alias2ids, th, is_chinese=is_chinese)
        #print(f'dominant speakers:{speakers}')
        if not speakers:
            continue

        left_pred_spk, left_cfd = pred_cfd(left_pred, speakers)
        right_pred_spk, right_cfd = pred_cfd(right_pred, speakers)

        # 		for seg_sent in left_pred.seg_sents:
        # 			print(''.join(seg_sent))
        # 		print('Left pred:', id2alias[left_pred_spk], 'Cfd:', left_cfd)
        # 		print('---SEP---')
        # 		for seg_sent in right_pred.seg_sents:
        # 			print(''.join(seg_sent))
        # 		print('Right pred:', id2alias[right_pred_spk], 'Cfd:', right_cfd)
        # 		print([id2alias[x] for x in speakers])
        # 		print('----------------------------------------------')

        side = 'left' if left_cfd > right_cfd else 'right'
        spk_id = left_pred_spk if left_cfd > right_cfd else right_pred_spk
        sap_rev_res = sap_figure_out_speaker(ctn_quotes, side, speakers.index(spk_id))

        rev_dict.update({k: speakers[v] for k, v in sap_rev_res.items()})

    return rev_dict