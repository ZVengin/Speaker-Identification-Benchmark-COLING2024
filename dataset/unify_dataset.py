# convert the format of different datasets into the uniform format
# the uniform format for each instance:
# {
#     "preceding_paragraphs": the paragraph before the dialogue and does not contain quotes,
#     "dialogue": a sequence of paragraphs containing quotes,
#     "succeeding_paragraphs": the paragraph after the dialogue,
#     "character": the dictionary of character name and its aliases,
#     "id": the index of instance
# }
# the uniform format for each paragraph:
# {
#    "paragraph": the text of paragraph (string type),
#    "paragraph_index" the index of paragraph (string type) in the format of '{book_name}-{para_index}',
#    "offset": the offset of the paragraph (tuple type),
#    "utterance_span": the offset of the utterances inside the paragraph (list of tuples),
#    "utterance": the utterances inside the paragraph (list of string),
#    "speaker": the speaker for the utterances (list of string),
#    "utterance_id": the index of utterances (list of string), each index is in the format of
#                    '{book_name}-{para_index}-{utter_index}'
#    "utterance_type": the type of utterances (list of string),
#    "book_name": the name of the book,
#    "mode": the mode of the paragraph,
# }
# the uniform format for each paragraph:
# {
#    "paragraph": the text of paragraph (string type),
#    "paragraph_index" the index of paragraph (string type) in the format of '{book_name}-{para_index}',
#    "offset": the offset of the paragraph (tuple type),
#    "utterance": the list of utterance objects (list of dictionary)
#    "book_name": the name of the book,
#    "mode": the mode of the paragraph,
# }
# the uniform format for utterance object
# {
#   "text":"the quote text",
#   "quote_span":"the position of each quote within each paragraph",
#   "speaker":"the speaker name for each quote",
#   "quote_id":"the unique id for each quote",
#   "quote_type":"the quote type of each quote (Implicit, Explicit, Anaphoric(other), Anaphoric(pronoun))",
# }


"""
Todo 1: Check the ClusterQuote function. The quotes
        cannot be clustered correctly

Todo 2: SAPR uses alternative speaking pattern
Todo 3: verify the classification of utterances
"""
import copy
import json
import os,sys
import re
import pickle
import random
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR,'..'))

import spacy
from collections import defaultdict,Counter
from tqdm import tqdm
from utils import cut_sentence_with_quotation_marks, find_words_with_non_alpha_boundaries,CharMaskId,GetSubCharDict,AddUnkToken


nlp_zh = spacy.load("zh_core_web_sm")
nlp_en = spacy.load('en_core_web_sm')

chinses_pronoun_list=['我','你','她','他','它','我们','你们','她们','他们','它们']
english_pronoun_list = ['i', 'you', 'he', 'she', 'it', 'they', 'we']
chinses_speech_tags =['说','道','问', '叹' , '诉','劝','叫','吼','哭','喊','想','谈','乞求','介绍','以为','反对','反驳','吱唔','告别','告诉','呻吟','呼喊',
                      '命令','咄呐','哽咽','喃喃','嘟囔','嗫嚅','嘱咐','嘲笑','回应','回答','央求','夸','安慰','宣布','帮腔',
                      '应承','建议','开口','开玩笑','张口结舌','强调','心想','恭维','惊讶','感激','打招呼','抱怨','拒绝','拥护',
                      '指','指着','挖苦','提醒','插嘴','撒谎','沉吟','破口大骂','禀告','笑','笑了','笑笑','补充','表扬','表示','觉得','解释',
                      '言传','言语','询问','说明','说罢','说着','说完','谴责','追问','骂','鼓舞','点头','打断','同意','咧开','常说',
                      '想了','想出','想到','感到','感叹','掩饰','明白','沉默','犹豫','盘算','看着','调转']
english_speech_tags = [
    "say",
    "reply",
    "tell",
    "ask",
    "yell",
    "answer",
    "whisper",
    "respond",
    "speak",
    "inform",
    "suggest",
    "request",
    "comment",
    "reclaim",
    "blame",
    "murmur",
    "mutter",
    "remark",
    "continue",
    "add",
    "shout",
    "scream",
    "bellow",
    "holler",
    "roar",
    "hiss",
    "laugh",
    "giggle",
    "sigh",
    "sob",
    "cry",
    "moan",
    "snarl",
    "growl",
    "sneer",
    "quip",
    "retort",
    "snap",
    "jeer",
    "taunt",
    "scold",
    "admonish",
    "lecture",
    "encourage",
    "stammer",
    "stutter",
    "drawl",
    "ramble",
    "rush",
    "blurt",
    "mumble",
    "drone",
    "chant",
    "gasp",
    "pant",
    "wheeze",
    "cough",
    "choke",
    "sputter",
    "yawn",
    "hiccup",
    "croak",
    "assert",
    "declare",
    "proclaim",
    "announce",
    "insist",
    "guess",
    "wonder",
    "speculate",
    "doubt",
    "explain",
    "elaborate",
    "inform",
    "reveal",
    "disclose",
    "divulge",
    "narrate",
    "recount",
    "describe",
    "inquire",
    "interrogate",
    "question",
    "probe",
    "query",
    "press",
    "grill",
    "pry",
    "urge",
    "coax",
    "cajole",
    "implore",
    "beg",
    "plead",
    "advise",
    "recommend",
    "propose",
    "agree",
    "concur",
    "acknowledge",
    "consent",
    "acquiesce",
    "object",
    "protest",
    "oppose",
    "dispute",
    "exclaim",
    "gasp",
    "gape",
    "blurt",
    "stammer",
    "marvel",
    "realize",
    "discover",
    "deduce",
    "rejoin"
]

# Add neural coref to SpaCy's pipe
#import neuralcoref
#neuralcoref.add_to_pipe(nlp_en)




def has_speaker(text, aliases, is_chinese):
    exist = False
    for alias in aliases:
        results = find_words_with_non_alpha_boundaries(text.lower(),alias.lower(),is_chinese)
        if len(results)>0:
            exist = True
            break
    return exist


## Directly match the phrases like "pronoun + speech_tag or speech_tag + pronoun"
def anaphoric_template_match(text_parse,is_chinese):
    tokens = list(text_parse)
    pronoun_list = english_pronoun_list if not is_chinese else chinses_pronoun_list
    speech_tags = english_speech_tags if not is_chinese else chinses_speech_tags
    for i,token in enumerate(tokens):
        if token.lemma_ in speech_tags:
            precede_token = tokens[i-1] if i-1>= 0 else None
            succeede_token = tokens[i+1] if i+1<len(tokens) else None
            if precede_token is not None and precede_token.text.lower().strip() in pronoun_list:
                match_str =  ' '.join([precede_token.text,token.text])
                print(f'match str:{match_str}')
                return match_str
            if succeede_token is not None and succeede_token.text.lower().strip() in pronoun_list:
                match_str = ' '.join([token.text,succeede_token.text])
                print(f'match str:{match_str}')
                return match_str
    return None



def find_nodes_with_compound_edge(node, is_chinese):
    result = []
    for child in node.children:
        if child.dep_ == 'compound':
            result += find_nodes_with_compound_edge(child, is_chinese)
    result += [node.text]
    if is_chinese:
        result = ''.join(result)
    else:
        result = ' '.join(result)
    return [result]


def find_speech_tags(node, is_chinese):
    speech_tag = None
    if is_chinese:
        if any(map(lambda x:x in node.text,chinses_speech_tags)):
            speech_tag = node.text
        else:
            for child in node.children:
                if child.dep_=='conj' and any(map(lambda x:x in child.text,chinses_speech_tags)):
                    speech_tag = child.text
                    break
    else:
        if node.lemma_ in english_speech_tags:
            speech_tag = node.text
        else:
            for child in node.children:
                if child.dep_=='conj' and child.lemma_ in english_speech_tags:
                    speech_tag = child.text
                    break
    return speech_tag


def get_evidence_for_qtype(sent,sent_parse,names,is_chinese):
    if is_chinese:
        pronoun_list = chinses_pronoun_list
    else:
        pronoun_list = english_pronoun_list
    qtype=None
    if has_speaker(sent, names, is_chinese):
        qtype = 'Explicit'
    else:
        #sent_parse = nlp_zh(sent) if is_chinese else nlp_en(sent)
        ana_match = anaphoric_template_match(sent_parse, is_chinese)
        if ana_match is not None:
            qtype = 'Anaphoric(pronoun)'
        else:
            sent_roots = list(filter(lambda x: x.dep_ == 'ROOT', sent_parse))
            if len(sent_roots) > 0:
                sent_nsubj = list(filter(lambda x: x.dep_ == 'nsubj', sent_roots[0].children))
                if len(sent_nsubj) > 0:
                    sent_nsubj_text = find_nodes_with_compound_edge(sent_nsubj[0], is_chinese)[0]
                    sent_speech_tag = find_speech_tags(sent_roots[0], is_chinese)
                    if (sent_speech_tag is not None and sent_nsubj_text.lower() in pronoun_list):
                        qtype = 'Anaphoric(pronoun)'
                    elif (sent_speech_tag is not None and sent_nsubj_text.lower() not in pronoun_list):
                        qtype = 'Anaphoric(other)'
    return qtype



def get_surrounding_narr_sents(text, quote, is_chinese):
    sents = cut_sentence_with_quotation_marks(text,is_chinese)
    #try:
    quote_idx = list(filter(lambda x:x[1]['sentence'].find(quote) != -1,enumerate(sents)))[0][0]
    #except Exception as e:
    #    print(f'unrecognized quote {quote} within the text:{text}')
    #    return None,None
    if quote_idx>0 and sents[quote_idx-1]['mode'] == 'Narrative':
        preceding_sent = sents[quote_idx-1]['sentence']
    else:
        preceding_sent = None

    if quote_idx<len(sents)-1 and sents[quote_idx+1]['mode'] == 'Narrative':
        subseq_sent = sents[quote_idx+1]['sentence']
    else:
        subseq_sent = None
    return preceding_sent,subseq_sent



def GetQuoteType(aliases,
                 preceding_sent,
                 preceding_sent_parse,
                 subseq_sent,
                 subseq_sent_parse,is_chinese):
    # text: the fragment of text includes the quote
    # aliases: the aliases for the speaker
    # we assume that any narrative sentence within the paragraph indicates that the quote is Explicit or Anaphoric,
    # then the quote type is specified to the type indicated by that sentence.
    preceding_sent_qtype,subseq_sent_qtype=None,None
    #preceding_sent,subseq_sent = get_surrounding_narr_sents(text,quote,is_chinese)
    if preceding_sent is not None:
        preceding_sent_qtype = get_evidence_for_qtype(preceding_sent, preceding_sent_parse, aliases, is_chinese)

    if subseq_sent is not None:
        subseq_sent_qtype = get_evidence_for_qtype(subseq_sent,subseq_sent_parse,aliases,is_chinese)

    if preceding_sent_qtype is None and subseq_sent_qtype is not None:
        qtype = subseq_sent_qtype

    elif preceding_sent_qtype is not None and subseq_sent_qtype is None:
        qtype = preceding_sent_qtype

    elif preceding_sent_qtype is None and subseq_sent_qtype is None:
        qtype = 'Implicit'

    else:
        if preceding_sent_qtype is 'Explicit' or subseq_sent_qtype is 'Explicit':
            qtype = 'Explicit'
        elif preceding_sent_qtype is 'Anaphoric(pronoun)' or subseq_sent_qtype is 'Anaphoric(pronoun)':
            qtype = 'Anaphoric(pronoun)'
        else:
            qtype = 'Anaphoric(other)'
    return qtype


def GetQuoteTypeForCorpus(instances, is_chinese):
    # get the quote type for each quote within each instance
    id2inst = {inst['id']:inst for inst in instances}
    records = []
    for inst in instances:
        for para_idx,para in enumerate(inst['dialogue']):
            for quote_idx,quote_dict in enumerate(para['utterance']):
                record = {}
                record['id'] = inst['id']
                record['quote_id']=quote_dict['quote_id']
                record['quote']=quote_dict['quote']
                record['paragraph'] = para['paragraph']
                try:
                    preceding_sent,subseq_sent = get_surrounding_narr_sents(para['paragraph'],quote_dict['quote'],is_chinese)
                except Exception as e:
                    print(f'instance:{inst["id"]},para:{para_idx},quote:{quote_idx}')
                    preceding_sent, subseq_sent=None,None
                record['preceding_sent'] = preceding_sent
                record['subsequent_sent'] = subseq_sent
                record['speaker'] = quote_dict['speaker']
                record['character'] = inst['character']
                record['pos']=(para_idx,quote_idx)
                records.append(record)
    batch_size = 32
    batch=[]
    for i in range(len(records)):
        if records[i]['preceding_sent'] is not None:
            batch.append(records[i])
        else:
            records[i]['preceding_sent_parse'] = None

        if len(batch)==batch_size or i == len(records)-1:
            batch_text = list(map(lambda x:x['preceding_sent'],batch))
            if is_chinese:
                parse_results = nlp_zh.pipe(batch_text)
            else:
                parse_results = nlp_en.pipe(batch_text)
            for record,parse_result in zip(batch,parse_results):
                record['preceding_sent_parse'] = parse_result
            batch = []

    batch = []
    for i in range(len(records)):
        if records[i]['subsequent_sent'] is not None:
            batch.append(records[i])
        else:
            records[i]['subsequent_sent_parse'] = None

        if len(batch)==batch_size or i == len(records)-1:
            batch_text = list(map(lambda x:x['subsequent_sent'],batch))
            if is_chinese:
                parse_results = nlp_zh.pipe(batch_text)
            else:
                parse_results = nlp_en.pipe(batch_text)
            for record, parse_result in zip(batch,parse_results):
                record['subsequent_sent_parse'] = parse_result
            batch = []

    for record in records:
        char_dict = record['character']
        if record['speaker'] not in char_dict['name2id']:
            qtype = 'Implicit'
        else:
            aliases = char_dict['id2names'][str(char_dict['name2id'][record['speaker']])]
            qtype = GetQuoteType(aliases,record['preceding_sent'],record['preceding_sent_parse'],
                             record['subsequent_sent'],record['subsequent_sent_parse'],is_chinese)
        inst = id2inst[record['id']]
        para_idx,quote_idx = record['pos']
        inst['dialogue'][para_idx]['utterance'][quote_idx]['quote_type'] = qtype
    return instances



def GetSurroundingQuoteOld(root):
    if len(root['preceding_paragraphs'])>0 and root['preceding_paragraphs'][-1]['mode'] == 'Dialogue':
        preceding_para = root['preceding_paragraphs'][-1]
    else:
        if len(root['preceding_paragraphs'])>1 and root['preceding_paragraphs'][-2]['mode'] == 'Dialogue':
            preceding_para = root['preceding_paragraphs'][-2]
        else:
            preceding_para = None

    if len(root['succeeding_paragraphs'])>0 and root['succeeding_paragraphs'][0]['mode'] == 'Dialogue':
        next_para = root['succeeding_paragraphs'][0]
    else:
        if len(root['succeeding_paragraphs'])>1 and root['succeeding_paragraphs'][1]['mode'] == 'Dialogue':
            next_para = root['succeeding_paragraphs'][1]
        else:
            next_para = None

    return preceding_para,next_para

def GetSurroundingQuote(root):
    interval_para_num = 2
    preceding_para,next_para= None,None
    for i in range(min(interval_para_num+1,len(root["preceding_paragraphs"]))):
        p_i = len(root['preceding_paragraphs'])-i-1
        p = root['preceding_paragraphs'][p_i]
        if p['mode'] == 'Dialogue':
            preceding_para = p
            break

    for i in range(min(interval_para_num+1,len(root["succeeding_paragraphs"]))):
        p = root['succeeding_paragraphs'][i]
        if p['mode'] == 'Dialogue':
            next_para = p
            break
    return preceding_para,next_para






def GetSingleCluster(root,candidates):
    root_para = root['dialogue'][0]
    pre_para,next_para = GetSurroundingQuote(root)

    if next_para is None and pre_para is None:
        return [root]

    left_inst,right_inst = None,None
    for candidate in candidates:
        candidate_para = candidate['dialogue'][0]
        candidate_pre_para, candidate_next_para = GetSurroundingQuote(candidate)
        if (next_para is not None and
            candidate_pre_para is not None and
            next_para['paragraph'] == candidate_para['paragraph'] and
            candidate_pre_para['paragraph']==root_para['paragraph']):
            right_inst= candidate
        elif (pre_para is not None and
              candidate_next_para is not None and
              pre_para['paragraph'] == candidate_para['paragraph'] and
              candidate_next_para['paragraph']==root_para['paragraph']):
            left_inst= candidate

    left_insts,right_insts=[],[]
    removed_ids = list(map(lambda x: x['id'], filter(lambda y:y is not None,[right_inst,left_inst])))
    if right_inst is not None :
        reduced_candidates = list(filter(lambda x: x['id'] not in removed_ids, candidates))
        right_insts = GetSingleCluster(right_inst,reduced_candidates)
    if left_inst is not None:
        #removed_ids = list(map(lambda x: x['id'], [left_inst]))
        reduced_candidates = list(filter(lambda x: x['id'] not in removed_ids, candidates))
        left_insts = GetSingleCluster(left_inst, reduced_candidates)
    return left_insts+[root]+right_insts



def MergeCharacterList(d1,d2):
    d3 = {'id2names':{},'name2id':{},'id2gender':{}}
    char_count = 0
    for k,v in d1['id2names'].items():
        if k == str(CharMaskId):
            d3['id2names'][k]=v
        else:
            d3['id2names'][str(char_count)] =  v
        if 'id2gender' in d1:
            if k==str(CharMaskId):
                d3['id2gender'][k] = d1['id2gender'][k]
            else:
                d3['id2gender'][str(char_count)] = d1['id2gender'][k]
        char_count += 1

    for k,v in d2['id2names'].items():
        exist = False
        for sk,sv in d3['id2names'].items():
            if set(sv)==set(v):
                exist = True
                break
        if not exist:
            if k == str(CharMaskId):
                d3['id2names'][k] = v
            else:
                d3['id2names'][str(char_count)] = v
            if 'id2gender' in d2:
                if k == str(CharMaskId):
                    d3['id2gender'][k] = d2['id2gender'][k]
                else:
                    d3['id2gender'][str(char_count)] = d2['id2gender'][k]
            char_count +=1

    for k,v in d3['id2names'].items():
        for name in v:
            d3['name2id'][name] = int(k)
    return d3


def MergeInsideCluster(cluster):
    dialogue = []
    new_char_dict = {'id2names':{},'name2id':{},'id2gender':{}}
    for i in range(len(cluster)):
        inst = cluster[i]
        new_char_dict = MergeCharacterList(new_char_dict,inst['character'])
        dialogue.extend(inst['dialogue'])
        if i<len(cluster)-1 and inst['succeeding_paragraphs'][0]['mode'] == 'Narrative':
            dialogue.append(inst['succeeding_paragraphs'][0])
    inst = {
        'preceding_paragraphs':cluster[0]['preceding_paragraphs'],
        'dialogue':dialogue,
        'succeeding_paragraphs':cluster[-1]['succeeding_paragraphs'],
        'id':None,
        'character':new_char_dict
    }
    return inst


def ClusterQuote(insts):
    clusters = []
    while len(insts)>0:
        inst = insts[0]
        reduced_insts = insts[1:]
        cluster = GetSingleCluster(inst,reduced_insts)
        clusters.append(cluster)
        removed_ids = list(map(lambda x:x['id'],cluster))
        reduced_insts = list(filter(lambda x:x['id'] not in removed_ids,insts))
        insts = reduced_insts

    new_insts = []
    for cid,cluster in enumerate(clusters):
        new_inst = MergeInsideCluster(cluster)
        new_inst['id'] = cid
        new_insts.append(new_inst)
    return new_insts




def FilterCluster(clusters, entities, pronoun_list):
    new_clusters = {}
    rule = r'(\W{}\W)|(\W{}\Z)|(\b{}\W)|(\b{}\Z)'
    pronoun_list = set(map(lambda x:x.lower(),pronoun_list))
    for cluster in clusters:
        main = cluster.main.text
        invalid_cluster = False
        for pron in pronoun_list:
            if re.search(rule.format(pron,pron,pron,pron),main.lower()) != None:
                invalid_cluster = True
                break

        if invalid_cluster:
            continue
        mentions = list(map(lambda x:x.text,cluster.mentions))
        new_mentions = []
        invalid_mention = False
        for mention in mentions:
            for pron in pronoun_list:
                if re.search(rule.format(pron,pron,pron,pron),mention.lower().strip()) != None:
                    invalid_mention = True
                    break
            if not invalid_mention and len(mention.split())<=5:
                new_mentions.append(mention)
        new_mentions = set(new_mentions)
        if len(new_mentions)>0 and len(set(entities).intersection(new_mentions))>0:
            new_clusters[main] = list(new_mentions)
    return new_clusters


# each instance only has one turn dialogue
def FindSurroundingQuotes(cluster, added_insts, para2inst, id2inst, skip_steps):
    # print(para2inst)
    preceding_paragraphs = cluster['preceding_paragraphs']
    left_inst, right_inst = None, None
    left_most_pos = max(len(preceding_paragraphs) - 1 - skip_steps, 0)
    for i in range(len(preceding_paragraphs) - 1, left_most_pos - 1, -1):
        para = preceding_paragraphs[i]
        if para['paragraph'] in para2inst:
            for inst_info in para2inst[para['paragraph']]:
                if inst_info['instance_id'] in added_insts:
                    continue

                cand_inst = id2inst[inst_info['instance_id']]
                cand_succ_paras = cand_inst['succeeding_paragraphs']
                temp_ori_pre_paras = preceding_paragraphs[i + 1:] + cluster['dialogue'] + cluster[
                    'succeeding_paragraphs']
                match = True
                for k in range(min(len(temp_ori_pre_paras), len(cand_succ_paras))):
                    if cand_succ_paras[k]['paragraph'] != temp_ori_pre_paras[k]['paragraph']:
                        match = False
                if match:
                    added_insts.append(inst_info['instance_id'])
                    left_inst = (i, cand_inst)
                    break
        if left_inst is not None:
            break

    succeeding_paragraphs = cluster['succeeding_paragraphs']
    right_most_pos = min(skip_steps + 1, len(succeeding_paragraphs))
    for i in range(right_most_pos):
        para = succeeding_paragraphs[i]
        if para['paragraph'] in para2inst:
            # print(f'find:{para["paragraph"]} ')
            for inst_info in para2inst[para['paragraph']]:
                if inst_info['instance_id'] in added_insts:
                    continue

                cand_inst = id2inst[inst_info['instance_id']]
                cand_pre_paras = cand_inst['preceding_paragraphs']
                temp_ori_succ_paras = cluster['preceding_paragraphs'] + cluster['dialogue'] + cluster[
                                                                                                  'succeeding_paragraphs'][
                                                                                              :i]
                match = True
                for k in range(min(len(cand_pre_paras), len(temp_ori_succ_paras))):
                    if cand_pre_paras[len(cand_pre_paras) - 1 - k]['paragraph'] != \
                            temp_ori_succ_paras[len(temp_ori_succ_paras) - 1 - k]['paragraph']:
                        match = False
                        # print(f"p1:{cand_pre_paras[len(cand_pre_paras)-1-k]['paragraph']},\np2:{temp_ori_succ_paras[len(temp_ori_succ_paras)-1-k]['paragraph']}")
                if match:
                    added_insts.append(inst_info['instance_id'])
                    right_inst = (i, cand_inst)
                    break
                print(f'match:{match}')
        if right_inst is not None:
            # print(right_inst[1]['dialogue'])
            break

    if left_inst is None and right_inst is None:
        new_cluster = None
    else:
        preceding_paragraphs = (left_inst[1]['preceding_paragraphs']
                                if left_inst is not None else cluster['preceding_paragraphs'])
        succeeding_paragraphs = (right_inst[1]['succeeding_paragraphs']
                                 if right_inst is not None else cluster['succeeding_paragraphs'])
        dialogue = ((left_inst[1]['dialogue'] + cluster['preceding_paragraphs'][
                                                left_inst[0] + 1:] if left_inst is not None else []) + cluster[
                        'dialogue'] + (cluster['succeeding_paragraphs'][:right_inst[0]] + right_inst[1][
            'dialogue'] if right_inst is not None else []))
        char_dict = copy.deepcopy(cluster['character'])
        if left_inst is not None:
            for k in char_dict.keys():
                char_dict[k].update(left_inst[1]['character'][k])
        if right_inst is not None:
            for k in char_dict.keys():
                char_dict[k].update(right_inst[1]['character'][k])

        new_cluster = {
            'preceding_paragraphs': preceding_paragraphs,
            'dialogue': dialogue,
            'succeeding_paragraphs': succeeding_paragraphs,
            'character': char_dict
        }

    return new_cluster


def GetClusters(insts, skip_steps=1):
    para2inst = defaultdict(list)
    id2inst = {inst['id']: inst for inst in insts}
    for inst in insts:
        for para_idx, para in enumerate(inst['dialogue']):
            para2inst[para['paragraph']].append({'instance_id': inst['id'], 'paragraph_idx': para_idx})

    added_insts = []
    clusters = []
    for inst in insts:
        if inst['id'] not in added_insts:
            cluster = copy.deepcopy(inst)
            added_insts.append(inst['id'])
            while True:
                new_cluster = FindSurroundingQuotes(cluster, added_insts, para2inst, id2inst, skip_steps)
                if new_cluster is None:
                    cluster['id'] = len(clusters)
                    clusters.append(cluster)
                    break
                else:
                    cluster = new_cluster
                    # print(f'new cluster:{len(cluster["dialogue"])}')
            # break
    return clusters







def FilterEntity(ents):
    new_ents = []
    for ent in ents:
        if ent.label_ == 'PERSON':
            new_ents.append(ent.text)
    new_ents = list(set(new_ents))
    print(f'entities:{new_ents}')
    return new_ents


def GetCharDictEn(parser, texts, pronoun_path):
    char_dicts = []
    with open(pronoun_path,'r') as f:
        pronoun_list = eval(f.read())
    docs = parser.pipe(texts)
    for doc in docs:
        ents = doc.ents
        ents = FilterEntity(ents)
        clusters = doc._.coref_clusters
        clusters = FilterCluster(clusters,ents,pronoun_list)
        char_dict = {'id2names':{},'name2id':{},'id2gender':{}}
        for i,k in enumerate(clusters.keys()):
            char_dict['id2names'][str(i)] = clusters[k]
            char_dict['id2gender'][str(i)] = 'None'
            for n in clusters[k]:
                char_dict['name2id'][n] = i
        char_dicts.append(char_dict)
    return char_dicts






# Convert the PAP dataset
def ExtractDialogue(paragraphs, context_len=None, interval_paragraph=1):
    dialogue_indexs = []
    ps, pe = 0, 0
    def has_utterance(paragraphs, start,end):
        exist = False
        for paragraph in paragraphs[start:end]:
            #if paragraph['mode'] == 'Dialogue':
            if len(paragraph['utterance'])>0:
                exist = True
        return exist
    while pe < len(paragraphs):
        #if paragraphs[pe]['mode'] == 'Dialogue':
        if len(paragraphs[pe]['utterance'])>0:
            #while pe < len(paragraphs) and paragraphs[pe]['mode'] == 'Dialogue':
            while pe<len(paragraphs) and has_utterance(paragraphs,pe,pe+interval_paragraph):
                pe += 1
            #if paragraphs[ps]['mode'] == 'Dialogue':
            if len(paragraphs[ps]['utterance'])>0:
                dialogue_indexs.append([ps, pe])
            else:
                dialogue_indexs.append([ps + 1, pe])
        ps = pe
        pe += 1
    insts = []
    for start, end in dialogue_indexs:
        dial_paras = list(filter(lambda x:len(x['utterance'])>0,paragraphs[start:end]))
        if len(dial_paras) >0:
            inst = {
                'preceding_paragraphs': paragraphs[:start] if context_len is None else paragraphs[
                                                                                  max(0, start - context_len):start],
                'succeeding_paragraphs': paragraphs[end :] if context_len is None else paragraphs[
                                                                                     end :end  + context_len],
                'dialogue': paragraphs[start:end]}
            insts.append(inst)
    return insts


#def ReadPAPCharFile(char_file_path):
#    char_dict = {'name2id': {}, 'id2names': {}}
#    with open(char_file_path, 'r') as f:
#        lines = f.read().split('\n')
#        lines = [[e.strip() for e in line.split(';')] for line in lines]
#    for c_id, line in enumerate(lines):
#        char_dict['id2names'][str(c_id)] = line[:1] + line[2:]
#        char_dict['name2id'].update(dict([(alias, c_id) for alias in line[:1] + line[2:]]))

#    return char_dict

# TODO: check how the quote_id is determined (v)
# TODO: check the the utterance span (v)
# TODO:
def FormatPAP(sour_path, target_path, char_file_path):
    with open(sour_path, 'r') as f:
        sour_data = json.load(f)

    with open(char_file_path,'r') as f:
        char_dict = json.load(f)
    char_dict = AddUnkToken(char_dict)

    dest_data = []
    dest_count = 0
    for chapter_id, chapter in tqdm(sour_data,desc="FormatPAP progress"):
        dialogues = ExtractDialogue(chapter,context_len=10)
        dialogues = copy.deepcopy(dialogues)
        for dialogue in dialogues:
            dial_paras = dialogue['preceding_paragraphs']+dialogue['dialogue']+dialogue['succeeding_paragraphs']
            text = ' '.join(list(map(lambda x:x['paragraph'],dial_paras)))
            for para_index in range(len(dial_paras)):
                offset = len(' '.join(list(map(lambda x: x['paragraph'], dial_paras[:para_index]))))
                dial_paras[para_index]['offset'] = [offset, offset + len(dial_paras[para_index]['paragraph'])]
            for para in dialogue['dialogue']:
                #utter_offsets = []
                for utter in para['utterance']:
                    offset = para['paragraph'].find(utter['quote'])
                    utter['quote_span'] = [offset,offset+len(utter) if offset!=-1 else -1]
                #para['utterance_span'] = utter_offsets
            sub_char_dict = GetSubCharDict(char_dict,text)
            dialogue['character'] = sub_char_dict
            dialogue['id'] = f'PAP-{dest_count}'
            dest_count+=1
        dest_data += dialogues

    tqdm.write(f"# instance for FormatPAP:{len(dest_data)}")
    with open(target_path, 'w') as f:
        json.dump(dest_data, f, indent=2)




# TODO: recompute offset for paragraphs, utterance span
# TODO: recreate the character dictionary, the name2id 's key is char instead of name

def FormatRIQUA(sour_path, target_path):
    with open(sour_path, 'r') as f:
        sour_data = json.load(f)

    pronoun_path = os.path.join(SCRIPT_DIR,'../pronoun_list.txt')

    dest_data = []
    dest_count = 0
    for k in tqdm(sour_data.keys(),desc="FormatRIQUA progress"):
        paragraphs = sour_data[k]['paragraphs']
        char_dict = BuildRIQUACharDict(paragraphs,pronoun_path)
        dialogues = ExtractDialogue(paragraphs, context_len=10)
        dialogues = copy.deepcopy(dialogues)
        for dialogue in dialogues:
            dial_paras = dialogue['preceding_paragraphs']+dialogue['dialogue']+dialogue['succeeding_paragraphs']
            text = ' '.join(list(map(lambda x: x['paragraph'],dial_paras)))
            for para_index in range(len(dial_paras)):
                offset = len(' '.join(list(map(lambda x:x['paragraph'],dial_paras[:para_index]))))
                dial_paras[para_index]['offset'] = [offset,offset+len(dial_paras[para_index]['paragraph'])]
            for para_index,para in enumerate(dialogue['dialogue']):
                #para['utterance_id'] = [f'{dest_count}-{para_index}-{utter_index}'
                #                       for utter_index in range(len(para['utterance_id']))]
                #utter_offsets = []
                for utter in para['utterance']:
                    offset = para['paragraph'].find(utter['quote'])
                    utter['quote_span'] = [offset,offset+len(utter) if offset!=-1 else -1]
                    #utter['quote_id'] = f'{dest_count}-{para_index}-{utter_index}'
                #para['utterance_span'] = utter_offsets
            sub_char_dict = GetSubCharDict(char_dict,text)
            dialogue['character'] = sub_char_dict
            dialogue['id'] = f'RIQUA-{dest_count}'
            dest_count+=1

        dest_data += dialogues

    tqdm.write(f"# instance for FormatRIQUA:{len(dest_data)}")
    with open(target_path, 'w') as f:
        json.dump(dest_data, f, indent=2)


# TODO: the id list for paragraph is repeated
# TODO: recomputet the offset for paragraphs

def FormatPDNC(sour_path, char_file_path):
    with open(sour_path, 'r') as f:
        sour_data = json.load(f)

    with open(char_file_path, 'rb') as f:
        char_dict = pickle.load(f)
        new_char_dict = {'id2names':{},'name2id':{}}
        for k,v in char_dict['id2names'].items():
            new_char_dict['id2names'][str(k)] = list(v)
            for n in v:
                new_char_dict['name2id'][n] = k
        char_dict = new_char_dict

    dialogues = ExtractDialogue(sour_data, context_len=10)
    dialogues = copy.deepcopy(dialogues)
    dest_count = 0
    for dialogue in dialogues:
        dial_paras = dialogue['preceding_paragraphs'] + dialogue['dialogue'] + dialogue['succeeding_paragraphs']
        text = ' '.join(list(map(lambda x: x['paragraph'], dial_paras)))
        for para_index in range(len(dial_paras)):
            offset = len(' '.join(list(map(lambda x: x['paragraph'], dial_paras[:para_index]))))
            dial_paras[para_index]['offset'] = [offset, offset + len(dial_paras[para_index]['paragraph'])]
        for para_index,para in enumerate(dialogue['dialogue']):
            #para['utterance_id'] = [f'{dest_count}-{para_index}-{utter_index}' for utter_index in
            #                        range(len(para['utterance_id']))]
            #utter_offsets = []
            for utter in para['utterance']:
                offset = para['paragraph'].find(utter['quote'])
                utter['quote_span'] = [offset, offset + len(utter) if offset != -1 else -1]
                #utter_offsets.append([offset, offset + len(utter) if offset != -1 else -1])
            #para['utterance_span'] = utter_offsets
        sub_char_dict = GetSubCharDict(char_dict, text)
        dialogue['character'] = sub_char_dict
        dialogue['id'] = str(dest_count)
        dest_count += 1
    dest_data = dialogues

    return dest_data


def FormatSQLITE(sour_path, char_file_path):
    with open(sour_path, 'r') as f:
        sour_data = json.load(f)

    with open(char_file_path, 'r') as f:
        char_dict = json.load(f)

    dialogues = ExtractDialogue(sour_data, context_len=10)
    dialogues = copy.deepcopy(dialogues)
    dest_count = 0
    for dialogue in dialogues:
        dial_paras = dialogue['preceding_paragraphs'] + dialogue['dialogue'] + dialogue['succeeding_paragraphs']
        text = ' '.join(list(map(lambda x: x['paragraph'], dial_paras)))
        for para_index in range(len(dial_paras)):
            offset = len(' '.join(list(map(lambda x: x['paragraph'], dial_paras[:para_index]))))
            dial_paras[para_index]['offset'] = [offset, offset + len(dial_paras[para_index]['paragraph'])]
        for para_index,para in enumerate(dialogue['dialogue']):
            #para['utterance_id'] = [f'{dest_count}-{para_index}-{utter_index}' for utter_index in
            #                        range(len(para['utterance_id']))]
            #utter_offsets = []
            for utter in para['utterance']:
                offset = para['paragraph'].find(utter['quote'])
                utter['quote_span'] = [offset, offset + len(utter) if offset != -1 else -1]
                #utter_offsets.append([offset, offset + len(utter) if offset != -1 else -1])
            #para['utterance_span'] = utter_offsets
        sub_char_dict = GetSubCharDict(char_dict, text)
        dialogue['character'] = sub_char_dict
        dialogue['id'] = str(dest_count)
        dest_count += 1
    dest_data = dialogues

    return dest_data


def ProcessCSI(sour_dir, target_dir):
    sour_path = os.path.join(sour_dir, f'train_paragraphs.json')
    with open(sour_path,'r') as f:
        data = json.load(f)
    random.seed(123)
    data = random.sample(data,k=len(data))
    dev_size = int(len(data)*0.1)
    dev_set = data[:dev_size]
    dev_set = ClusterQuote(dev_set)
    train_set = data[dev_size:]
    train_set = ClusterQuote(train_set)
    for set_name in ['train','dev']:
        target_path = os.path.join(target_dir,f'{set_name}.json')
    with open(target_path,'w') as f:
        if set_name == 'train':
            json.dump(train_set,f,indent=2)
        else:
            json.dump(dev_set,f,indent=2)
    sour_path = os.path.join(sour_dir,'test_paragraphs.json')
    with open(sour_path,'r') as f:
        test_set = json.load(f)
    test_set = ClusterQuote(test_set)
    target_path = os.path.join(target_dir,'test.json')
    with open(target_path,'w') as f:
        json.dump(test_set,f,indent=2)



def ProcessJY(sour_dir, target_dir):
    for set_name in ['train','dev','test']:
        sour_path = os.path.join(sour_dir,f'{set_name}_paragraphs.json')
        with open(sour_path,'r') as f:
            data = json.load(f)
        data = ClusterQuote(data)
        target_path = os.path.join(target_dir,f'{set_name}.json')
        with open(target_path,'w') as f:
            json.dump(data,f,indent=2)



def ProcessWP2021(sour_dir,target_dir):
    #sour_dir = 'data/raw_data/WP2021'
    #target_dir = 'data/proc_data/WP2021'
    os.makedirs(target_dir,exist_ok=True)
    t = tqdm(['train', 'dev','test'])
    for set_name in t:
        t.set_description(desc=f"Process WP2021: {set_name}")
        sour_path = os.path.join(sour_dir, f'{set_name}_paragraphs.json')
        target_path = os.path.join(target_dir, f'{set_name}.json')
        with open(sour_path,'r') as f:
            insts = json.load(f)
        insts = ClusterQuote(insts)
        with open(target_path,'w') as f:
            json.dump(insts,f,indent=2)






def ProcessRIQUA(sour_dir,target_dir):
    #sour_dir = 'data/raw_data/RIQUA'
    #target_dir = 'data/proc_data/RIQUA'
    t = tqdm(['train',  'test'])
    os.makedirs(target_dir,exist_ok=True)
    for set_name in t:
        t.set_description(desc=f"Process RIQUA: {set_name}")
        sour_path = os.path.join(sour_dir, f'{set_name}_paragraphs.json')
        target_path = os.path.join(target_dir, f'{set_name}.json')
        FormatRIQUA(sour_path, target_path)

    with open(os.path.join(target_dir,'test.json'),'r') as f:
        data = json.load(f)
    dev_size = int(len(data)*0.5)
    random.seed(123)
    data = random.sample(data,k=len(data))
    dev_set = data[:dev_size]
    test_set = data[dev_size:]
    for set_name in ['test','dev']:
        target_path = os.path.join(target_dir,f'{set_name}.json')
        with open(target_path,'w') as f:
            if set_name == 'test':
                json.dump(test_set,f,indent=2)
            else:
                json.dump(dev_set,f,indent=2)




def ProcessPDNC(sour_dir,target_dir):
    dest_data = []

    os.makedirs(target_dir,exist_ok=True)
    t = tqdm(os.listdir(sour_dir))
    for book_name in t:
        book_path = os.path.join(sour_dir,book_name,'book_anno.json')
        char_file_path = os.path.join(sour_dir,book_name,'charDict.pkl')
        if os.path.exists(book_path):
            t.set_description(desc=f"Process PDNC: {book_name}")
            book_data = FormatPDNC(book_path,char_file_path)
            for inst in book_data:
                #for para in inst['preceding_paragraphs']+inst['dialogue']+inst['succeeding_paragraphs']:
                #    para['paragraph_index'] = f'{book_name}-{para["paragraph_index"]}'
                #for para in inst['dialogue']:
                #    para['utterance_id'] = [f'{book_name}-{utter_id}' for utter_id in para['utterance_id']]
                inst['id'] = f'{book_name}-{inst["id"]}'
            dest_data += book_data

    random.seed(123)
    dest_data = random.sample(dest_data,k=len(dest_data))
    test_size = int(len(dest_data)*0.1)
    set_dict = {}
    set_dict['test'] = dest_data[:test_size]
    set_dict['dev'] = dest_data[test_size:2*test_size]
    set_dict['train'] = dest_data[2*test_size:]
    os.makedirs(os.path.join(target_dir, 'fold0'), exist_ok=True)
    for set_name in ['train','dev','test']:
        set_path = os.path.join(target_dir,'fold0',set_name+'.json')
        with open(set_path,'w') as f:
            json.dump(set_dict[set_name],f,indent=2)


def ProcessSQLITE(sour_dir,target_dir):

    test_data = []

    os.makedirs(target_dir,exist_ok=True)
    t = tqdm(["PAP_train",'PAP_dev','PAP_test',"Emma","Steppe"])
    for book_name in t:
        book_path = os.path.join(sour_dir,f'{book_name}_paragraphs.json')
        char_file_path = os.path.join(sour_dir,f"{book_name}_character.json")
        if os.path.exists(book_path):
            t.set_description(desc=f"Process SQLITE: {book_name}")
            book_data = FormatSQLITE(book_path,char_file_path)
            for inst in book_data:
                #for para in inst['preceding_paragraphs']+inst['dialogue']+inst['succeeding_paragraphs']:
                #    para['paragraph_index'] = f'{book_name}-{para["paragraph_index"]}'
                #for para in inst['dialogue']:
                #    para['utterance_id'] = [f'{book_name}-{utter_id}' for utter_id in para['utterance_id']]
                inst['id'] = f'{book_name}-{inst["id"]}'
            if book_name == 'PAP_train':
                train_data = book_data
            elif book_name == 'PAP_dev':
                dev_data = book_data
            else:
                test_data+=book_data

    os.makedirs(target_dir, exist_ok=True)
    for set_name in ['train','dev','test']:
        set_path = os.path.join(target_dir,set_name+'.json')
        with open(set_path,'w') as f:
            if set_name == 'train':
                save_data = train_data
            elif set_name == 'dev':
                save_data = dev_data
            else:
                save_data = test_data
            json.dump(save_data,f,indent=2)

def ProcessPDNCCategory(book_info_path, sour_dir,target_dir):
    random.seed(123)
    book_info_df = pd.read_csv(book_info_path)
    genre_clusters = book_info_df.groupby(['Coarse Label'])
    print([(k, 21 - len(g)) for k, g in genre_clusters])
    datasets = {}

    for k, g in genre_clusters:
        g_ids = list(g.index)
        r_ids = list(book_info_df[book_info_df['Coarse Label'] != k].index)
        train_dev_set = []
        datasets[k] = {'train_dev':[],'test':[]}
        for r_id in r_ids:
            book_name = book_info_df.at[r_id, 'Folder Name']
            book_path = os.path.join(sour_dir,book_name,'book_anno.json')
            char_path = os.path.join(sour_dir,book_name,'charDict.pkl')
            book_data = FormatPDNC(book_path,char_path)
            for inst in book_data:
                #for para in inst['preceding_paragraphs']+inst['dialogue']+inst['succeeding_paragraphs']:
                #    para['paragraph_index'] = f'{book_name}-{para["paragraph_index"]}'
                #for para in inst['dialogue']:
                #    para['utterance_id'] = [f'{book_name}-{utter_id}' for utter_id in para['utterance_id']]
                inst['id'] = f'{book_name}-{inst["id"]}'
            train_dev_set += book_data

        datasets[k]['train_dev'] = train_dev_set

        test_set = []
        for g_id in g_ids:
            book_name = book_info_df.at[g_id, 'Folder Name']
            book_path = os.path.join(sour_dir, book_name, 'book_anno.json')
            char_path = os.path.join(sour_dir, book_name, 'charDict.pkl')
            book_data = FormatPDNC(book_path, char_path)
            for inst in book_data:
                #for para in inst['preceding_paragraphs'] + inst['dialogue'] + inst['succeeding_paragraphs']:
                #    para['paragraph_index'] = f'{book_name}-{para["paragraph_index"]}'
                #for para in inst['dialogue']:
                #    para['utterance_id'] = [f'{book_name}-{utter_id}' for utter_id in para['utterance_id']]
                inst['id'] = f'{book_name}-{inst["id"]}'
            test_set += book_data
        datasets[k]['test'] = test_set

    min_train_dev_size = min([len(datasets[k]['train_dev']) for k in datasets.keys()])
    min_test_size = min([len(datasets[k]['test']) for k in datasets.keys()])

    for k in datasets.keys():
        dataset = {}
        train_dev_set = datasets[k]['train_dev']
        test_set = datasets[k]['test']

        train_dev_set = random.sample(train_dev_set,k=len(train_dev_set))
        dev_size = int(0.1*min_train_dev_size)
        dataset['dev'] = train_dev_set[:dev_size]
        dataset['train'] = train_dev_set[dev_size:min_train_dev_size]
        test_set = random.sample(test_set,k=len(test_set))
        dataset['test'] = test_set[:min_test_size]

        genre_dir = os.path.join(target_dir,k)
        os.makedirs(genre_dir,exist_ok=True)
        for set_name in ['train','dev','test']:
            set_path = os.path.join(genre_dir,f'{set_name}.json')
            with open(set_path,'w') as f:
                json.dump(dataset[set_name],f,indent=2)




import argparse,os
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sour_dir', type=str, help="the path to load the raw data")
    parser.add_argument('target_dir', type=str, help='the path to save the processed data')
    args = parser.parse_args()
    #ProcessCSI(os.path.join(args.sour_dir, 'CSI'), os.path.join(args.target_dir, 'CSI'))
    #ProcessJY(os.path.join(args.sour_dir, 'JY'), os.path.join(args.target_dir, 'JY'))
    #ProcessWP2021(os.path.join(args.sour_dir, 'WP2021'), os.path.join(args.target_dir, 'WP2021'))
    #ProcessPAP(os.path.join(args.sour_dir, 'PAP'), os.path.join(args.target_dir, 'PAP'))
    ProcessSQLITE(os.path.join(args.sour_dir,'SQLITE'),os.path.join(args.target_dir,'SQLITE'))
    #ProcessRIQUA(os.path.join(args.sour_dir, 'RIQUA'), os.path.join(args.target_dir, 'RIQUA'))
    #ProcessPDNC(os.path.join(args.sour_dir, 'PDNC'), os.path.join(args.target_dir, 'PDNC_merge'))
    #ProcessPDNCCategory(os.path.join(args.sour_dir,'PDNC','book_info.csv'),
    #                    os.path.join(args.sour_dir, 'PDNC')
    #                    ,os.path.join(args.target_dir, 'PDNC_genre'))