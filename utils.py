import copy
import re
from spacy.lang.zh import Chinese
from spacy.lang.en import English
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
from tqdm import tqdm

sent_tokenizer_zh = Chinese()
sent_tokenizer_en = English()
try:
    sent_tokenizer_zh.add_pipe('sentencizer')
    sent_tokenizer_en.add_pipe("sentencizer")
except Exception as e:
    sent_tokenizer_zh.add_pipe(sent_tokenizer_zh.create_pipe('sentencizer'))
    sent_tokenizer_en.add_pipe(sent_tokenizer_en.create_pipe('sentencizer'))
max_seq_len = 500
CharMaskId = 10000
MaxCharNum=40

# cut text into sentences. the text inside quotes is regarded as a single sentence.
# each instance is in the format as follows:
# {
#   "sentence": sentence_text,
#   "mode": "Utterance" or "Narrative"
# }

def find_words_with_non_alpha_boundaries(text, word, is_chinese):
    if not is_chinese:
        pattern = r'\b' + re.escape(word) + r'\b'
    else:
        pattern = re.escape(word)
    matches = list(re.finditer(pattern, text))
    return matches

def AddUnkToken(char_dict):
    if str(CharMaskId) in char_dict['id2names']:
        if 'None' in char_dict['id2names'][str(CharMaskId)]:
            return char_dict
        else:
            temp = char_dict['id2names'][str(CharMaskId)]
            temp_gender = char_dict['id2gender'][str(CharMaskId)]
            char_dict['id2names'][str(CharMaskId)] = ['None']
            char_dict['id2gender'][str(CharMaskId)] = 'None'
            char_dict['name2id']['None'] = CharMaskId
            new_temp_id = len(char_dict['id2names'])
            char_dict['id2names'][str(new_temp_id)] = temp
            char_dict['id2gender'][str(new_temp_id)] = temp_gender
            for name in temp:
                char_dict['name2id'][name] = new_temp_id
    else:
        char_dict['name2id']['None'] = CharMaskId
        char_dict['id2names'][str(CharMaskId)] = ['None']
        char_dict['id2gender'][str(CharMaskId)]='None'
    return char_dict

char_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def GetSubCharDict(instance,char_dict,is_chinese=False):
    dial_paras = instance['preceding_paragraphs'] + instance['dialogue'] + instance['succeeding_paragraphs']
    text = ' '.join(list(map(lambda x: x['paragraph'], dial_paras)))
    if not is_chinese:
        text = char_tokenizer.convert_tokens_to_string(char_tokenizer.tokenize(text))
    speakers = [quote_dict['speaker'] for para in instance['dialogue'] for quote_dict in para['utterance']
                if quote_dict['speaker'] in char_dict['name2id']]
    speaker_ids = [char_dict['name2id'][speaker] for speaker in speakers]
    name_ids = speaker_ids
    for name, name_id in char_dict['name2id'].items():
        if not is_chinese:
            name = char_tokenizer.convert_tokens_to_string(char_tokenizer.tokenize(name))
        if text.find(name) != -1:
            name_ids.append(name_id)
    name_ids = set(name_ids)
    sub_char_dict = {'name2id': {}, 'id2names': {}, 'id2gender':{}}
    for name_id in name_ids:
        sub_char_dict['id2names'][str(name_id)] = char_dict['id2names'][str(name_id)]
        sub_char_dict['id2gender'][str(name_id)]=char_dict['id2gender'][str(name_id)]
        for name in char_dict['id2names'][str(name_id)]:
            sub_char_dict['name2id'][name] = name_id

    return sub_char_dict

def cut_sentence_with_quotation_marks(text, is_chinese):
    p = re.compile('(“.*?”)|(".*?")|(``.*?\'\')')
    sents = []
    index = 0
    length = len(text)
    for i in p.finditer(text):
        temp = ''
        start = i.start()
        end = i.end()
        for j in range(index, start):
            temp += text[j]
        if temp.strip() != '':
            temp_list = [{'sentence': sent.text.strip(), 'mode': 'Narrative'} for sent in (sent_tokenizer_en(temp).sents
                                                                                           if not is_chinese else
                                                                                           sent_tokenizer_zh(
                                                                                               temp).sents) if
                         sent.text.strip()]
            sents += temp_list
        temp = ''
        for k in range(start, end):
            temp += text[k]
        if temp.strip() != '':
            sents.append({'sentence': temp.strip(), 'mode': 'Utterance'})
        index = end

    temp = ''
    for k in range(index, length):
        temp += text[k]
    if temp.strip() != '':
        temp_list = [{'sentence': sent.text.strip(), 'mode': 'Narrative'} for sent in (sent_tokenizer_en(temp).sents
                                                                                       if not is_chinese else
                                                                                       sent_tokenizer_zh(temp).sents) if
                     sent.text.strip()]
        sents += temp_list
    return sents


def split_text_by_quotes(text,quotes):
    fragments = []
    current_index = 0

    for quote in quotes:
        index = text.find(quote,current_index)
        if index != -1:
            if index != current_index:
                fragments.append({'type':'Narrative','text':text[current_index:index]})
            fragments.append({'type':'Dialogue','text':quote})
            current_index = index + len(quote)
    if current_index < len(text):
        fragments.append({'type':'Narrative','text':text[current_index:]})
    return fragments

import string
def remove_punctuation(text):
    """去除文本中的标点符号，但保留空格"""
    # 定义需要去除的标点符号
    punctuation = string.punctuation + '，。！？；：“”‘’【】《》{}【】()（）【】/／—'
    # 使用str.translate()方法删除标点符号
    text = text.translate(str.maketrans('', '', punctuation))
    return text

def compute_accuracy(records):
    if len(records) == 0:
        return {'acc':str(0), 'correct':str(len(list(filter(lambda x:x["correct"],records)))), 'total':str(len(records))}

    for record in records:
        predict = record['predict_speaker']
        label = record['label']
        char_dict = record['character']

        aliases = char_dict['id2names'][str(char_dict['name2id'][label])]
        aliases_lower = [(alias,''.join(alias.lower().split())) for alias in aliases]
        predict_lower = (predict,''.join(predict.lower().split()))
        matches = list(filter(lambda x:x[1]==predict_lower[1],aliases_lower))
        if len(matches)>0:
            record['correct'] = True
        else:
            record['correct'] = False
    tqdm.write(f'correct records:{len(list(filter(lambda x:x["correct"],records)))}/{len(records)}')
    acc=accuracy_score([True]*len(records),list(map(lambda x: x['correct'],records)))
    acc = {'acc':str(round(acc,3)), 'correct':str(len(list(filter(lambda x:x["correct"],records)))), 'total':str(len(records))}
    return acc

def find_words_with_non_alpha_boundaries(text, word, is_chinese):
    if not is_chinese:
        pattern = r'\b' + re.escape(word) + r'\b'
    else:
        pattern = re.escape(word)
    matches = list(re.finditer(pattern, text))
    return matches


def HasSpeaker(text, aliases, is_chinese):
    exist = False
    for alias in aliases:
        results = find_words_with_non_alpha_boundaries(text.lower(),alias.lower(),is_chinese)
        if len(results)>0:
            exist = True
            break
    return exist

def SplitMultiSpeakerParagraph(paragraph,char_dict):
    fragments = split_text_by_quotes(paragraph['paragraph'],list(map(lambda x:x['quote'],paragraph['utterance'])))
    quotes = list(filter(lambda x:x['type']=='Dialogue',fragments))
    for q,s in zip(quotes,paragraph['utterance']):
        q['speaker'] = s['speaker']

    groups,group = [],[]
    i = 0
    while i<len(fragments):
        if fragments[i]['type']=='Narrative':
            group.append(fragments[i])
        else:
            speaker = fragments[i]['speaker']
            group.append(fragments[i])
            if i+1<len(fragments)-1 and fragments[i+1]['type']=='Narrative' and HasSpeaker(
                    fragments[i+1]['text'], char_dict['id2names'][str(char_dict['name2id'][speaker])]):
                group.append(fragments[i+1])
                i+=1
            groups.append({'speaker':speaker,'group':group})
            group = []
        i+=1

    i,j=0,0
    merged_groups = []
    while j<len(groups):
        if groups[i]['speaker'] == groups[j]['speaker']:
            j+=1
        else:
            group = sum(map(lambda x:x['group'],groups[i:j]),[])
            speaker = groups[i]['speaker']
            merged_groups.append({'speaker':speaker,'group':group})
            i=j
    group = sum(map(lambda x: x['group'], groups[i:j]), [])
    speaker = groups[i]['speaker']
    merged_groups.append({'speaker': speaker, 'group': group})
    sub_paras = []
    quote_count = 0
    for i in range(len(merged_groups)):
        sub_para_text = ' '.join(map(lambda x:x['text'],merged_groups[i]['group']))
        group_quotes = list(map(lambda x:x['text'],filter(lambda z:z['type']=='Dialogue',merged_groups[i]['group'])))
        sub_para = copy.deepcopy(paragraph)
        sub_para['paragraph'] = sub_para_text
        sub_para['utterance'] = sub_para['utterance'][quote_count:quote_count+len(group_quotes)]
        sub_para['paragraph_index'] = f'{paragraph["paragraph_index"]}-{i}'
        sub_paras.append(sub_para)
        quote_count = quote_count+len(group_quotes)
    return  sub_paras



def ConstructSingleQuoteInstance1(data):
    constructed_insts = []
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    rule = r'((?<=[^a-zA-Z])|^){}((?=[^a-zA-Z])|$)'
    for inst in data:
        char_dict = inst['character']
        for utter_index in range(len(inst['dialogue'])) :
            utter = inst['dialogue'][utter_index]
            if utter['speaker'] == 'None' or len(tokenizer.tokenize(utter['paragraph']))>max_seq_len:
                continue
            context_para_num = 0
            context_len = len(tokenizer.tokenize(utter['paragraph']))
            origin_preceding_paras =  inst['preceding_paragraphs']+inst['dialogue'][:utter_index]
            origin_succeeding_paras = inst['dialogue'][utter_index+1:]+inst['succeeding_paragraphs']
            while (context_len<=max_seq_len) and (context_para_num<max(len(origin_preceding_paras),
                                                                  len(origin_succeeding_paras))):
                context_para_num += 1
                preceding_paras = origin_preceding_paras[max(-len(origin_preceding_paras),-context_para_num):]
                succeeding_paras = origin_succeeding_paras[:min(len(origin_succeeding_paras),context_para_num)]
                context_text = ' '.join(map(lambda x:x['paragraph'],preceding_paras+[utter]+succeeding_paras))
                context_len = len(tokenizer.tokenize(context_text))
            if context_len > max_seq_len:
                context_para_num -= 1
            preceding_paras = origin_preceding_paras[-context_para_num:]
            succeeding_paras = origin_succeeding_paras[:context_para_num]
            #speaker_id_set = set(map(lambda x:char_dict['name2id'][x['speaker']],utter['utterance']))
            sub_paras = SplitMultiSpeakerParagraph(utter, char_dict)
            for sub_idx, sub_para in enumerate(sub_paras):
                speaker = sub_para['utterance'][0]['speaker']
                constructed_inst = {
                    'preceding_paragraphs': preceding_paras + sub_paras[:sub_idx],
                    'succeeding_paragraphs': sub_paras[sub_idx + 1:] + succeeding_paras,
                    'dialogue': [sub_para],
                    'character': inst['character'],
                    'id': f'{inst["id"]}-{utter_index}-{sub_idx}'}
                context = ' '.join(map(lambda x:x['paragraph'],preceding_paras+[utter]+succeeding_paras))
                if (speaker in inst['character']['name2id']) and (
                any([re.search(rule.format(alias),context) != None for alias in
                     inst['character']['id2names'][str(inst['character']['name2id'][speaker])]])):
                    constructed_insts.append(constructed_inst)
    return constructed_insts


def ConstructSingleQuoteInstance(data, tokenizer, max_seq_len=500):
    constructed_insts = []
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    rule = r'((?<=[^a-zA-Z])|^){}((?=[^a-zA-Z])|$)'
    for inst in data:
        for utter_index in range(len(inst['dialogue'])) :
            utter = inst['dialogue'][utter_index]
            if len(utter['utterance']) == 0 or len(tokenizer.tokenize(utter['paragraph']))>max_seq_len:
                continue

            context_para_num = 0
            context_len = len(tokenizer.tokenize(utter['paragraph']))
            if 'original_instance' in utter:
                origin_inst = utter['original_instance']
                origin_preceding_paras = origin_inst['preceding_paragraphs']
                origin_succeeding_paras = origin_inst['succeeding_paragraphs']
                char_dict = origin_inst['character']
            else:
                origin_inst = inst
                origin_preceding_paras = origin_inst['preceding_paragraphs'] + origin_inst['dialogue'][:utter_index]
                origin_succeeding_paras = origin_inst['dialogue'][utter_index + 1:] + origin_inst[
                    'succeeding_paragraphs']
                char_dict = origin_inst['character']


            while (context_len<=max_seq_len) and (context_para_num<max(len(origin_preceding_paras),
                                                                  len(origin_succeeding_paras))):
                context_para_num += 1
                preceding_paras = origin_preceding_paras[max(0,len(origin_preceding_paras)-context_para_num):]
                succeeding_paras = origin_succeeding_paras[:min(len(origin_succeeding_paras),context_para_num)]
                context_text = ' '.join(map(lambda x:x['paragraph'],preceding_paras+[utter]+succeeding_paras))
                context_len = len(tokenizer.tokenize(context_text))
            right_context_para_num,left_context_para_num = context_para_num,context_para_num
            #print(f"context len:{context_len}>max len:{max_seq_len}")
            #print(f"context_para_num:{context_para_num},para_num:{len(preceding_paras+[utter]+succeeding_paras)}")
            if context_len > max_seq_len:
                right_context_para_num -= (1 if right_context_para_num>0 else 0)
                preceding_paras = origin_preceding_paras[max(0,len(origin_preceding_paras)-context_para_num):]
                succeeding_paras = origin_succeeding_paras[:min(len(origin_succeeding_paras), right_context_para_num)]
                context_text = ' '.join(map(lambda x: x['paragraph'], preceding_paras + [utter] + succeeding_paras))
                context_len = len(tokenizer.tokenize(context_text))
            if context_len>max_seq_len:
                left_context_para_num -= (1 if left_context_para_num>0 else 0)
                preceding_paras = origin_preceding_paras[max(0, len(origin_preceding_paras)-left_context_para_num):]
                succeeding_paras = origin_succeeding_paras[:min(len(origin_succeeding_paras), right_context_para_num)]
                context_text = ' '.join(map(lambda x: x['paragraph'], preceding_paras + [utter] + succeeding_paras))
                context_len = len(tokenizer.tokenize(context_text))
            #print(f"left_context_para_num:{left_context_para_num},right_context_para_num:{right_context_para_num},"
            #      f"para_num:{len(preceding_paras+[utter]+succeeding_paras)}")
            assert context_len<=max_seq_len,f"invalid length. context len:{context_len}>max len:{max_seq_len}"

            preceding_paras = origin_preceding_paras[-left_context_para_num:]
            succeeding_paras = origin_succeeding_paras[:right_context_para_num]
            #speaker_id_set = set(map(lambda x:char_dict['name2id'][x['speaker']],utter['utterance']))
            #sub_paras = SplitMultiSpeakerParagraph(utter, char_dict)
            sub_paras = [utter]
            for sub_idx, sub_para in enumerate(sub_paras):
                constructed_inst = {
                    'preceding_paragraphs': preceding_paras + sub_paras[:sub_idx],
                    'succeeding_paragraphs': sub_paras[sub_idx + 1:] + succeeding_paras,
                    'dialogue': [sub_para],
                    'character': char_dict,
                    'original_id':origin_inst['id'],
                    'id': len(constructed_insts)}
                context = ' '.join(map(lambda x:x['paragraph'],preceding_paras+[utter]+succeeding_paras))
                remove=False
                for quote_dict in sub_para['utterance']:
                    #if quote_dict['speaker'] not in inst['character']['name2id']:
                    #    remove = True
                    if (quote_dict['speaker'] not in inst['character']['name2id']) or all([re.search(rule.format(alias),context) is None for alias in
                     inst['character']['id2names'][str(inst['character']['name2id'][quote_dict['speaker']])]]):
                        remove = True
                #if (all([quote_dict['speaker'] in inst['character']['name2id'] for quote_dict in sub_para['utterance']])
                #        and any([re.search(rule.format(alias),context) != None for alias in
                #     inst['character']['id2names'][str(inst['character']['name2id'][speaker])]])):
                if not remove:
                    constructed_insts.append(constructed_inst)
    return constructed_insts

