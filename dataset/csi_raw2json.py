import copy,re
import json,os,sys,random,spacy
from collections import Counter
from unify_dataset import GetClusters,cut_sentence_with_quotation_marks,GetSubCharDict,AddUnkToken
from unify_dataset import GetQuoteTypeForCorpus
from tqdm import tqdm
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR,'..'))
from utils import split_text_by_quotes
nlp_zh = spacy.load("zh_core_web_sm")



def GetCharDictZh(data):
    batch_size = 16
    batch_text = []
    batch_inst = []
    character_set = set()
    for inst in tqdm(data, desc="Generate Character"):
        text = inst['paragraphs'][0]['context']
        batch_text.append(text)
        batch_inst.append(inst)
        if len(batch_text) % batch_size == 0:
            docs = list(nlp_zh.pipe(batch_text))
            for inst, doc in zip(batch_inst, docs):
                characters = []
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        characters.append(ent.text)
                answers_dict = list(map(lambda x: x['answers'], inst['paragraphs'][0]['qas']))
                ans = []
                for answer_dict in answers_dict:
                    name_count = Counter(list(map(lambda x: x['text'], answer_dict)))
                    name = list(name_count.most_common(1))[0][0]
                    ans.append(name)
                characters += ans
                characters = list(set(characters))
                characters = list(filter(lambda x:len(x)>1 and len(x)<5,characters))
                character_set = character_set.union(set(characters))
            batch_text = []
            batch_inst = []
    char_dict = {'id2names':{},'name2id':{},'id2gender':{}}
    for char_id, char in enumerate(character_set):
        char_dict['id2names'][str(char_id)] = [char]
        char_dict['name2id'][char] = char_id
        char_dict['id2gender'][str(char_id)] = 'None'
    # leave the id 10000 for None
    # this has been implemented in AddUnkToken function
    #if len(character_set)>CharMaskId:
    #    temp = char_dict['id2names'][str(CharMaskId)]
    #    temp_id = len(character_set)
    #    char_dict['id2names'][str(temp_id)] = temp
    #    for n in temp:
    #        char_dict['name2id'][n] = temp_id

    char_dict = AddUnkToken(char_dict)
    print(f'char_dict:{char_dict}')
    return char_dict





def FormatCSI(sour_path, target_path, char_dict):
    with open(sour_path, 'r') as f:
        origin_data = json.load(f)['data']
    dest_data = []
    for origin_inst in tqdm(origin_data,desc="FormatCSI progress"):
        index = origin_inst['paragraphs'][0]['id']
        text = origin_inst['paragraphs'][0]['context']
        qas = list(map(lambda x: x['question'], origin_inst['paragraphs'][0]['qas']))
        answers_dict = list(map(lambda x: x['answers'], origin_inst['paragraphs'][0]['qas']))
        ans = []
        for answer_dict in answers_dict:
            name_count = Counter(list(map(lambda x:x['text'],answer_dict)))
            name = list(name_count.most_common(1))[0][0]
            ans.append(name)
        # cut the input text into fragments according to the quotes
        # input_text =　[{'type':'Narrative','text':'frag_1'},{'type':'Dialogue','text':'quote_1'},...,]
        fragments = split_text_by_quotes(text, qas)
        qas_count = 0
        for fragindex in range(len(fragments)):
            fragment = fragments[fragindex]
            if fragment['type'] == 'Dialogue':
                preceding_paragraphs = []
                for subfragindex in range(fragindex):
                    subfragment = fragments[subfragindex]
                    offset = len(' '.join(map(lambda x: x['text'], fragments[:subfragindex])))
                    preceding_paragraph = {
                        'paragraph_index': f'{index}-{subfragindex}',
                        'paragraph': subfragment['text'],
                        'offset': [offset, offset + len(subfragment['text'])],
                        'utterance': [],
                        'book_name': 'CSI',
                        'mode':'Narrative'
                    }
                    preceding_paragraphs.append(preceding_paragraph)
                sents = cut_sentence_with_quotation_marks(fragment['text'],True)
                quotes = list(map(lambda y:y['sentence'],filter(lambda x:x['mode']=='Utterance',sents)))
                quotes_span = []
                quote_dicts = []
                for qidx,quote in enumerate(quotes):
                    quote_start = fragment['text'].find(quote)
                    assert quote_start!= -1, 'cannot find quote in the given text'
                    quote_dicts.append({
                        'quote':quote,
                        'quote_span':[quote_start,quote_start+len(quote)],
                        'quote_type':'None',
                        'quote_id':f'{index}-{fragindex}-{qidx}',
                        'speaker':ans[qas_count],
                        'speaker_gender':'None',
                        'speaker_cue':'None'
                    })
                if len(quote_dicts)<=0:
                    quote_dicts.append({
                        'quote': fragment['text'],
                        'quote_span': [0, len(fragment['text'])],
                        'quote_type': 'None',
                        'quote_id': f'{index}-{fragindex}-0',
                        'speaker': ans[qas_count],
                        'speaker_gender': 'None',
                        'speaker_cue': 'None'
                    })
                    #quotes_span.append([quote_start,quote_start+len(quote)])
                offset = len(' '.join(map(lambda x: x['text'], fragments[:fragindex])))
                dial_paragraph = {
                    'paragraph_index': f'{index}-{fragindex}',
                    'paragraph': fragment['text'],
                    'offset': [offset, offset + len(fragment['text'])],
                    'utterance': quote_dicts,
                    'book_name': 'CSI',
                    'mode':'Dialogue'
                }
                succeeding_paragraphs = []
                for subfragindex in range(fragindex+1,len(fragments),1):
                    subfragment = fragments[subfragindex]
                    offset = len(' '.join(map(lambda x: x['text'], fragments[:subfragindex])))
                    succeeding_paragraph = {
                        'paragraph_index': f'{index}-{subfragindex}',
                        'paragraph': subfragment['text'],
                        'offset': [offset, offset + len(subfragment['text'])],
                        'utterance': [],
                        'book_name': 'CSI',
                        'mode':'Narrative'
                    }
                    succeeding_paragraphs.append(succeeding_paragraph)
                #para_idxs = list(map(lambda x:x['paragraph_index'],preceding_paragraphs+[dial_paragraph]+succeeding_paragraphs))
                #dial_paragraph['original_paragraph_indexs'] = para_idxs

                dest_inst = {
                    'id':len(dest_data),
                    'preceding_paragraphs':preceding_paragraphs,
                    'dialogue':[dial_paragraph],
                    'succeeding_paragraphs':succeeding_paragraphs,
                    'character':None
                }

                sub_char_dict = GetSubCharDict(dest_inst,char_dict,is_chinese=True)
                sub_char_dict = AddUnkToken(sub_char_dict)
                dest_inst['character'] = sub_char_dict
                dest_inst['dialogue'][0]['original_instance'] = copy.deepcopy(dest_inst)

                if dest_inst['dialogue'][0]['utterance'][0]['speaker'] in char_dict['name2id']:
                    dest_data.append(dest_inst)
                qas_count+=1

    dest_data = GetQuoteTypeForCorpus(dest_data,is_chinese=True)
    tqdm.write(f'# instances for FormatCSI:{len(dest_data)}')
    with open(target_path, 'w') as f:
        json.dump(dest_data, f, indent=2)


def csi2json(sour_dir,target_dir):

    os.makedirs(target_dir,exist_ok=True)
    t = tqdm(['train','test'])
    sour_data = []
    for set_name in t:
        sour_path = os.path.join(sour_dir, f'{set_name}_v1.json')
        with open(sour_path,'r') as f:
            sour_data += json.load(f)['data']
    char_dict = GetCharDictZh(sour_data)

    for set_name in t:
        t.set_description(desc=f"Process CSI: {set_name}")
        sour_path = os.path.join(sour_dir, f'{set_name}_v1.json')
        target_path = os.path.join(target_dir, f'{set_name}_paragraphs.json')
        FormatCSI(sour_path, target_path,char_dict)


def ProcessCSI(sour_dir, target_dir):
    sour_path = os.path.join(sour_dir, f'train_paragraphs.json')
    with open(sour_path,'r') as f:
        data = json.load(f)
    random.seed(123)
    data = random.sample(data,k=len(data))
    dev_size = int(len(data)*0.1)
    dev_set = data[:dev_size]
    dev_set = GetClusters(dev_set)
    train_set = data[dev_size:]
    train_set = GetClusters(train_set)
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
    test_set = GetClusters(test_set)
    target_path = os.path.join(target_dir,'test.json')
    with open(target_path,'w') as f:
        json.dump(test_set,f,indent=2)

sour_dir = 'raw_data_new/CSI'
target_dir = 'proc_data_new2/CSI'
csi2json(sour_dir,target_dir)
ProcessCSI(target_dir,target_dir)