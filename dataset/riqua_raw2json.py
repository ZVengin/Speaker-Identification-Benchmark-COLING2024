import copy,json,os,random,re
from collections import defaultdict
from tqdm import tqdm
from brat_parser import get_entities_relations_attributes_groups
from unify_dataset import AddUnkToken,GetCharDictEn,ExtractDialogue,GetSubCharDict, nlp_en, cut_sentence_with_quotation_marks

from transformers import AutoTokenizer
from spacy.lang.en import English

#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sent_tokenizer_en = English()
sent_tokenizer_en.add_pipe(sent_tokenizer_en.create_pipe('sentencizer'))


random.seed(123)
tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')



def ReadAnnoFile(anno_path):
    annos = []
    ents, rels, atts, gs = get_entities_relations_attributes_groups(anno_path)
    cues = set([rel.obj for rel_id, rel in rels.items() if rel.type == 'Cueing'])
    anno_count = 0
    for rel_id in rels.keys():
        rel = rels[rel_id]
        if rel.type == 'Speaker':
            quote = ents[rel.obj].text
            quote_span = ents[rel.obj].span
            speaker = ents[rel.subj].text
            speaker_span = ents[rel.subj].span
            anno = {
                'id': rel_id,
                'utterance': quote,
                'utterance_span': list(quote_span),
                'utterance_type': 'Explicit' if rel.obj in cues else 'Implicit',
                'speaker': speaker,
                'speaker_span': list(speaker_span[0]),
                'utterance_id': f'Q{anno_count}'
            }
            annos.append(anno)
            anno_count += 1

    return annos


def GetParagraphs(text_path, anno_path):
    book_name = os.path.basename(text_path)
    annos = ReadAnnoFile(anno_path)
    with open(text_path, 'r') as f:
        text = f.read()
    paras = cut_sentence_with_quotation_marks(text, False)
    paragraphs = []
    start_pos = 0
    for para_idx, para in enumerate(paras):
        para_text = para['sentence']
        para_offset = text.find(para_text, start_pos)
        para_span = [para_offset, para_offset + len(para_text)]
        mode = 'Dialogue' if para['mode'] == 'Utterance' else 'Narrative'
        para = {
            'paragraph': para_text,
            'paragraph_index': para_idx,
            'offset': para_span,
            'utterance': [],
            'book_name': book_name,
            'mode': mode
        }
        paragraphs.append(para)
        start_pos = para_span[1]

    for para in paragraphs:
        para_span = para['offset']
        para_text = para['paragraph']
        for anno in annos:
            anno_span = anno['utterance_span'][0]
            quote = anno['utterance']
            if para_span[0] <= anno_span[0] and para_span[1] >= anno_span[1]:
                if para_text.find(quote) != -1:
                    quote_dict = {
                        'quote': para_text,
                        'quote_span': para_span,
                        'quote_id': f'{book_name}_{anno["utterance_id"]}',
                        'quote_type': anno["utterance_type"],
                        'speaker': anno["speaker"],
                        'speaker_gender': 'None',
                        'speaker_cue': 'None'
                    }
                    para['utterance'].append(quote_dict)
    return paragraphs


"""
def GetParagraphs(text_path, anno_path):
    annos = ReadAnnoFile(anno_path)
    with open(text_path, 'r') as f:
        text = f.read()
    book_name = os.path.basename(text_path)
    paragraphs = []
    annos = sorted(annos, key=lambda x: x['utterance_span'][0][0])
    start_pointer = 0
    for anno_idx,anno in enumerate(annos):
        anno_span = anno['utterance_span'][0]
        pre_text = text[start_pointer:anno_span[0]]
        if pre_text.strip():
            sentences = []
            for i, sent in enumerate(cut_sentence_with_quotation_marks(pre_text, False)):
                if sent.strip():
                    sentences.append(sent)
                else:
                    sentences[-1] += sent
            for i in range(len(sentences)):
                sent_start = text.find(sentences[i], start_pointer, anno_span[0])
                sent_end = sent_start + len(sentences[i])
                sent_index = len(paragraphs)
                start_pointer = sent_end
                paragraph = {
                    'paragraph_index': f'{book_name}_{sent_index}',
                    'paragraph': sentences[i],
                    'offset': [sent_start, sent_end],
                    'utterance': [],
                    'book_name': book_name,
                    'mode':'Narrative'
                }
                paragraphs.append(paragraph)
        quote_dicts=[{
            'quote':anno['utterance'],
            'quote_span':anno['utterance_span'],
            'quote_id':f'{book_name}_{anno["utterance_id"]}',
            'quote_type':anno['utterance_type'],
            'speaker':anno['speaker'],
            'speaker_gender':'None',
            'speaker_cue':'None'
        }]
        if anno_idx<len(annos)-1:
            subseq_text = text[anno_span[1]:]
        utterance = anno['utterance']
        utter_context = ' '.join(paragraphs[-2:]+[])
        paragraph = {
            'paragraph_index': f'{book_name}_{len(paragraphs)}',
            'paragraph': anno['utterance'],
            'offset': anno['utterance_span'],
            'utterance': quote_dicts,
            'book_name': book_name,
            'mode':'Dialogue'
        }
        paragraphs.append(paragraph)
        start_pointer = anno_span[1]
    if start_pointer < len(text):
        pre_text = text[start_pointer:]
        sentences = cut_sentence_with_quotation_marks(pre_text, False)
        for i in range(len(sentences)):
            sent_start = text.find(sentences[i], start_pointer)
            sent_end = sent_start + len(sentences[i])
            sent_index = len(paragraphs)
            paragraph = {
                'paragraph_index': f'{book_name}_{sent_index}',
                'paragraph': sentences[i],
                'offset': [sent_start, sent_end],
                'utterance': [],
                'book_name': book_name,
                'mode':'Narrative'
            }
            paragraphs.append(paragraph)
    return paragraphs
"""

def BuildRIQUACharDict(paragraphs, pronoun_path):
    # here we use lower() function to convert the name into lower case
    # due to that some speakers are annotated with pronouns
    # in this case, "she" or "She" should be the same speaker.
    # to make the context consistent with the speaker, the context should also be converted into lower case
    # pay attention to the case where the pronouns could be matched to spans inside some words
    title_pattern = r"(?:Mr\.|Mrs\.|Ms\.|Miss\.|Miss|Mrs|Mr|Ms)( *[A-Za-z]+)"
    pattern = re.compile(title_pattern)
    speakers = list(set(map(lambda x: x['speaker'], sum(list(map(lambda z:z['utterance'],paragraphs)),[]))))
    text = ' '.join(list(map(lambda x:x['paragraph'],paragraphs)))
    char_dict = GetCharDictEn(nlp_en,[text],pronoun_path)[0]
    #old_char_dict = copy.deepcopy(char_dict)
    filt_speakers = []
    # merge the speaker annotated in the corpus into the character dictionary
    for speaker in speakers:
        if len(speaker.split())>5 or speaker == 'None':
            continue
        find = False
        for mention in char_dict['name2id'].keys():
            speaker_match = pattern.search(speaker)
            mention_match = pattern.search(mention)
            if speaker == mention:
                find = True
                break
            elif speaker.lower().strip() == mention.lower().strip():
                find = True
                filt_speakers.append((char_dict['name2id'][mention],speaker))
                break
            elif speaker_match!=None and speaker_match.group(1).lower().strip() == mention.lower().strip():
                find = True
                filt_speakers.append((char_dict['name2id'][mention], speaker))
                break
            elif mention_match!=None and mention_match.group(1).lower().strip()==speaker.lower().strip():
                find = True
                filt_speakers.append((char_dict['name2id'][mention], speaker))
                break
        if not find:
            filt_speakers.append((-1,speaker))
    for speaker_id,speaker in filt_speakers:
        if speaker_id != -1:
            char_dict['id2names'][str(speaker_id)].append(speaker)
            char_dict['name2id'][speaker] = speaker_id
            char_dict['id2gender'][str(speaker_id)] = 'None'
        else:
            exist = False
            for name,nid in char_dict['name2id'].items():
                if speaker.lower().strip() == name.lower().strip():
                    exist = True
                    break
            if exist:
                char_dict['name2id'][speaker] = nid
                char_dict['id2names'][str(nid)].append(speaker)
                char_dict['id2gender'][str(nid)] = 'None'
            else:
                char_dict['name2id'][speaker] = len(char_dict['id2names'])
                char_dict['id2gender'][str(len(char_dict['id2names']))] = 'None'
                char_dict['id2names'][str(len(char_dict['id2names']))] = [speaker]


    return char_dict

def riqua2json(book_dir,proc_dir):
    book_dict = defaultdict(dict)
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)

    for file_name in os.listdir(book_dir):
        if file_name.startswith('.'):
            continue
        book_name, suffix = file_name.split('.')
        if suffix == 'ann':
            book_dict[book_name]['anno'] = file_name
        else:
            book_dict[book_name]['text'] = file_name
        print(book_dict)
    test_set = {k: book_dict[k] for k in random.sample(book_dict.keys(), int(len(book_dict) * 0.2))}
    train_set = {k: book_dict[k] for k in book_dict.keys() if k not in test_set}

    for book_name, file_dict in train_set.items():
        print(file_dict)
        text_path = os.path.join(book_dir, file_dict['text'])
        anno_path = os.path.join(book_dir, file_dict['anno'])
        paras = GetParagraphs(text_path, anno_path)
        file_dict['paragraphs'] = paras

    train_para_file = os.path.join(proc_dir, 'train_paragraphs.json')
    with open(train_para_file, 'w') as f:
        json.dump(train_set, f, indent=2)

    for book_name, file_dict in test_set.items():
        try:
            text_path = os.path.join(book_dir, file_dict['text'])
            anno_path = os.path.join(book_dir, file_dict['anno'])
        except Exception as e:
            print(file_dict)
            raise Exception(e)
        paras = GetParagraphs(text_path, anno_path)
        file_dict['paragraphs'] = paras

    test_para_file = os.path.join(proc_dir, 'test_paragraphs.json')
    with open(test_para_file, 'w') as f:
        json.dump(test_set, f, indent=2)


def FormatRIQUA(sour_path, target_path):
    with open(sour_path, 'r') as f:
        sour_data = json.load(f)

    #pronoun_path = os.path.join(SCRIPT_DIR,'../pronoun_list.txt')
    pronoun_path =  '../pronoun_list.txt'
    dest_data = []
    dest_count = 0
    for k in tqdm(sour_data.keys(),desc="FormatRIQUA progress"):
        paragraphs = sour_data[k]['paragraphs']
        char_dict = BuildRIQUACharDict(paragraphs,pronoun_path)
        dialogues = ExtractDialogue(paragraphs, context_len=10, interval_paragraph=1)
        dialogues = copy.deepcopy(dialogues)
        for dialogue in dialogues:
            dial_paras = dialogue['preceding_paragraphs']+dialogue['dialogue']+dialogue['succeeding_paragraphs']
            #text = ' '.join(list(map(lambda x: x['paragraph'],dial_paras)))
            for para_index in range(len(dial_paras)):
                offset = len(' '.join(list(map(lambda x:x['paragraph'],dial_paras[:para_index]))))
                dial_paras[para_index]['offset'] = [offset,offset+len(dial_paras[para_index]['paragraph'])]
            for para_index,para in enumerate(dialogue['dialogue']):
                #para['utterance_id'] = [f'{dest_count}-{para_index}-{utter_index}'
                #                       for utter_index in range(len(para['utterance_id']))]
                #utter_offsets = []
                for utter in para['utterance']:
                    offset = para['paragraph'].find(utter['quote'])
                    utter['quote_span'] = [offset,offset+len(utter['quote']) if offset!=-1 else -1]
                    #utter['quote_id'] = f'{dest_count}-{para_index}-{utter_index}'
                #para['utterance_span'] = utter_offsets
            sub_char_dict = GetSubCharDict(dialogue,char_dict)
            sub_char_dict = AddUnkToken(sub_char_dict)
            dialogue['character'] = sub_char_dict
            dialogue['id'] = dest_count
            dest_count+=1

        dest_data += dialogues

    tqdm.write(f"# instance for FormatRIQUA:{len(dest_data)}")
    with open(target_path, 'w') as f:
        json.dump(dest_data, f, indent=2)


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

book_dir = 'raw_data_new/RIQUA'
proc_dir='proc_data_new/RIQUA'

riqua2json(book_dir,proc_dir)
ProcessRIQUA(proc_dir,proc_dir)