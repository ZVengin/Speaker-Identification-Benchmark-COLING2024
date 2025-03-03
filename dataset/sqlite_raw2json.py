import json,os,copy
import xml.dom.minidom
from xml.dom.minidom import Text,Element
from tqdm import tqdm
from unify_dataset import ExtractDialogue,GetSubCharDict,AddUnkToken

pronoun_list = ['i', 'you', 'he', 'she', 'it', 'they', 'we']

def get_subtree_text(node):
    if type(node)==Text:
        return node.data
    text = ''
    for sub_node in node.childNodes:
        text += get_subtree_text(sub_node)
    return text


def get_characters(anno_path):
    dom = xml.dom.minidom.parse(anno_path)
    root = dom.documentElement
    char_dict = {}
    for character in root.getElementsByTagName('character'):
        name = character.getAttribute('name')
        aliases = list(map(lambda x:x.strip(),character.getAttribute('aliases').split(';')))
        id = character.getAttribute('id')
        gender = 'Female' if character.getAttribute('gender') =='female' else 'Male'
        char_dict[name]={'id':id,'aliases':aliases,'gender':gender}
    return char_dict

def annotate_paras(book_name,anno_path,para_out_path,char_out_path):
    omission = 0
    char_dict = get_characters(anno_path)
    dom = xml.dom.minidom.parse(anno_path)
    root = dom.documentElement
    root = root.getElementsByTagName('text')[0]
    if len(root.getElementsByTagName("chapter"))>0:
        childNodes = []
        for chapter in root.getElementsByTagName("chapter"):
            childNodes+=list(chapter.childNodes)
    else:
        childNodes = list(root.childNodes)
    paras = []
    quotes = []
    mentions = []

    para=''
    quote=[]
    mention=[]
    for node in childNodes:
        if type(node)==Text:
            #para+=node.data
            node_text = node.data

            sub_paras = node_text.split('\n\n')
            if len(sub_paras)>1:
                sub_paras[0] = para+sub_paras[0]
                paras += sub_paras[:-1]
                quotes += ([quote]+[[]]*(len(sub_paras)-2))
                mentions += ([mention]+[[]]*(len(sub_paras)-2))
                #if "Come, not done howling yet" in sub_paras[0]:
                #    print(f'quotes:{quotes[-5:]}')
                para = sub_paras[-1]
                quote = []
                mention = []
                print(f'para:{len(paras)},quote:{len(quotes)}')
            else:
                para += sub_paras[0]
        else:
            node_text = get_subtree_text(node)
            if node.tagName == 'quote':
                quote.append({'id':node.getAttribute('id'),'speaker':node.getAttribute('speaker'),'quote':node_text})
            elif node.tagName == 'mention':
                mention.append({'id':node.getAttribute('id'),'connection':node.getAttribute('connection'),'mention':node_text})
            para += node_text

    if para.strip():
        paras.append(para)
        quotes.append(quote)
        mentions.append(mention)
    para_quote_pairs=list(zip(paras,quotes,mentions))
    print(f'quotes:{sum(map(lambda x:len(x),quotes))}')
    print(f'paras:{len(paras)},quotes:{len(quotes)},mentions:{len(mentions)}')
    para_dicts = []
    for i in range(len(para_quote_pairs)):
        para,para_quotes,para_mentions = para_quote_pairs[i]
        offset = len('\n'.join(map(lambda x:x[0],para_quote_pairs[:i])))
        quote_spans = []
        quote_texts = []
        quote_ids=[]
        quote_types=[]
        speaker_cues =[]
        #speakers = [para_quote['speaker'] for para_quote in para_quotes]
        #filt_speakers = list(filter(lambda x:x in char_dict,speakers))
        #if len(set(filt_speakers))>1 or len(set(filt_speakers))==0:
        #    para_quotes = []
        #    speaker = 'None'
        #    mode = 'Narrative'
        #    if len(set(filt_speakers))==0 and len(set(speakers))!=0:
        #        print(f'speakers:{speakers}')
        #        omission+=1
        #else:
        #    speaker = char_dict[filt_speakers[0]]['aliases'][0]
        #    mode='Dialogue'
        para_quotes = list(filter(lambda x:x['speaker'] in char_dict,para_quotes))
        speakers = list(map(lambda x:char_dict[x['speaker']]['aliases'][0],para_quotes))
        genders = list(map(lambda x:char_dict[x['speaker']]['gender'],para_quotes))
        if len(speakers) == 0:
            mode='Narrative'
        else:
            mode = 'Dialogue'
        for para_quote in para_quotes:
            para_quote_text = para_quote['quote']
            quote_texts.append(para_quote_text)
            quote_ids.append(f'{book_name}-{i}-{para_quote["id"]}')

            para_quote_start = para.find(para_quote_text)
            if para_quote_start == -1:
                para_quote_start = None
                para_quote_end = None

            else:
                para_quote_end = para_quote_start+len(para_quote_text)
            quote_spans.append((para_quote_start,para_quote_end))
            quote_cues = list(filter(lambda x:para_quote['id'] in x['connection'].strip().split(','),para_mentions))
            quote_cues = list(map(lambda x:x['mention'],quote_cues))
            assert len(quote_cues)<=1,'more than one mention connected to the quote'
            if len(quote_cues)==0:
                quote_type= 'Implicit'
            elif quote_cues[0].strip() in char_dict[para_quote['speaker']]['aliases']:
                quote_type = 'Explicit'
            elif quote_cues[0].strip().lower() in pronoun_list:
                quote_type= 'Anaphoric(pronoun)'
            else:
                quote_type='Anaphoric(other)'

            speaker_cues.append(quote_cues)
            quote_types.append(quote_type)
        quote_dict_keys = ['quote','quote_id','quote_span','quote_type','speaker','speaker_gender','speaker_cue']
        print([(k,len(v)) for k,v in zip(quote_dict_keys,[quote_texts,quote_ids,quote_spans,quote_types,speakers,genders,speaker_cues])])
        quote_dicts = []
        for group in zip(quote_texts,quote_ids,quote_spans,quote_types,speakers,genders,speaker_cues):
            quote_dicts.append(dict(zip(quote_dict_keys,group)))
        para_dict = {
            'paragraph':para,
            'paragraph_index':f'{book_name}-{i}',
            'offset':[offset,offset+len(para)],
            #'utterance_span':quote_spans,
            'utterance':quote_dicts,#quote_texts,
            #'speaker':speakers,
            #'utterance_id':quote_ids,
            #'utterance_type':quote_types,
            'book_name': book_name,
            'mode':mode,
            #'speaker_gender':genders,
            #'speaker_cue':speaker_cues
        }
        para_dicts.append(para_dict)

    new_char_dict = {'id2names':{},'name2id':{},'id2gender':{}}
    for _,id2name_dict in char_dict.items():
        new_char_dict['id2names'][id2name_dict['id']]=id2name_dict['aliases']
        new_char_dict['id2gender'][id2name_dict['id']]=id2name_dict['gender']

        for alias in id2name_dict['aliases']:
            new_char_dict['name2id'][alias] = eval(id2name_dict['id'])

    with open(para_out_path,'w') as f:
        json.dump(para_dicts,f,indent=2)

    with open(char_out_path,'w') as f:
        json.dump(new_char_dict,f,indent=2)

    return para_dicts,char_dict



# Generate Training and Dev data for SQLITE
## The dev set is the chapters of PAP from 27-33
## The test set is the chapters of PAP from 19-26
## The training set is the remaining chapters

def sqlite2json(book_dir,proc_dir):
    os.makedirs(proc_dir,exist_ok=True)
    test_chapters = list(range(18,26))
    dev_chapters =list(range(26,33))
    train_chapters = list(set(range(0,61))-set(test_chapters)-set(dev_chapters))

    dev_paras,train_paras =[],[]
    char_dicts = []
    for i in range(61):
        book_name = f'pp-{i}'
        chapter_path = os.path.join(book_dir,'PAP_chapters',f'{book_name}-annotated.xml')
        para_path = os.path.join(book_dir,'PAP_chapters',f'{book_name}-annotated.json')
        char_path = os.path.join(book_dir,'PAP_chapters',f'{book_name}-character.json')
        if not os.path.exists(chapter_path):
            continue
        chapter_paras,chapter_char_dict = annotate_paras(book_name,chapter_path,para_path,char_path)
        if i in dev_chapters:
            dev_paras+=chapter_paras
        elif i in train_chapters:
            train_paras += chapter_paras
        char_dicts.append(chapter_char_dict)

    char_dict={'id2names':{},'name2id':{},'id2gender':{}}
    temp_dict = {}
    for cd in char_dicts:
        temp_dict.update(cd)
    for cid,(k,v) in enumerate(temp_dict.items()):
        char_dict['id2names'][str(cid)] = v['aliases']
        char_dict['id2gender'][str(cid)] = v['gender']
        for alias in v['aliases']:
            char_dict['name2id'][alias] = cid

    for set_name in ['train','dev']:
        set_path = os.path.join(proc_dir,f'PAP_{set_name}_paragraphs.json')
        with open(set_path,'w') as f:
            json.dump(train_paras if set_name =='train' else dev_paras,f,indent=2)
        char_path = os.path.join(proc_dir,f'PAP_{set_name}_character.json')
        with open(char_path,'w') as f:
            json.dump(char_dict,f,indent=2)

    for dataset_name in ['PAP_test', 'Emma', 'Steppe']:
        anno_path = os.path.join(book_dir, f'{dataset_name}.xml')
        para_path = os.path.join(proc_dir, f'{dataset_name}_paragraphs.json')
        char_path = os.path.join(proc_dir, f'{dataset_name}_character.json')
        annotate_paras(dataset_name, anno_path, para_path, char_path)


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
        #text = ' '.join(list(map(lambda x: x['paragraph'], dial_paras)))
        for para_index in range(len(dial_paras)):
            offset = len(' '.join(list(map(lambda x: x['paragraph'], dial_paras[:para_index]))))
            dial_paras[para_index]['offset'] = [offset, offset + len(dial_paras[para_index]['paragraph'])]
        for para_index,para in enumerate(dialogue['dialogue']):
            #para['utterance_id'] = [f'{dest_count}-{para_index}-{utter_index}' for utter_index in
            #                        range(len(para['utterance_id']))]
            #utter_offsets = []
            start_pos=0
            for utter in para['utterance']:
                offset = para['paragraph'].find(utter['quote'],start_pos)
                utter['quote_span'] = [offset, offset + len(utter['quote']) if offset != -1 else -1]
                start_pos = offset+len(utter['quote']) if offset != -1 else 0
                #utter_offsets.append([offset, offset + len(utter) if offset != -1 else -1])
            #para['utterance_span'] = utter_offsets
        sub_char_dict = GetSubCharDict(dialogue,char_dict)
        sub_char_dict = AddUnkToken(sub_char_dict)
        dialogue['character'] = sub_char_dict
        dialogue['id'] = dest_count
        dest_count += 1
    dest_data = dialogues

    return dest_data

def ProcessSQLITE(sour_dir,target_dir):

    test_data = []

    os.makedirs(target_dir,exist_ok=True)
    t = tqdm(["PAP_train",'PAP_dev','PAP_test',"Emma","Steppe"])
    inst_index=0
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
                inst['id'] = inst_index
                inst_index += 1
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

book_dir = './raw_data_new/SQLITE'
proc_dir='./proc_data_new2/SQLITE'
sqlite2json(book_dir,proc_dir)
ProcessSQLITE(proc_dir,proc_dir)