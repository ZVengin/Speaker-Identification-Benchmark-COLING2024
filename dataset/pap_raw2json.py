import os,re,json,copy
from tqdm import tqdm
from transformers import AutoTokenizer
from unify_dataset import AddUnkToken,ExtractDialogue,GetSubCharDict, cut_sentence_with_quotation_marks,GetQuoteTypeForCorpus

MAX_CONTEXT_LEN=500
tokenizer = AutoTokenizer.from_pretrained('roberta-large')


def PreprocessText(text):
    return ' '.join(text.replace('\n', ' ').split())


def ReadCharFile(char_file_path):
    char_dict = {'name2alias': {}, 'name2gender': {},'alias2name':{}}
    with open(char_file_path, 'r') as f:
        lines = f.read().split('\n')
        lines = [[e.strip() for e in line.split(';')] for line in lines]
    for line in lines:
        char_dict['name2alias'][line[0]] = line[:1] + line[2:]
        char_dict['name2gender'][line[0]] = line[1]
        for alias in line[:1] + line[2:]:
            char_dict['alias2name'][alias]=line[0]
    return char_dict


def GetBookData(book_path, anno_path, char_dict):
    with open(book_path, 'r') as f:
        text = f.read()
    annos = []
    with open(anno_path, 'r') as f:
        annos = f.read().split('\n')
        annos = [dict(zip(
            ['chapter_index', 'speaker', 'utterance'],
            anno.split('\t')))
            for anno in annos if len(anno.split('\t')) == 3]
    for anno_idx, anno in enumerate(annos):
        utters = [' '.join(u.strip().split()) for u in anno['utterance'].split('[X]')]
        anno['utterance'] = []
        anno['offset'] = []
        anno['id'] = []
        if anno['speaker'] not in char_dict['alias2name']:
            continue
        anno['speaker_gender'] = char_dict['name2gender'][char_dict['alias2name'][anno['speaker']]]
        cur_pos = 0
        for utter in utters:
            start_pos = text.find(utter, cur_pos)
            if start_pos != -1:
                anno['utterance'].append(utter)
                anno['offset'].append([start_pos, start_pos + len(utter)])
                anno['id'].append(f'{anno_idx}_{len(anno["id"])}')
                cur_pos = start_pos + len(utter)

    annos = list(filter(lambda x: len(x['offset']) > 0, annos))
    chapters = text.split('CHAPTER')
    print('chapter number', len(chapters))
    chapter_offset_dict = {
        chapter_index: [len('CHAPTER'.join(chapters[:chapter_index])),
                        len('CHAPTER'.join(chapters[:chapter_index + 1]))]
        for chapter_index, _ in enumerate(chapters)}
    paragraphs = GetParagraphs(text)
    aligned_paragraphs = AlignParagraphWithSpeaker(paragraphs, annos)
    data = {}
    for chapter_index, chapter_offset in chapter_offset_dict.items():
        chapter_paragraphs = []
        for paragraph in aligned_paragraphs:
            if (paragraph['offset'][0] >= chapter_offset[0]
                    and paragraph['offset'][1] <= chapter_offset[1]):
                paragraph['chapter_index'] = chapter_index
                paragraph['book_name'] = os.path.basename(book_path)
                chapter_paragraphs.append(paragraph)
        if len(('\n\n'.join([p['paragraph'] for p in chapter_paragraphs]).split())) < 100:
            continue
        else:
            data[chapter_index] = chapter_paragraphs
    return data


def GetParagraphs(text):
    paragraphs = text.split('\n\n')
    # paragraphs = [PreprocessText(p) for p in paragraphs]
    paragraph_dict = []
    for i in range(len(paragraphs)):
        start_offset = len('\n\n'.join(paragraphs[:i]))
        end_offset = len('\n\n'.join(paragraphs[:i + 1]))
        paragraph_dict.append({
            'paragraph_index': f'pap_{i}',
            'paragraph': paragraphs[i],
            'offset': [start_offset, end_offset],
            'utterance':[],
            'book_name':'None',
            'mode':'Narrative'
            #'utterance_span':[],
            #'utterance_id':[],
            #'speaker':[],
            #'speaker_gender':[],
            #'speaker_cue':[]
        })
    return paragraph_dict


def AlignParagraphWithSpeaker(paragraphs, annos):
    alignments = []
    for index, paragraph in enumerate(paragraphs):
        if index % 100 == 0:
            print(f'matching [{index}]/[{len(paragraphs)}]')
        para_span = paragraph['offset']
        for anno in annos:
            # print('anno:',anno['qSpan'])
            anno_span = anno['offset']
            utter_span = [int(anno_span[0][0]), int(anno_span[-1][1])
                           if len(anno_span) > 1 else int(anno_span[0][1])]
            if utter_span[0] >= para_span[0] and utter_span[1] <= para_span[1]:
                quote_dicts=[]
                for k in range(len(anno['utterance'])):
                    quote = anno['utterance'][k]
                    para_text = paragraph['paragraph']
                    para_sents = cut_sentence_with_quotation_marks(para_text,False)
                    para_quotes = list(filter(lambda x:x['mode']=='Utterance',para_sents))
                    matches = list(filter(lambda x:re.search(r"(?<=\`\`).*?(?=\'\')",x['sentence']).group(0).strip()==quote.strip(),para_quotes))
                    if len(matches)==0:
                        continue
                    quote = matches[0]['sentence']
                    quote_offset = para_text.find(quote)
                    quote_span = [quote_offset,quote_offset+len(quote)]
                    quote_dict = {
                        'quote':quote,
                        'quote_span':quote_span,
                        'quote_id': anno['id'][k],
                        'quote_type':'None',
                        'speaker':anno['speaker'],
                        'speaker_gender':"female" if anno['speaker_gender']=="F" else "male",
                        'speaker_cue':'None'
                    }
                    quote_dicts.append(quote_dict)
                paragraph['utterance'] = quote_dicts
                paragraph['mode'] = 'Dialogue'
                #paragraph['utterance_span'].extend(anno_span)
                #paragraph['utterance'].extend(anno['utterance'])
                #paragraph['speaker'].extend([anno['speaker']]*len(anno['utterance']))
                #paragraph['utterance_type'] = []
                #paragraph['utterance_id'].extend( [f"{anno['id']}_{qid}" for qid in range(len(anno['utterance']))])
                #paragraph['speaker_gender'].extend([anno['speaker_gender']]*len(anno['utterance']))
        alignments.append(paragraph)
    return alignments





def pap2json(book_dir,proc_dir):
    os.makedirs(proc_dir, exist_ok=True)
    text_file_path = os.path.join(book_dir,'pride_and_prejudice.txt')
    proc_text_file_path = os.path.join(proc_dir,'pride_and_prejudice.txt')
    with open(text_file_path, 'r') as f:
        text = f.read()
    paragraphs = text.split('\n\n')
    paragraphs = [PreprocessText(p) for p in paragraphs if p.strip()]
    text = '\n\n'.join(paragraphs)


    with open(proc_text_file_path, 'w') as f:
        f.write(text)

    anno_file_path = os.path.join(book_dir,'pride_and_prejudice_anno.txt')
    char_file_path = os.path.join(book_dir,'PeopleList_Revised.txt')
    char_dict = ReadCharFile(char_file_path)
    chapters = GetBookData(proc_text_file_path, anno_file_path, char_dict)
    test_chapters = [(c_id, chapters[c_id]) for c_id in range(19, 27)]
    dev_chapters = [(c_id, chapters[c_id]) for c_id in range(27, 34)]
    train_chapters = [(c_id, chapters[c_id]) for c_id in list(chapters.keys()) if c_id not in list(range(19, 34))]

    new_char_dict = {'id2names':{},'name2id':{},'id2gender':{}}
    for idx,(k,v) in enumerate(char_dict['name2alias'].items()):
        new_char_dict['id2names'][str(idx)] = v
        new_char_dict['id2gender'][str(idx)] = 'Female' if char_dict['name2gender'][k] == 'F' else 'Male'
        for name in v:
            new_char_dict['name2id'][name] = idx

    test_chapter_path = os.path.join(proc_dir, 'test_chapter.json')
    dev_chapter_path = os.path.join(proc_dir, 'dev_chapter.json')
    train_chapter_path = os.path.join(proc_dir, 'train_chapter.json')
    char_dict_path = os.path.join(proc_dir,'character.json')

    with open(test_chapter_path, 'w') as f:
        json.dump(test_chapters, f, indent=2)

    with open(dev_chapter_path, 'w') as f:
        json.dump(dev_chapters, f, indent=2)

    with open(train_chapter_path, 'w') as f:
        json.dump(train_chapters, f, indent=2)

    with open(char_dict_path,'w') as f:
        json.dump(new_char_dict,f,indent=2)


def FormatPAP(sour_path, target_path, char_file_path):
    with open(sour_path, 'r') as f:
        sour_data = json.load(f)

    with open(char_file_path,'r') as f:
        char_dict = json.load(f)


    dest_data = []
    dest_count = 0
    for chapter_id, chapter in tqdm(sour_data,desc="FormatPAP progress"):
        dialogues = ExtractDialogue(chapter,context_len=10)
        dialogues = copy.deepcopy(dialogues)
        for dialogue in dialogues:
            dial_paras = dialogue['preceding_paragraphs']+dialogue['dialogue']+dialogue['succeeding_paragraphs']
            #text = ' '.join(list(map(lambda x:x['paragraph'],dial_paras)))
            for para_index in range(len(dial_paras)):
                offset = len(' '.join(list(map(lambda x: x['paragraph'], dial_paras[:para_index]))))
                dial_paras[para_index]['offset'] = [offset, offset + len(dial_paras[para_index]['paragraph'])]
            for para in dialogue['dialogue']:
                #utter_offsets = []
                for quote_dict in para['utterance']:
                    # compute the offset of each quote within the paragraph
                    offset = para['paragraph'].find(quote_dict['quote'])
                    quote_dict['quote_span'] = [offset,offset+len(quote_dict['quote'])] if offset!=-1 else [-1,-1]
                    # determine the quote type
                    #para_sents = cut_sentence_with_quotation_marks(para['paragraph'], is_chinese=False)
                    #narr_sents = list(map(lambda x: x['sentence'],
                    #                      filter(lambda y: y['mode'] == 'Narrative', para_sents)))
                    #aliases = char_dict['id2names'][str(char_dict['name2id'][quote_dict['speaker']])]
                    #qtype = GetQuoteType(para['paragraph'],quote_dict['quote'],aliases,is_chinese=False)
                    #quote_dict['quote_type'] = qtype
                #para['utterance_span'] = utter_offsets
            sub_char_dict = GetSubCharDict(dialogue,char_dict)
            sub_char_dict = AddUnkToken(sub_char_dict)
            dialogue['character'] = sub_char_dict


            dialogue['id'] = dest_count
            dest_count+=1
        dest_data += dialogues

    dest_data = GetQuoteTypeForCorpus(dest_data,is_chinese=False)
    tqdm.write(f"# instance for FormatPAP:{len(dest_data)}")
    with open(target_path, 'w') as f:
        json.dump(dest_data, f, indent=2)


def ProcessPAP(sour_dir,target_dir):
    #sour_dir = 'data/raw_data/PAP'
    #target_dir = 'data/proc_data/PAP'
    t = tqdm(['train', 'dev', 'test'])
    os.makedirs(target_dir,exist_ok=True)
    for set_name in t:
        t.set_description(desc=f"Process PAP: {set_name}")
        sour_path = os.path.join(sour_dir, f'{set_name}_chapter.json')
        target_path = os.path.join(target_dir, f'{set_name}.json')
        char_path = os.path.join(sour_dir,'character.json')
        FormatPAP(sour_path, target_path,char_path)


book_dir = './raw_data_new/PAP'
proc_dir = './proc_data_new2/PAP'
pap2json(book_dir,proc_dir)
ProcessPAP(proc_dir,proc_dir)