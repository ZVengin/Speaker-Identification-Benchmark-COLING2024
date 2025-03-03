import argparse
import copy
import json
import logging
import os
import pickle
import random
import re
from tqdm import tqdm
from collections import Counter
import shutil
from urllib import parse

import pandas as pd
from transformers import AutoTokenizer
from unify_dataset import ExtractDialogue,GetSubCharDict,AddUnkToken,cut_sentence_with_quotation_marks

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
MAX_CONTEXT_LEN = 500

random.seed(123)
pronoun_list = ['I','we','you','they','she','he','it']


def iter_flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i : i + 1] = l[i]
        i += 1
    return ltype(l)


def add_men_str(row):
    ments = row["mention_ents"]
    if not isinstance(ments, list):
        ments = eval(row["mention_ents"])
    ments = iter_flatten(ments)
    mstr = "_".join(ments)
    row["mention_texts"] = mstr
    return row


def GetBookData(book_path, anno_path, char_path,book_name):
    with open(book_path, "r") as f:
        text = f.read()
    with open(char_path,'rb') as f:
        char_dict = pickle.load(f)
    annos = pd.read_csv(anno_path, index_col=0, keep_default_na=False, dtype=str)
    annos = annos.apply(add_men_str, axis=1)
    speaker_counter = Counter(annos["speaker"])
    remove_speakers = [
        x
        for x in speaker_counter
        if len(x) == 0 or x[0] == "_" #or speaker_counter[x] < 10
    ]
    out_of_scope_speakers = list(set(list(speaker_counter)).difference(set(char_dict['name2id'].keys())))
    remove_speakers = list(set(remove_speakers).union(set(out_of_scope_speakers)))
    print(f'{len(out_of_scope_speakers)} speakers are out-of-scope:\n{out_of_scope_speakers}')
    annos = annos[~annos["speaker"].isin(remove_speakers)]
    annos = annos[annos["text"] != ""]
    annos = annos.to_dict("records")

    print(f"# of annotations:{len(annos)}")
    for anno in annos:
        quotes_span = [eval(anno.pop("span"))]
        quotes_arr = [text[qs[0] : qs[1]] for qs in quotes_span]
        anno["qSpan"] = quotes_span
        anno["qTextArr"] = quotes_arr
        anno["qId"] = f'{book_name}_{anno.pop("QuoteID")}'
        anno["qType"] = anno.pop("qtype")
        anno['addressee'] = eval(anno['addressee'])
        anno['speaker_cue'] = anno.pop("refExp")

    paragraphs = GetParagraphs(text,book_name)
    aligned_paragraphs = AlignParagraphWithSpeaker(paragraphs, annos)
    #print(
    #    f'unaligned annotations: {set(map(lambda x:x["qId"],annos))-set(sum(map(lambda x:[d["quote_id"] for d in x["utterance"]],aligned_paragraphs),[]))}'
    #)
    return aligned_paragraphs


def GetParagraphs(text,text_name):
    paragraphs = text.split("\n\n")
    paragraph_dict = []
    for i in range(len(paragraphs)):
        start_offset = len("\n\n".join(paragraphs[:i]))
        end_offset = len("\n\n".join(paragraphs[: i + 1]))
        paragraph_dict.append(
            {
                "paragraph_index": f'{text_name}_{i}',
                "paragraph": paragraphs[i],
                "offset": [start_offset, end_offset],
                "book_name":text_name
            }
        )
    print(f"paragraph:{len(paragraph_dict)}")
    return paragraph_dict


def AlignParagraphWithSpeaker(paragraphs, annos):
    alignments = []
    aligned_num = 0
    for index, paragraph in enumerate(paragraphs):
        if index % 100 == 0:
            print(f"matching [{index}]/[{len(paragraphs)}]")
        paragraph["utterance"] = []
        para_span = paragraph["offset"]
        for anno in annos:
            # print('anno:',anno['qSpan'])
            anno_span = anno["qSpan"]
            utter_span = [
                int(anno_span[0][0]),
                int(anno_span[-1][1]) if len(anno_span) > 1 else int(anno_span[0][1]),
            ]

            if utter_span[0] >= para_span[0] and utter_span[1] <= para_span[1]:
                if anno['qType'] == 'Anaphoric':
                    is_pronoun = False
                    for pr in pronoun_list:
                        if re.search((r'\b{}\b'.format(pr)).lower(),anno['speaker_cue'].lower()) is not None:
                            is_pronoun = True
                            break
                    if is_pronoun:
                        anno['qType']='Anaphoric(pronoun)'
                    else:
                        anno['qType']='Anaphoric(other)'
                #quote_dicts = []
                para_sents = cut_sentence_with_quotation_marks(paragraph['paragraph'],is_chinese=False)
                para_sents = list(map(lambda x:x['sentence'],para_sents))
                for qidx in range(len(anno["qTextArr"])):
                    quote = anno["qTextArr"][qidx]
                    matches = list(filter(lambda x: x.find(quote.strip())!=-1,para_sents))
                    if len(matches)>0:
                        quote = matches[0]
                        quote_offset = paragraph['paragraph'].find(quote)
                        quote_span = [quote_offset,quote_offset+len(quote)]
                    else:
                        continue

                    quote_dict = {
                        'quote':quote,
                        'quote_span':quote_span,
                        'quote_id':anno["qId"],
                        'quote_type':anno["qType"],
                        'speaker':anno['speaker'],
                        'speaker_gender':'None',
                        'speaker_cue':anno['speaker_cue'],
                        'addressee':anno['addressee']
                    }
                    inside = False
                    for para_quote_dict in paragraph['utterance']:
                        if para_quote_dict['quote'].strip() == quote_dict['quote'].strip():
                            inside = True
                            break
                    if not inside:
                        paragraph['utterance'].append(quote_dict)
                    #quote_dicts.append(quote_dict)
                #paragraph['utterance']+=quote_dicts
                aligned_num += 1
        paragraph['utterance'] = sorted(paragraph['utterance'],key=lambda x:x['quote_span'][0])
        paragraph['mode'] = 'Dialogue' if len(paragraph['utterance']) > 0 else 'Narrative'
        alignments.append(paragraph)
    print(
        f"# of aligned annotation:{aligned_num}, # of unaligned annotation: {len(annos)-aligned_num}"
    )
    return alignments


def pdnc2json(book_dir,proc_dir):
    multi_speaker_count = 0
    total_count = 0
    for book_name in os.listdir(book_dir):
        if book_name.startswith('.') or os.path.isfile(os.path.join(book_dir,book_name)):
            continue
        print(f"generate instances for book :{book_name}")
        book_path = os.path.join(book_dir, book_name, "text.txt")
        anno_path = os.path.join(book_dir, book_name, "quote_data.csv")
        book_anno_path = os.path.join(proc_dir, book_name, "book_anno.json")
        char_path = os.path.join(book_dir,book_name,'charDict.pkl')
        book_data = GetBookData(book_path, anno_path, char_path, book_name)
        shutil.copy(char_path,os.path.join(proc_dir,book_name,'charDict.pkl'))
        with open(char_path,'rb') as f:
            char_dict = pickle.load(f)
        if not os.path.exists(os.path.join(proc_dir, book_name)):
            os.makedirs(os.path.join(proc_dir, book_name))
        with open(book_anno_path, "w") as f:
            json.dump(book_data, f, indent=2)

        for para in book_data:
            para_speaker_ids = []
            for quote_dict in para['utterance']:
                speaker = quote_dict['speaker']
                if speaker in char_dict['name2id']:
                    para_speaker_ids.append(char_dict['name2id'][speaker])
            para_speaker_ids = list(set(para_speaker_ids))
            if len(para_speaker_ids)>1:
                multi_speaker_count+=1
            if len(para_speaker_ids)>0:
                total_count+=1
    print(f'multi_para_count:{multi_speaker_count},multi_para_portion:{multi_speaker_count/total_count}')


def FormatPDNC(sour_path, char_file_path):
    with open(sour_path, 'r') as f:
        sour_data = json.load(f)

    with open(char_file_path, 'rb') as f:
        char_dict = pickle.load(f)
        new_char_dict = {'id2names':{},'name2id':{},'id2gender':{}}
        for k,v in char_dict['id2names'].items():
            new_char_dict['id2names'][str(k)] = list(v)
            new_char_dict['id2gender'][str(k)] = 'None'
            for n in v:
                new_char_dict['name2id'][n] = k

        #for paragraph in sour_data:
        #    for quote_dict in paragraph['utterance']:
        #        speaker = quote_dict['speaker']
        #        speaker_gender = quote_dict['speaker_gender']
        #        speaker_id = new_char_dict['name2id'][speaker]
        #        new_char_dict['id2gender'][str(speaker_id)]=speaker_gender
        char_dict = new_char_dict

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
            for utter in para['utterance']:
                offset = para['paragraph'].find(utter['quote'])
                utter['quote_span'] = [offset, offset + len(utter['quote']) if offset != -1 else -1]
                #utter_offsets.append([offset, offset + len(utter) if offset != -1 else -1])
            #para['utterance_span'] = utter_offsets
        sub_char_dict = GetSubCharDict(dialogue,char_dict)
        sub_char_dict = AddUnkToken(sub_char_dict)
        dialogue['character'] = sub_char_dict
        dialogue['id'] = str(dest_count)
        dest_count += 1
    dest_data = dialogues

    return dest_data


def ProcessPDNC(sour_dir,target_dir):
    dest_data = []

    os.makedirs(target_dir,exist_ok=True)
    t = tqdm(os.listdir(sour_dir))
    inst_index=0
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
                inst['id'] = inst_index
                inst_index += 1
            dest_data += book_data

    random.seed(123)
    dest_data = random.sample(dest_data,k=len(dest_data))
    test_size = int(len(dest_data)*0.1)
    set_dict = {}
    set_dict['test'] = sorted(dest_data[:test_size],key=lambda x:x['id'])
    set_dict['dev'] = sorted(dest_data[test_size:2*test_size],key=lambda x:x['id'])
    set_dict['train'] = sorted(dest_data[2*test_size:],key=lambda x:x['id'])
    os.makedirs(target_dir, exist_ok=True)
    for set_name in ['train','dev','test']:
        set_path = os.path.join(target_dir,set_name+'.json')
        with open(set_path,'w') as f:
            json.dump(set_dict[set_name],f,indent=2)

book_dir = "./raw_data_new/PDNC"
proc_dir = "./proc_data_new2/PDNC_merge"
proc_book_dir=os.path.join(proc_dir,'book_data')
pdnc2json(book_dir,proc_book_dir)
ProcessPDNC(proc_book_dir,proc_dir)
