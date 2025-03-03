import copy
import json,os,sys
from tqdm import tqdm
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(SCRIPT_DIR,'..'))
from unify_dataset import GetSubCharDict,AddUnkToken,ClusterQuote,cut_sentence_with_quotation_marks
def GenerateCharacterDict(characters):
    characters = sorted(characters,key=lambda x:len(x),reverse=True)
    char_dict = {'id2names':{},'name2id':{}}
    clusters = []
    for character in characters:
        exist = False
        for cluster in clusters:
            if any([cluster_char.endswith(character) for cluster_char in cluster]):
                cluster.append(character)
                exist=True
        if not exist:
            clusters.append([character])
    for cluster_id,cluster in enumerate(clusters):
        char_dict['id2names'][cluster_id]=cluster
        for char in cluster:
            char_dict['name2id'][char] = cluster_id
    return char_dict

def ReadCharDict(char_path):
    char_dict = {'id2names':{},'name2id':{},'id2gender':{}}
    with open(char_path,'r') as f:
        lines = f.read().split('\n')
    for char_id,line in enumerate(lines):
        line = [token.strip() for token in line.split()]
        gender,aliases = line[0],line[1:]
        char_dict['id2names'][str(char_id)] = aliases
        for alias in aliases:
            char_dict['name2id'][alias] = char_id
        char_dict['id2gender'][str(char_id)] = "Female" if gender=="0" else "Male"
    return char_dict

def Text2Json(text_path,char_dict,text_name):
    new_insts = []

    with open(text_path,'r') as f:
        lines = f.read()
        insts = list(set([inst.strip() for inst in lines.split('\n\n') if inst.strip()]))
    for idx,inst in enumerate(insts):
        lines = inst.split('\n')
        assert len(lines)==25,'the invalid lines'
        speaker = lines[-3].split(':')[1].strip()
        quote_type = lines[-1].split(':')[1].strip()
        if quote_type == 'explicit':
            quote_type = 'Explicit'
        elif quote_type == 'implicit':
            quote_type = 'Anaphoric(pronoun)'
        else:
            quote_type = 'Implicit'

        new_inst = {
            'preceding_paragraphs':[],
            'dialogue':[],
            'succeeding_paragraphs':[],
            'id':idx
        }
        lines = lines[1:22]
        for lid,line in enumerate(lines):
            line = line.strip()
            line_offset = len('\n'.join(lines[:lid]))
            if lid == 10:
                #utter = [line]
                #utter_span = [[line_offset,line_offset+len(line)]]
                #utter_speaker = [speaker]
                #utter_type = [quote_type]
                #utter_id = [f'{text_name}_{idx}_{lid}']
                #utter_gender = [char_dict['id2gender'][str(char_dict['name2id'][speaker])]]
                gender = char_dict['id2gender'][str(char_dict['name2id'][speaker])]
                quote_dicts = [{
                    'quote':line,
                    'quote_span':[line_offset,line_offset+len(line)],
                    'quote_id':f'{text_name}_{idx}_{lid}',
                    'quote_type':quote_type,
                    'speaker':speaker,
                    'speaker_gender':gender,
                    'speaker_cue':'None'
                }]

            else:
                quote_dicts=[]
                sents = cut_sentence_with_quotation_marks(line,is_chinese=True)
                for sent_idx in range(len(sents)):
                    sent_offset = line.find(sents[sent_idx]['sentence'])
                    if sents[sent_idx]['mode'] == 'Utterance':
                        quote_dicts.append({
                            'quote':sents[sent_idx]['sentence'],
                            'quote_span':[sent_offset,sent_offset+len(sents[sent_idx]['sentence'])],
                            'quote_id':f'{text_name}_{idx}_{lid}_{sent_idx}',
                            'quote_type':'None',
                            'speaker':'None',
                            'speaker_gender':'None',
                            'speaker_cue':'None'
                        })

            paragraph = {
                'paragraph':line,
                'paragraph_index':f'{text_name}_{idx}_{lid}',
                'offset':[line_offset,line_offset+len(line)],
                'book_name':text_name,
                'utterance':quote_dicts,#utter,
                'mode':'Dialogue' if len(quote_dicts)>0 else 'Narrative',
                #'utterance_span':utter_span,
                #'speaker':utter_speaker,
                #'utterance_type':utter_type,
                #'utterance_id':utter_id,
                #'speaker_gender':utter_gender,
                #'speaker_cue':[]
            }
            if lid<10:
                new_inst['preceding_paragraphs'].append(paragraph)
            elif lid==10:
                new_inst['dialogue'].append(paragraph)
            else:
                new_inst['succeeding_paragraphs'].append(paragraph)

        sub_char_dict = GetSubCharDict(new_inst,char_dict)
        sub_char_dict = AddUnkToken(sub_char_dict)
        new_inst['character'] = sub_char_dict
        #origin_para_idxs=list(map(lambda x:x['paragraph_index'],new_inst['preceding_paragraphs']+new_inst['dialogue']+new_inst['succeeding_paragraphs']))
        new_inst['dialogue'][0]['original_instance']=copy.deepcopy(new_inst)
        new_insts.append(new_inst)

    return new_insts




def wp2json(book_dir,proc_dir):
    os.makedirs(proc_dir,exist_ok=True)
    char_path = f'{raw_dir}/name_list.txt'
    char_dict = ReadCharDict(char_path)

    for set_name in ['train','dev','test']:
        set_dir = f'{book_dir}/{set_name}'
        file_path = os.path.join(set_dir, f'{set_name}_unsplit.txt')
        insts = Text2Json(file_path,char_dict,f'wp_{set_name}')
        target_path = os.path.join(proc_dir,f'{set_name}_paragraphs.json')
        with open(target_path,'w') as f:
            json.dump(insts,f,indent=2)
    char_dict_path = os.path.join(proc_dir,'character.json')
    with open(char_dict_path,'w') as f:
        json.dump(char_dict,f,indent=2)


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


import os
raw_dir='raw_data_new/WP2021'
proc_dir='proc_data_new/WP2021'
wp2json(raw_dir,proc_dir)
ProcessWP2021(proc_dir,proc_dir)

