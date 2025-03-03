import copy
import json,os,sys
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(SCRIPT_DIR,'..'))
from unify_dataset import GetSubCharDict,AddUnkToken,GetClusters
from collections import defaultdict
from tqdm import tqdm
def FormatJY(sour_path, targ_path, char_dict):
    with open(sour_path, 'r') as f:
        sour_data = json.load(f)

    dest_data = []
    inst_index = 0
    for _,origin_inst in tqdm(sour_data.items(),desc="FormatJY progress"):
        utter_paragraph_index = len(origin_inst['context_pre'])
        paragraph_texts = origin_inst['context_pre'] + [origin_inst['quote']] + origin_inst['context_next']
        text = ' '.join(paragraph_texts)
        paragraph_index = 0
        paragraphs = []
        for paragraph_text in paragraph_texts:
            offset = text.find(paragraph_text)
            paragraph = {
                'paragraph_index': f'{inst_index}-{paragraph_index}',
                'paragraph': paragraph_text,
                'offset': [[offset, offset + len(paragraph_text)]],
                #'utterance_span': [],
                'utterance': [],
                #'speaker': [],
                #'utterance_type': [],
                #'utterance_id': [],
                'book_name': origin_inst['novel'],
                'mode':'Narrative'
                #'speaker_gender':[],
                #'speaker_cue':[]
            }
            paragraphs.append(paragraph)
            paragraph_index += 1
        offset = text.find(origin_inst['quote'])
        quote_dicts = [{
            'quote':origin_inst['quote'],
            'quote_span': [[0, len(origin_inst['quote'])]],
            'speaker':origin_inst['entity'],
            'quote_id':f"{inst_index}-{utter_paragraph_index}",
            'speaker_gender':'None',
            'speaker_cue':'None',
            'quote_type':'None'
        }]
        paragraphs[utter_paragraph_index] = {
            'paragraph_index': f'{inst_index}-{utter_paragraph_index}',
            'paragraph': origin_inst['quote'],
            'offset': [[offset, offset + len(origin_inst['quote'])]],
            #'utterance_span': [[0, len(origin_inst['quote'])]],
            'utterance': quote_dicts,#[origin_inst['quote']],
            #'speaker': [origin_inst['entity']],
            #'utterance_type': [],
            #'utterance_id': [f"{inst_index}-{utter_paragraph_index}"],
            'book_name': origin_inst['novel'],
            'mode':'Dialogue'
            #'speaker_gender':[],
            #'speaker_cue':[]
        }


        #origin_para_idxs = list(map(lambda x:x['paragraph_index'],paragraphs))

        dest_inst = {
            'preceding_paragraphs':paragraphs[:utter_paragraph_index],
            'succeeding_paragraphs':paragraphs[utter_paragraph_index+1:],
            'dialogue':paragraphs[utter_paragraph_index:utter_paragraph_index+1],
            'character':None,
            'id':inst_index
        }
        sub_char_dict = GetSubCharDict(dest_inst,char_dict,is_chinese=True)
        dest_inst['character'] = sub_char_dict
        dest_inst['dialogue'][0]['original_instance'] = copy.deepcopy(dest_inst)

        dest_data.append(dest_inst)
        inst_index += 1

    tqdm.write(f"# instance of FormatJY:{len(dest_data)}")
    with open(targ_path, 'w') as f:
        json.dump(dest_data, f, indent=2)

def MergeSetsWithCommonElements(sets):
    merged_sets = []

    while sets:
        current_set = sets.pop(0)
        merged = False

        for i in range(len(merged_sets)):
            if current_set.intersection(merged_sets[i]):
                merged_sets[i] = current_set.union(merged_sets[i])
                merged = True
                break

        if not merged:
            merged_sets.append(current_set)

    return merged_sets


def ClusterNames(names):
    clusters = defaultdict(list)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[i].find(names[j]) != -1 or names[j].find(names[i]) != -1:
                clusters[names[i]].append(names[j])
                clusters[names[j]].append(names[i])
    clusters = MergeSetsWithCommonElements(list(map(lambda x: set([x[0]] + x[1]), clusters.items())))
    merge_set = set()
    for cluster in clusters:
        merge_set = merge_set.union(cluster)
    unresolved_set = set(names).difference(merge_set)
    clusters = clusters + [set([n]) for n in unresolved_set]
    return clusters

def jy2json(sour_dir,target_dir):
    os.makedirs(target_dir,exist_ok=True)
    t = tqdm(['train', 'dev', 'test'])
    characters = []
    for set_name in t:
        sour_path = os.path.join(sour_dir, f'{set_name}.json')
        with open(sour_path,'r') as f:
            sour_data = json.load(f)
        characters += sum(list(map(lambda k:sour_data[k]['candidate'],sour_data.keys())),[])
    characters = set(characters)
    clusters = ClusterNames(list(characters))
    char_dict = {
        'id2names': dict([(str(k), list(v)) for k, v in enumerate(clusters)]),
        'name2id': dict(sum([[(n, k) for n in v] for k, v in enumerate(clusters)], [])),
        'id2gender':dict([(str(k),'None') for k,_ in enumerate(clusters)])
    }
    char_dict = AddUnkToken(char_dict)
    for set_name in t:
        t.set_description(desc=f"Process JY: {set_name}")
        sour_path = os.path.join(sour_dir, f'{set_name}.json')
        target_path = os.path.join(target_dir, f'{set_name}_paragraphs.json')
        FormatJY(sour_path,target_path,char_dict)

def ProcessJY(sour_dir, target_dir):
    for set_name in ['train','dev','test']:
        sour_path = os.path.join(sour_dir,f'{set_name}_paragraphs.json')
        with open(sour_path,'r') as f:
            data = json.load(f)
        data = GetClusters(data)
        target_path = os.path.join(target_dir,f'{set_name}.json')
        with open(target_path,'w') as f:
            json.dump(data,f,indent=2)


sour_dir = 'raw_data_new/JY'
target_dir = 'proc_data_new2/JY'
jy2json(sour_dir,target_dir)
ProcessJY(target_dir,target_dir)