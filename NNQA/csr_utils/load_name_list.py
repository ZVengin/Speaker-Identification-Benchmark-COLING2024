# load name list from name_list.txt

name_list_path = './data/name_list.txt'

# load the name list from file
def get_alias2id(name_list_path):
    with open(name_list_path, 'r', encoding='utf-8') as fin:
        name_lines = fin.readlines()
    alias2id = {}
    for i, line in enumerate(name_lines):
        for alias in (line.strip().split()[1:] if '\t' not in line else 
                      line.strip().split('\t')[1:]):
            alias2id[alias] = i
    print(f'alias2id:{alias2id}')
    return alias2id

def get_id2alias(name_list_path):
    with open(name_list_path, 'r', encoding='utf-8') as fin:
        name_lines = fin.readlines()
    id2alias = []
    for i, line in enumerate(name_lines):
        id2alias.append(line.strip().split()[1] if '\t' not in line else
                      line.strip().split('\t')[1])
    return id2alias

import json
def get_chardict(char_dict_path):
    with open(char_dict_path,'r') as f:
        char_dict = json.load(f)
    return char_dict