import time,requests,json,os,sys
from tqdm import tqdm
SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIP_DIR,'./..'))
from utils import AddUnkToken,compute_accuracy,ConstructSingleQuoteInstance

settings = 'http://localhost:{}'

def IdentifySpeaker(instances,port):
    time.sleep(60)

    total_target_count = 0
    total_matched_count =0

    inst_count = 0
    omit_count = 0
    for inst in tqdm(instances):
        # 用空格连接是因为对于中文的数据集，每个段落是一个句子，所以原本同属一个段落的quote和其后面的句子被当成不同的段落
        # 例如 “你怎么这样？”，霍敏小声的说。
        context = '\n'.join([p['paragraph'] for p in inst['preceding_paragraphs']+inst['dialogue']+inst['succeeding_paragraphs']])
        response = requests.post(settings.format(port), data = {'data':context}).text
        try:
            quotes=json.loads(response)['quotes']
        except Exception as e:
            omit_count += 1
            print(f'context:{context}')
            print(f'response:{response}')
            print(f'omit count:{omit_count}')
            continue

        for para in inst['dialogue']:
            para['predict_speaker']=[]
            para['predict_quote'] = []
            total_target_count += len(para['utterance'])
            for para_quote in para['utterance']:
                for quote in quotes:
                    if para_quote['quote'].find(quote['text']) != -1:
                        para_quote['predict_speaker']=quote['speaker']
                        para_quote['predict_quote']=quote['text']
                        total_matched_count+=1
        inst_count+=1

    print(f'#toatl target quotes:{total_target_count},#total matched quotes:{total_matched_count}')
    return instances


def ConstructRecords(results):
    records = []
    for inst in results:
        context = '\n'.join(map(lambda x: x['paragraph'],
                                inst['preceding_paragraphs']
                                + inst['dialogue']
                                + inst['succeeding_paragraphs']))
        char_dict = inst['character']
        char_dict = AddUnkToken(char_dict)
        for para in inst['dialogue']:
            if 'predict_speaker' not in para:
                    para['predict_speaker'] = ['None']*len(para['utterance'])
            else:
                para['predict_speaker'] = para['predict_speaker']+['None']*(len(para['utterance'])-len(para['predict_speaker']))
            for quote_index in range(len(para['utterance'])):
                qid = para['utterance'][quote_index]['quote_id']
                quote = para['utterance'][quote_index]['quote']
                label = para['utterance'][0]['speaker']
                predict_speaker = para['predict_speaker'][quote_index]
                record = {
                    'context': context,
                    'quote_id': qid,
                    'quote': quote,
                    'instance_id': inst['id'],
                    'predict_speaker': predict_speaker,
                    'label': label,
                    'character':char_dict
                }
                records.append(record)
    return records