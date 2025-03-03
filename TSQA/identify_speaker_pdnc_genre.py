import subprocess
import json,os,argparse
from tsqa_utils import IdentifySpeaker,ConstructRecords,compute_accuracy,ConstructSingleQuoteInstance
from multiprocessing.pool import Pool

para_path = './../dataset/proc_data/PDNC_genre/{}/test.json'
java_path="./stanford-corenlp-4.5.4"
out_path="./../analysis/result_records/PDNC_genre"


def StartServer(port):
    subprocess.run(
        ["java", "-Xmx8g","-cp", f"{java_path}/*", "edu.stanford.nlp.pipeline.StanfordCoreNLPServer","-outputFormat","json",
         "-serverProperties", f"{java_path}/StanfordCoreNLP-Quote-English.properties", "-port", f"{port}", "-timeout", "150000"])




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('genre',choices=['Classic','Children_Adventure','Detective_Mystery','Period_Romance','Science_Fiction_Fantacy'])
    args = parser.parse_args()
    proc_num = 4
    base_port = 9300
    with open(para_path.format(args.genre),'r') as f:
        insts = json.load(f)

    insts = ConstructSingleQuoteInstance(insts)
    group_size = len(insts)//proc_num+1
    groups = [insts[i:i+group_size] for i in range(0,len(insts),group_size)]
    pool_server = Pool(proc_num)
    pool_client = Pool(proc_num)
    results = []
    for i in range(proc_num):
        pool_server.apply_async(StartServer,args=[base_port+i])
        r=pool_client.apply_async(IdentifySpeaker,args=[groups[i],base_port+i])
        results.append(r)
    pool_server.close()
    pool_client.close()
    pool_client.join()
    insts = sum([r.get() for r in results],[])

    records = ConstructRecords(insts)

    score = compute_accuracy(records)
    print(f'score:{score}')

    os.makedirs(out_path,exist_ok=True)
    with open(os.path.join(out_path,f'TSQA_PDNC_{args.genre}_result_records.json'),'w') as f:
        json.dump(records,f,indent=2)

