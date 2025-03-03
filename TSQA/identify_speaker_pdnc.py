import subprocess
import json,os
from tsqa_utils import IdentifySpeaker,ConstructRecords,compute_accuracy,ConstructSingleQuoteInstance
from multiprocessing.pool import Pool

para_path = './../dataset/proc_data/PDNC_merge/fold0/test.json'
java_path="./stanford-corenlp-4.5.4"
out_path="./../analysis/result_records/PDNC_merge"


def StartServer(port):
    subprocess.run(
        ["java", "-Xmx8g","-cp", f"{java_path}/*", "edu.stanford.nlp.pipeline.StanfordCoreNLPServer","-outputFormat","json",
         "-serverProperties", f"{java_path}/StanfordCoreNLP-Quote-English.properties", "-port", f"{port}", "-timeout", "150000"])




if __name__=='__main__':
    proc_num = 4
    base_port = 9300
    with open(para_path,'r') as f:
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
    with open(os.path.join(out_path,'TSQA_PDNC_merge_result_records.json'),'w') as f:
        json.dump(records,f,indent=2)

