#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import json,math,sys
from pre_ml import loadData

def loadD(filename,fileDir,topk,refile):
    basedic = dict()
    num = 0
    avg_NDCG = 0.0
    precison = 0.0
    recall = 0.0
    most_pre = 0.0
    quid_ls = list() # 原始数据的所有quid
    with open(refile, "r",encoding="utf-8") as cf:
        line = cf.readline()
        while line:
            sample_0 = json.loads(line)
            quid = sample_0["question_id"]
            quid_ls.append(quid)
            line = cf.readline()
    print(len(quid_ls))
    sample_fake_dic = loadData(fileDir)  # 标注数据
    with open(filename,"r",encoding="utf-8") as ff:
        line = ff.readline()
        while line:
            sample = json.loads(line)
            quid = sample["question_id"]
            if quid not in quid_ls: # quid 不在原有的quidls中
                line = ff.readline()
                continue
            fake = sample_fake_dic[quid]
            did_pid_score = sample["didx_pidx_answer_question_score_fake_span"]  #根据答案选择的para排序
            topk = min(topk,len(did_pid_score))
            DCG_K = 0
            IDCG_K = 0
            pre_k = 0.0
            most_p = 0
            level3_len,level2_len,level1_len = 0,0,0
            for did,doc in enumerate(sample["documents"]):
                most_para = doc["most_related_para"]
                if most_para in fake[did][0]:
                    most_p += 1
                elif most_para in fake[did][1]:
                    most_p += 1
                else:
                    pass
            most_pre += most_p/len(sample["documents"])
            for d_id,level_para in fake.items():
                level3_len += len(level_para[0])
                level2_len += len(level_para[1])
                level1_len += len(level_para[2])
            for index in range(topk):
                f_did,f_pid = did_pid_score[index][0],did_pid_score[index][1]
                reli = -1
                if f_pid in fake[f_did][0]: # level3
                    reli = 2
                    pre_k += 1
                elif f_pid in fake[f_did][1]: # level 2
                    reli = 1
                    pre_k += 1
                elif f_pid in fake[f_did][2]:  # level 1
                    reli = 0
                else:
                    raise ValueError("paraid_error")
                DCG_K += (2**reli-1)/math.log2(index+1+1)
                if index <=level3_len:
                    IDCG_K += (2**2-1)/math.log2(index+1+1)
                elif index <=level3_len+level2_len:
                    IDCG_K += (2**1-1)/math.log2(index+1+1)
                else:
                    pass
            avg_NDCG += (DCG_K/IDCG_K) if IDCG_K>0 else 0

            precison += pre_k/topk if topk>0 else 0
            num += 1
            line = ff.readline()
    print(avg_NDCG/num,num,precison/num, most_pre/num)
    return avg_NDCG

def main():
    filename = sys.argv[1]
    fileDir = sys.argv[2]
    refile = sys.argv[3]
    topk = int(sys.argv[4])
    loadD(filename,fileDir,topk,refile)

if __name__ =="__main__":
    main()

"""
baseline
search NDCG_5 0.6599 precis_5 0.48  most_precis 0.4217
zhidao NDCG_5 0.5365 precis_5 0.51  most_precis 0.5325  
"""