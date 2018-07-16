#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import os,sys
import math

def NDCG_k(ls,topk):
    """
    # 两类
    :param ls:
    :param topk:
    :return:
    """
    topk = min(topk,len(ls))
    temp_ls = ls[:]
    temp_ls = sorted(temp_ls,key=lambda x:x[2],reverse=True) # 相关性等级降序排列
    iDCG = 0
    DCG = 0
    for i in range(topk):
        DCG += (2**(ls[i][2]-1)-1)/math.log2(i+2)
        iDCG +=(2**(temp_ls[i][2]-1)-1)/math.log2(i+2)
    return  DCG/iDCG if iDCG>0 else 0

def precRecall(ls,topk):
    if topk<=0:return 0,0
    num = 0
    mnum=0
    t_len = min(topk,len(ls))
    for i in range(t_len):
        if ls[i][-2]!=1:
            num += 1
    prec = num/float(topk) if topk>0 else 0
    for item in ls:
        if item[-2]!=1:
            mnum+=1
    recall = num/float(mnum) if mnum>0 else 0.0
    return prec,recall

def MRR(ls):
    ans = 0
    num = 0
    for index,item in enumerate(ls):
        if item[-2]!=1:
            ans += 1/float(index+1)
            num += 1
    return ans/num if num>0 else 0

def MAP(ls):
    num = 0
    ans = 0
    for index,item in enumerate(ls):
        if item[-2]!=1:
            num+=1
            ans += num/float(index+1)
    return ans/num if num>0 else 0

def evalute(datatype,resultDir,i):
    """

    :return:
    """
    sfile = os.path.join("sfile",datatype+".test.save_"+str(i))
    resultfile = resultDir+str(i)
    # Map,Mrr = 0.0, 0.0
    # NDCG = 0.0
    Prec,Recall = 0.0, 0.0
    ques_num = 0
    with open(sfile,"r",encoding="utf-8") as sf:
        with open(resultfile,"r",encoding="utf-8") as rf:
            sline = sf.readline()
            rline = rf.readline()
            result = dict()
            question = list()
            pre_qid = 0
            while sline and rline:
                q = list(map(lambda x:int(x),sline.strip().split("\t")))
                score = float(rline.strip())
                if q[0]==pre_qid or pre_qid==0:
                    question.append([q[1],q[2],q[3],score])
                else:
                    # new sample
                    result[pre_qid]=sorted(question,key=lambda x:x[-1],reverse=True)  # global sorted
                    if len(result[pre_qid])==0:
                        pre_qid = q[0]
                        sline = sf.readline()
                        rline = rf.readline()
                        continue
                    # Map += MAP(result[pre_qid])
                    # Mrr += MRR(result[pre_qid])
                    # NDCG += NDCG_k(result[pre_qid],5)
                    p_r = precRecall(result[pre_qid],5)
                    # print(p_r)
                    Prec += p_r[0]
                    Recall += p_r[1]
                    #print(result[pre_qid])
                    ques_num+=1
                    question=list()
                pre_qid = q[0]
                sline = sf.readline()
                rline = rf.readline()
    # with open(resultDir+"log_"+str(i),"w") as f:
    #    f.write("SVM-rank"+"\tMAP "+str(Map/ques_num)+"\tMRR "+str(Mrr/ques_num)+"\tPrecision "+str(Prec/ques_num)+"\tRecall "+str(Recall/ques_num))
    print("SVM-rank","Precision",Prec/ques_num,"Recall ",Recall/ques_num)
    return Prec/ques_num, Recall/ques_num


def main():
    dataDir = "gendata/zhidao_82"
    modelDir = "model"
    resultDir = "result/zhidao_82"
    if not os.path.exists("model"):
        os.mkdir("model")
    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists(dataDir):
        raise ValueError("data dictionary not exists or unreadable")
    datatype = sys.argv[1]
    model_c = sys.argv[2]
    model_t = sys.argv[3]
    #i = int(sys.argv[4])
    if datatype == "search" or datatype == "zhidao":
        dataDir = os.path.join(dataDir,datatype)
        modelDir = os.path.join(modelDir,datatype)
        resultDir = os.path.join(resultDir,datatype)
    else:
        raise ValueError("type error")
    trainDir = dataDir+".train_"
    testDir = dataDir+".test_"
    modelDir = modelDir+".model_"
    resultDir = resultDir+".result_"
    pre,recall = 0.0,0.0
    for i in range(1,6):
        print("svm training "+trainDir+str(i))
        x = os.popen("./../svm_rank/svm_rank_learn -c "+model_c+" -t "+model_t+" "+trainDir+str(i)+" "+modelDir+str(i)).read()  # train shell
        print("train end...\n")
        print("svm testing "+testDir+str(i))
        y = os.popen("./../svm_rank/svm_rank_classify "+testDir+str(i)+" "+modelDir+str(i)+" "+resultDir+str(i)).read() #predict shell
        print("test end..\n")
        e = evalute(datatype,resultDir,i)
        pre += e[0]
        recall += e[1]
    print("avg precision",pre/5," recall ",recall/5)
    j = input() # 取第一个模型对全数据进行评估
    y = os.popen("./../svm_rank/svm_rank_classify " + os.path.join("gendata",datatype)+".dev.all" + " " + modelDir + str(j) + " " + resultDir +"all"+ str(j)).read()  # predict shell
    print(evalute(datatype, resultDir, j))
    # print("test end..\n")
    # evalute(datatype,resultDir,i)

if __name__=="__main__":
    main()

"""
search filtered 181897  181900  182426 182592 183701 183955 184042 184309 184402

search @5  precis 0.128 NDCG 0.3373
zhidao @5 precis 0.4234 NDCG 0.5263
"""