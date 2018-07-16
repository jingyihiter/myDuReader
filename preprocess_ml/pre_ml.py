#!/usr/bin/env python
# _*_ coding:utf-8 _*_
"""
svmrank for DuReader preprocess
from cornell
"""
import os
import json
import sys
from score_funcs import metric_max_over_ground_truths, f1_score, recall
from sklearn.metrics import pairwise_distances
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import datetime
import pickle

def loadData(filedir):
    """

    :param filedir:
    :return:
    """
    sample_fake_dict = dict()
    print(filedir)
    if filedir == "data/search/":
        print("loading search data")
    elif filedir == "data/zhidao/":
        print("loading zhidao data")
    else:
        raise ValueError("filedir wrong")
        # print("filedir wrong")
    for filename in range(100):
        with open(os.path.join(filedir,str(filename)+".json"), "r", encoding='utf-8') as f:
            sample = json.loads(f.readline(), encoding='utf-8')
            qid = sample["question_id"]
            fake = dict()
            for did,doc in enumerate(sample["documents"]):
                fake_para = doc["fake_paras"]
                a = fake_para[0] # level a
                b = fake_para[1] # level b
                n = len(doc["paragraphs"])
                c = [x for x in range(n) if x not in a + b]
                ''' # svm 三分类问题，三类'''
                fake[did] = [a, b, c] # level a b c
                # svm rank
                d = [x for x in b if x not in a]
                #fake[did] = [[n-x-1 for x in a + d + c],[len(a),len(d),len(c)]]  # rank reverse
            sample_fake_dict[qid]=fake

    return sample_fake_dict

def write(sample, questionid, ls, file, sfile, seg_question,tfidfObj, level):
    question_text = " ".join(seg_question)
    did_pids = list()
    min_tfidfscore, max_tfidfscore = 0.0,0.0
    min_overlap,max_overlap=0,0
    # feature list
    para_f1s,para_recalls,first_paras,tfidf_scores,bleu_scores,word_overlap = list(),list(),list(),list(),list(),list()
    for item in ls:
        did = item[0]
        for pid in item[1]:
            did_pids.append([did,pid])
            para_tokens = sample["documents"][did]["segmented_paragraphs"][pid]
            para_f1 = metric_max_over_ground_truths(f1_score, para_tokens, [seg_question])
            para_f1s.append(para_f1)
            para_recall = metric_max_over_ground_truths(recall, para_tokens, [seg_question])
            para_recalls.append(para_recall)

            first_para = 1 if pid == 0 else 0
            first_paras.append(first_para)

            para_text = " ".join(para_tokens)
            para_features = tfidfObj.transform([para_text])
            q_features = tfidfObj.transform([question_text])
            tfidf_score = (pairwise_distances(para_features, q_features, "cosine").ravel())[0] # 余弦相似度取值[-1,1]，通过tfidf得到貌似只有[0,1]，加abs为了保险
            if min_tfidfscore<tfidf_score: min_tfidfscore=tfidf_score
            if max_tfidfscore>tfidf_score: max_tfidfscore=tfidf_score
            tfidf_scores.append(tfidf_score)

            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu(seg_question, para_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu_score)

            stop_words = stopwords.words("chinese")
            q_words = {x for x in seg_question if x not in stop_words}
            found = set()
            [found.add(word) for word in para_tokens if word in seg_question]
            overlap = len(found)
            #print(overlap)
            if overlap<min_overlap: min_overlap=overlap
            if overlap>max_overlap: max_overlap=overlap
            word_overlap.append(overlap)
    feature =list()
    target = list()
    for i in range(len(para_f1s)):
        # normalization
        tfidf_sc = (tfidf_scores[i]-min_tfidfscore)/(max_tfidfscore-min_tfidfscore) if max_tfidfscore-min_tfidfscore >0 else 0 # 分母为0时
        overlap_ratio = (word_overlap[i]-min_overlap)/(max_overlap-min_overlap) if (max_overlap-min_overlap)>0 else 0
        sfile.write(str(questionid) + "\t" + str(did_pids[i][0]) + "\t" + str(did_pids[i][1]) + "\t" + str(level) + "\n")  # 一一对应
        # 1:para_f1 2:para_recall 3:first_para 4:tfidf 5:belu_score 6:word_overlap
        # str(round(para_f1s[i], 4)) + " 2:" +
        file.write(str(level) + " qid:" + str(questionid) +  \
                    " 1:" +  str(round(para_recalls[i], 4)) + "\n")#\
                   #" 2:" + str(overlap_ratio) +\
                   # " 3:" + str(round(tfidf_scores[i], 4)) + "\n")
                    #+ " 5:" + str(bleu_scores[i]) +" 6:"+str(overlap_ratio)+ "\n")
        feature.append([para_f1s[i],para_recalls[i],first_paras[i],tfidf_sc,bleu_scores[i],overlap_ratio])
        target.append(level)
    return feature,target


def getFeature(sample, file, sfile, sample_fake_dict, tfidfObj):
    """

    :param sample:
    :return:
    """
    questionid = sample["question_id"]
    seg_question = sample["segmented_question"]
    question_text = " ".join(seg_question)
    fake_dict = sample_fake_dict[questionid]
    """
    所有的一级的放前面，二级的放后面，剩下的放最后
    """
    num = 0
    a_list = list()
    b_list = list()
    c_list = list()
    for did,fake in fake_dict.items():
        a_list.append([did,fake[0]])
        b_list.append([did,fake[1]])
        c_list.append([did,fake[2]])
        #num +=(len(fake[0])+len(fake[1])+len(fake[2]))
    feature,target=list(),list()
    f3, t3 = write(sample,questionid,a_list,file,sfile,seg_question,tfidfObj,3)
    feature += f3
    target += t3
    f2, t2 = write(sample,questionid,b_list,file,sfile,seg_question,tfidfObj,2)
    feature += f2
    target += t2
    f1, t1 = write(sample,questionid,c_list,file,sfile,seg_question,tfidfObj,1)
    feature += f1
    target += t1
    '''
    for did,doc in enumerate(sample["documents"]):
        seg_title = doc["segmented_title"]
        fake_para = fake_dict[did][0] # target para
        para_num = fake_dict[did][1]
        
        title_f1 = metric_max_over_ground_truths(f1_score, seg_title, seg_question)
        title_recall = metric_max_over_ground_truths(recall, seg_title, seg_question)
        bs_rank_pos = doc["bs_rank_pos"] # unuseful

        para_score = dict()
        for pid,para_tokens in enumerate(doc["segmented_paragraphs"]):
            para_f1 = metric_max_over_ground_truths(f1_score, para_tokens, [seg_question])
            para_recall = metric_max_over_ground_truths(recall, para_tokens, [seg_question])
            para_score[pid] = [para_f1, para_recall]

            #whether the paragraph was the first in document
            first_para = 1 if pid==0 else 0

            para_text = " ".join(para_tokens)
            para_features = tfidfObj.transform([para_text])
            q_features = tfidfObj.transform([question_text])
            tfidf_score = pairwise_distances(para_features, q_features, "cosine").ravel()
            smoothie = SmoothingFunction().method4

            bleu_score = sentence_bleu(seg_question, para_tokens, smoothing_function=smoothie)
            #bleu_title_score = sentence_bleu(seg_question, seg_title, smoothing_function=smoothie) # unuseful
            # 1:para_f1 2:para_recall 3:first_para 4:tfidf 5:belu_score
            file.write(str(fake_para[pid])+" qid:"+str(questionid)+str(did)+" 1:"+str(round(para_f1,4))+" 2:"+str(round(para_recall,4))+\
                      #" 3:"+str(word_match_num)+" 4:"+str(round(word_match_ratio,4))+
                     " 3:"+str(first_para)+" 4:"+str(round(tfidf_score[0],4))\
                      +" 5:"+str(bleu_score)+"\n")
    '''
    #return len(sample["documents"])
    return feature,target


def genet_tfidfObj(filename, tfidfObj):
    text = []#整个语料库组成文档集
    # for data_path in args.train_files + args.dev_files + args.test_files:
    #debug
    with open(filename, "r", encoding="utf-8") as f_in:
        for line in f_in:
            sample = json.loads(line)
            text.append(' '.join(sample['segmented_question'])) #加入问题
            for doc in sample['documents']:
                for para_tokens in doc['segmented_paragraphs']: #加入篇章
                    text.append(' '.join(para_tokens))
    try:
        tfidfObj.fit(text)
        # print('bag of words num:',len(tfidfObj.get_feature_names()))
    except ValueError:
        pass


def preprocess(filedir, filename, targetfile, safile):
    tfidfObj = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stopwords.words('chinese'))
    genet_tfidfObj(filename, tfidfObj)
    num = 0
    fake_dict = loadData(filedir)
    tarfile = open(targetfile, "w", encoding='utf-8')
    sfile = open(safile, "w", encoding="utf-8")
    feature,target=list(),list()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line, encoding='utf-8')
            if len(sample["answers"])<1:continue
            f,t= getFeature(sample, tarfile, sfile, fake_dict, tfidfObj)
            feature += f
            target += t
    tarfile.close()
    sfile.close()
    with open(filename+".data3.pkl","wb") as f:
        pickle.dump(feature,f,2)
        pickle.dump(target,f,2)

def main():
    #   python pre_ml.py data/zhidao/ data/zhidao_82/zhidao.train.json gendata/zhidao_82/zhidao.train sfile/zhidao_82/zhidao.train.result
    #   python pre_ml.py data/zhidao/ data/zhidao_82/zhidao.test.json gendata/zhidao_82/zhidao.test sfile/zhidao_82/zhidao.test.result
    filedir = sys.argv[1]
    filename = sys.argv[2]
    targetfile = sys.argv[3]
    safile = sys.argv[4]
    start = datetime.datetime.now()
    #[preprocess(filedir, filename+"_"+str(i), targetfile+"_"+str(i), safile+"_"+str(i)) for i in range(1,6)]
    preprocess(filedir,filename,targetfile,safile)
    end = datetime.datetime.now()
    print(end-start)
if __name__== "__main__":
    main()

