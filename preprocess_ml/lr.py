#!/usr/bin/env python
# _*_ coding:utf-8 _*_
"""
逻辑斯谛回归
"""

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

def loadData(filename):
    with open(filename,"rb") as f:
        feature = pickle.load(f)
        target = pickle.load(f)
        return feature,target

def mylr(filename,datatype,i):
    feature,target = loadData(filename)
    feature = np.array(feature)
    #print(feature[0])
    # feature = feature[:,1].reshape([feature.shape[0],1]) # recall
    # feature = feature[:,:2].reshape([feature.shape[0], 2])  # f1+recall
    # feature = feature[:, [0,2]].reshape([feature.shape[0], 2])  # f1+firstpara
    # feature = feature[:, [1,2]].reshape([feature.shape[0], 2]) # recall+firstpara
    # feature = feature[:, [1, 4]].reshape([feature.shape[0], 2])  # recall+tfidf
    # feature = feature[:, [1, 5]].reshape([feature.shape[0], 2])    # recall+wordoverlap
    # feature = feature[:, i].reshape([feature.shape[0], 1])
    #print(feature[0])
    target = np.array(target)
    print(feature.shape,target.shape)
    # return
    # 划分训练集和测试集
    feature_train,feature_test,target_train,target_test = train_test_split(feature,target,test_size=0.4,random_state=0)
    sc = StandardScaler()
    sc.fit(feature_train)
    feature_train_std = sc.transform(feature_train)
    feature_test_std = sc.transform(feature_test)

    # LR
    lr = LogisticRegression(penalty='l2',C=1,random_state=0)
    lr.fit(feature_train_std,target_train)

    # predict
    target_pred = lr.predict_proba(feature_test_std)
    print(target_pred[0])
    target_predict = lr.predict(feature_test_std)
    print(target_predict[0])
    score_pre = lr.score(feature_test_std,target_test)
    print(score_pre)
    # weg = lr.get_params(deep=True)
    # print(weg)
    # print(feature_test[0],target_test[0])

    # save model
    with open(datatype+".model.pkl","wb") as f:
        pickle.dump(lr,f)

def main():
    zhidao_f = "data/100.zhidao.dev.json.data.pkl"
    search_f = "data/100.search.dev.json.data.pkl"
    # for i in range(6):
    i=0
    print("zhidao...",i)
    mylr(zhidao_f,"zhidao",i)
    print("search...",i)
    mylr(search_f,"search",i)

    # # load model
    test = [[0.05,0.42857143,0,0,0.162266,0.42857143]]
    test = np.array(test)
    with open("search.model.pkl","rb") as f:
        lr = pickle.load(f)
        x = lr.predict_proba(test) # 概率
        y = lr.predict(test) # 类别
        print(x)

if __name__=="__main__":
    main()

