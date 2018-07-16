#相比于v1 和 v3，区别在于训练集数据预处理采用全局挑选，而开发集和测试集数据预处理，search数据的开发集和测试集选para采用6个特征，借助ML挑选
#6个特征：f1,recall,is_first_para,tfidf,bleu,overlap
训练集para是通过answers和para在Recall下全局挑选top5，span是answers和各个para中所有可能span在F1下全局挑选的【选了前3，只用top1】【doc为is_selected==False不影响，仍挑选】

模型是在baseline基础上：
1. glove预训练词向量
2. 添加self-matching层，并且损失计算采用了shared-normalization机制
3. 确定最优模型时，改用Rougle-L作为最优指标

数据预处理：
采用baseline方案：
1)、即训练集para采用Recall计算各para和answers的匹配程度，全局挑选top5；
span是answers和所有paras，采用F1指标穷举各span与answers的匹配程度【doc的is_selected为False对应的para仍在挑选范围】
2)、zhidao:开发集和测试集的para的挑选采用ques和para在Recall上计算匹配，每个doc挑选一个最优。【与baseline一致】
3)、search:开发集和测试集的para的挑选采用ques和para之间计算6个特征，Logistic Regression融合特征，全局挑选TOP5 paras

#调用：
TRAIN: python run.py --train --algo BIDAF run_id 1 --dropout_keep_prob 1.0 --epochs 10
PREDICT: python run.py --predict --algo BIDAF run_id 1