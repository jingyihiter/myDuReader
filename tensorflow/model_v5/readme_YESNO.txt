#相比于v1，区别是多了YESNO模型
#两个model：BIDAF for QA & YESNO for YESNO classification
#其中的BIDAF为baseline基础上glove预训练词向量、self-matching层、 shared normalization计算loss
#YESNO模型，输入为问题和answer，self-matching后，对answer的表示做max-pooling然后接全连接层做三分类

QA模型是在baseline基础上：
1. glove预训练词向量
2. 添加self-matching层，并且损失计算采用了shared-normalization机制
3. 确定最优模型时，改用Rougle-L作为最优指标
YESNO模型：输入为问题和answer，self-matching后，对answer的表示做max-pooling然后接全连接层做三分类

数据预处理：
QA采用baseline方案：
1)、即训练集para采用Recall计算各para和answers的匹配程度，全局挑选top5；
span是answers和所有paras，采用F1指标穷举各span与answers的匹配程度【doc的is_selected为False对应的para仍在挑选范围】
2)、开发集和测试集是question和para在recall下每个doc挑选一个最优

#调用：
YESNO TASK:
TRAIN: python run.py --train --algo YESNO run_id 7777 --dropout_keep_prob 1.0 --epochs 10
PREDICT: python run.py --predict --algo YESNO run_id 7 #run_id=7是已训好的YESNO model

QA TASK:(待Debug)
TRAIN: python run.py --train --algo BIDAF run_id 1 --dropout_keep_prob 1.0 --epochs 10
PREDICT: python run.py --predict --algo BIDAF run_id 1