#相比于v1，区别在于训练集、开发集和测试集数据预处理，全部采用全局挑选
训练集para是通过answers和para在Recall下全局挑选top5，span是answers和各个para中所有可能span在F1下全局挑选的【选了前3，只用top1】【doc为is_selected==False不影响，仍挑选】
开发集和测试集是question和para在recall下全局挑选top5

模型是在baseline基础上：
1. glove预训练词向量
2. 添加self-matching层，并且损失计算采用了shared-normalization机制
3. 确定最优模型时，改用Rougle-L作为最优指标

数据预处理：
采用baseline方案：
1)、即训练集para采用Recall计算各para和answers的匹配程度，全局挑选top5；
span是answers和所有paras，采用F1指标穷举各span与answers的匹配程度【doc的is_selected为False对应的para仍在挑选范围】
2)、开发集和测试集是question和para在recall下全局挑选top5

#调用：
TRAIN: python run.py --train --algo BIDAF run_id 1 --dropout_keep_prob 1.0 --epochs 10
PREDICT: python run.py --predict --algo BIDAF run_id 1