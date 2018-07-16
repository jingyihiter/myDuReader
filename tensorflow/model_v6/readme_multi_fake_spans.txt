#相比于v4，区别是训练集采用multi answer span机制，全局选多answer span【对每个answer挑选一个最佳匹配span】

QA模型是在baseline基础上：
1. glove预训练词向量
2. 添加self-matching层，并且损失计算采用了shared-normalization机制
3. 确定最优模型时，改用Rougle-L作为最优指标


数据预处理：
QA采用baseline方案：
1)、即训练集para采用Recall计算各para和answers的匹配程度，全局挑选不一定top5，有可能到TOP9；
span是对每个answer，它和所有paras，采用F1指标穷举各span与answers的匹配程度【doc的is_selected为False对应的para仍在挑选范围】
那么fake span预处理得到每个sample多个fake_span(至多3个)
形成样本时，对于每个question样本，其para为1正，3top para，1顺次选，形成样本至多3个

2)、开发集和测试集与baseline保持一致，question和para在Recall下计算para匹配程度,每个doc挑选一个最优匹配

#调用：
TRAIN: python run.py --train --algo BIDAF run_id 1 --dropout_keep_prob 1.0 --epochs 10
PREDICT: python run.py --predict --algo BIDAF run_id 1