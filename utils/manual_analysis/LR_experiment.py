import sys
if sys.version[0] == '2':
	reload(sys)
	sys.setdefaultencoding("utf-8")
import json
import os
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.externals import joblib#用于保存模型

def load_manual_labeled_jsonfiles(manualjsonfiledir):
	manual_labeled_samples=[]
	filenames=range(100)#有序
	for filename in filenames:
		with open(os.path.join(manualjsonfiledir,str(filename)+'.json'), "r") as fr:
			manual_labeled_samples.append(json.loads(fr.readline(), encoding='utf-8'))
	print('loading manual labeled json files done!')
	return manual_labeled_samples

def load_feature_data(feature_data_path):
	feature_data=json.load(open(feature_data_path,'r'))
	print('loading feature data done!')
	return feature_data

def genet_dataset(manualjsonfiledir, feature_data_path,filtered_path=None):
	manual_labeled_samples = load_manual_labeled_jsonfiles(manualjsonfiledir)
	feature_data=load_feature_data(feature_data_path)

	if filtered_path:
		filtered_qids=[]
		with open(filtered_path,'r') as sf:
			filtered_qids=sf.readline().strip().split()
	
	dataset=[]
	target_data=[]
	lIdx2dIdx2pIdx=[]
	for sam_idx,sample in enumerate(manual_labeled_samples):
		if len(sample['answers'])==0:
			print('qid: {} is filtered coz no answer'.format(sample['question_id']))
			continue
		if filtered_path and str(sample['question_id']) in filtered_qids:
			print('qid: {} is filtered coz special no answer'.format(sample['question_id']))
			continue
		
		assert feature_data['feature_data'][sam_idx]['question_id']==sample['question_id']
		assert feature_data['qid2featLidx'][sam_idx]['qid']==sample['question_id']
		
		
		dataset_sample=feature_data['feature_data'][sam_idx]['feature_list']
		dataset.append(dataset_sample)

		lIdx2dIdx2pIdx_sample=[]
		for i in range(len(dataset_sample)):
			lIdx2dIdx2pIdx_sample.append(feature_data['qid2featLidx'][sam_idx]['lIdx2dIdx2pIdx'][str(i)])
		lIdx2dIdx2pIdx.append(lIdx2dIdx2pIdx_sample)

		target_data_sample=[]
		for d_idx,doc in enumerate(sample['documents']):
			target_data_doc=[0]*len(doc['paragraphs'])#初始化为全0
			
			manualAParas=doc['fake_paras'][0]
			manualBParas=doc['fake_paras'][1]
			for i in range(len(target_data_doc)):
				if i in manualAParas:
					target_data_doc[i]=1
				elif i in manualBParas:
					target_data_doc[i]=1
			target_data_sample += target_data_doc
		target_data.append(target_data_sample)

		assert len(dataset_sample)==len(lIdx2dIdx2pIdx_sample)
		assert len(dataset_sample)==len(target_data_sample)

		assert len(dataset)==len(lIdx2dIdx2pIdx)
		assert len(dataset)==len(target_data)

	print('len(dataset)',len(dataset))

	return dataset, target_data, lIdx2dIdx2pIdx#X,Y全数据集

def LR_algo(manualjsonfiledir, feature_data_path, filtered_path=None, feature_select_idx=range(6)):
	X,y,lIdx2dIdx2pIdx=genet_dataset(manualjsonfiledir, feature_data_path, filtered_path)
	# print('train_test_split, train_size=0.6, test_size=0.4, random_state=1136')
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1136)
	# _, _, lIdx2dIdx2pIdx_train, lIdx2dIdx2pIdx_test = train_test_split(X, lIdx2dIdx2pIdx, test_size=0.4, random_state=1136)

	# X_train_data=[]
	# for item in X_train:
	# 	X_train_data+=item

	# X_test_data=[]
	# for item in X_test:
	# 	X_test_data+=item

	# y_train_data=[]
	# for item in y_train:
	# 	y_train_data+=item

	# y_test_data=[]
	# for item in y_test:
	# 	y_test_data+=item

	print('封闭训练...')
	X_train_data=[]
	for item in X:
		X_train_data+=item

	X_test_data=[]
	for item in X:
		X_test_data+=item

	y_train_data=[]
	for item in y:
		y_train_data+=item

	y_test_data=[]
	for item in y:
		y_test_data+=item

	lIdx2dIdx2pIdx_test=lIdx2dIdx2pIdx

	print('feature_select_idx',feature_select_idx)
	feature_names=['f1','recall','is_first_para','tfidf','bleu','overlap']
	print('feature_names',feature_names)
	print('feature_names_used',[f_name for i,f_name in enumerate(feature_names) if i in feature_select_idx])

	X_train_data=np.array(X_train_data)[:,feature_select_idx]
	X_test_data=np.array(X_test_data)[:,feature_select_idx]

	print('X_train_data.shape',X_train_data.shape)
	#数据归一化---Z-score
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	sc.fit(X_train_data)
	#保存归一化模型
	joblib.dump(sc,'quesMatch_ml_models/sc.model')
	
	X_train_std = sc.transform(X_train_data)
	X_test_std = sc.transform(X_test_data)

	#数据标准化---离差标准化
	# from sklearn.preprocessing import MinMaxScaler
	# mmc = MinMaxScaler()
	# mmc.fit(X_train_data)
	# X_train_std = mmc.transform(X_train_data)
	# X_test_std = mmc.transform(X_test_data)

	from sklearn.linear_model import LogisticRegression
	print('LogisticRegression(C=1.0, random_state=0)')
	lr = LogisticRegression(C=1.0, random_state=0)
	lr.fit(X_train_std, y_train_data)
	#保存归一化模型
	joblib.dump(lr,'quesMatch_ml_models/lr.model')

	print('lr.coef_(coefs of the features)：',lr.coef_)
	predictions=lr.predict(X_test_std)#不知道是否有输出precision
	acc = accuracy_score(y_test_data, predictions)
	print('accuracy',acc)

	# lr.predict_proba(X_test_std[0,:]) # 查看第一个测试样本属于各个类别的概率
	# print(lr.predict_proba([X_test_std[0,:]]))

	pred_probs=lr.predict_proba(X_test_std)#[N,2]

	precision_list=[]
	recall_list=[]
	offset=0
	for sample_lIdx2dIdx2pIdx in lIdx2dIdx2pIdx_test:
		para_nums=len(sample_lIdx2dIdx2pIdx)#该样本总篇章数
		
		target_data_sample=np.array(y_test_data[offset : offset+para_nums+1])

		pred_probs_sample=pred_probs[offset : offset+para_nums+1][:,-1]
		top5Idx=np.argsort(-pred_probs_sample)[:5]#为1的概率从高到低排序，取top5

		# print('top5Idx',top5Idx)
		# print(pred_probs_sample[top5Idx])
		# print(pred_probs_sample[:5])

		right_num=sum(target_data_sample[top5Idx])
		top5precision_sample=right_num/len(top5Idx)
		top5recall_sample=right_num/sum(target_data_sample)

		precision_list.append(top5precision_sample)
		recall_list.append(top5recall_sample)

		offset+=para_nums

	macro_precision=np.mean(np.array(precision_list))
	macro_recall=np.mean(np.array(recall_list))
	print('macro precison of testset: ',macro_precision)
	print('macro recall of testset: ',macro_recall)


# manualjsonfiledirs=['search','zhidao']
# feature_data_paths=['quesMatch_ml/search.dev_quesMatch_ml.json','quesMatch_ml/zhidao.dev_quesMatch_ml.json']
# filtered_paths=['search.filtered.txt',None]
# feature_select_idxs=[[0,1,2,5],[1,2]]

manualjsonfiledirs=['search']
feature_data_paths=['quesMatch_ml/search.dev_quesMatch_ml.json']
filtered_paths=['search.filtered.txt']
feature_select_idxs=[[0,1,2,5]]

for manualjsonfiledir,feature_data_path,filtered_path,feature_select_idx in zip(manualjsonfiledirs,feature_data_paths,filtered_paths,feature_select_idxs):
	print('run for ',manualjsonfiledir)
	# LR_algo(manualjsonfiledir,feature_data_path,filtered_path,feature_select_idx)
	LR_algo(manualjsonfiledir,feature_data_path,filtered_path)