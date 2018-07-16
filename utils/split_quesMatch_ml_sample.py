import sys
if sys.version[0] == '2':
	reload(sys)
	sys.setdefaultencoding("utf-8")

import json
import numpy as np

def split(feature_path, qid2featLidx_path, samplepath, out_path):
	import linecache
	sampled_qids = linecache.getline(samplepath, 2).strip().split()
	print('load sample indexs file done!')
	
	# feature_data=[{}]*len(sampled_qids)
	# with open(feature_path, "r") as fr:
	# 	for line in fr:
	# 		feature_list = json.loads(line)

	# 		question_id=feature_list[0][1]
	# 		if str(question_id) in sampled_qids:
	# 			idx=sampled_qids.index(str(question_id))
	# 			feature_array = np.array(feature_list)[:,2:]
	# 			assert feature_array.shape[1]==6 #6个特征
	# 			feature_data[idx]={'question_id':question_id, 'feature_list':feature_array.tolist()}

	feature_data=[]
	with open(feature_path, "r") as fr:
		for line in fr:
			feature_list = json.loads(line)

			question_id=feature_list[0][1]
			if str(question_id) in sampled_qids:
				feature_array = np.array(feature_list)[:,2:]
				assert feature_array.shape[1]==6 #6个特征
				feature_data.append({'question_id':question_id, 'feature_list':feature_array.tolist()})

	print('load feature file done!')
	assert len(feature_data)==100

	# qid2featLidx=[{}]*len(sampled_qids)
	# with open(qid2featLidx_path, "r") as fr:
	# 	for line in fr:
	# 		sample=json.loads(line)
	# 		if str(sample['qid']) in sampled_qids:
	# 			idx=sampled_qids.index(str(sample['qid']))
	# 			qid2featLidx[idx]=sample
	
	qid2featLidx=[]
	with open(qid2featLidx_path, "r") as fr:
		for line in fr:
			sample=json.loads(line)
			if str(sample['qid']) in sampled_qids:
				qid2featLidx.append(sample)

	print('load qid2featLidx done!')
	assert len(qid2featLidx)==100
	
	rst={'feature_data':feature_data,'qid2featLidx':qid2featLidx}
	with open(out_path, "w") as fw:
		fw.write(json.dumps(rst))

feature_files=['../data/preprocessed/devset_v1/search.dev_ml_feature_data.json','../data/preprocessed/devset_v1/zhidao.dev_ml_feature_data.json']
qid2featLidx_paths=['../data/preprocessed/devset_v1/search.dev_ml_qid2featLidx.json','../data/preprocessed/devset_v1/zhidao.dev_ml_qid2featLidx.json']
dev_sample_paths=['manual_analysis/100sample/100.search.dev.txt','manual_analysis/100sample/100.zhidao.dev.txt']
out_paths=['manual_analysis/quesMatch_ml/search.dev_quesMatch_ml.json','manual_analysis/quesMatch_ml/zhidao.dev_quesMatch_ml.json']

for feature_path, qid2featLidx_path, samplepath, out_path in zip(feature_files,qid2featLidx_paths,dev_sample_paths,out_paths):
	split(feature_path, qid2featLidx_path, samplepath, out_path)
	print(feature_path,'is done!')
