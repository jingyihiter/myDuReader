import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import os
import json

test_files=['../data/preprocessed/testset/search.test.json','../data/preprocessed/testset/zhidao.test.json']
test_files_v1=['../data/preprocessed/testset_v1/search.test.json','../data/preprocessed/testset_v1/zhidao.test.json']

for test, test_v1 in zip(test_files,test_files_v1):
	print('comparision......')
	qids=[]
	with open(test,'r') as f:
		for line in f:
			sample=json.loads(line)
			qids.append(sample['question_id'])
	qids_v1=[]
	with open(test_v1,'r') as f1:
		for line in f1:
			sample=json.loads(line)
			qids_v1.append(sample['question_id'])
	qid_set=set(qids)
	qid_set_v1=set(qids_v1)
	diff_qid_set=qid_set-qid_set_v1
	if len(diff_qid_set)>0:
		print(len(diff_qid_set))