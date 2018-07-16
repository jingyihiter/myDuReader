import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import json

train_files_contest=['../data/preprocessed/trainset_v1/search.train.json', '../data/preprocessed/trainset_v1/zhidao.train.json']
train_files_DuReader=['../data/preprocessed/trainset/search.train.json', '../data/preprocessed/trainset/zhidao.train.json']
out_files=['../data/preprocessed/trainset_v1/search.train.diffQids.json','../data/preprocessed/trainset_v1/zhidao.train.diffQids.json']

for train_file_DuReader, train_file_contest, out_file in zip(train_files_DuReader, train_files_contest, out_files):
	subsetQids = []#subset
	with open(train_file_DuReader,'r') as f_in:
		for line in f_in:
			sample = json.loads(line)
			subsetQids.append(int(sample['question_id']))
	subsetQids_set=set(subsetQids)

	with open(train_file_contest,'r') as f_in:
		with open(out_file,'w') as f_out:
			for line in f_in:
				sample = json.loads(line)
				if int(sample['question_id']) not in subsetQids_set:
					f_out.write(line)
			
	print('done')