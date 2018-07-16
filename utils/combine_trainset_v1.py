import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import json

train_files_contest=['../data/preprocessed/trainset_v1/search.train.json', '../data/preprocessed/trainset_v1/zhidao.train.json']
train_files_9w=['../data/preprocessed/trainset_v1/search.train.diffQids_recall_f1.json', '../data/preprocessed/trainset_v1/zhidao.train.diffQids_recall_f1.json']
out_files=['../data/preprocessed/trainset_v1/search.train_recall_f1.json','../data/preprocessed/trainset_v1/zhidao.train_recall_f1.json']
for train_file_contest, train_file_9w, out_file in zip(train_files_contest, train_files_9w, out_files):
	subset_data = {}#subset-9w
	with open(train_file_9w,'r') as f_in:
		for line in f_in:
			sample = json.loads(line)
			subset_data[sample['question_id']]=sample
	
	with open(train_file_contest,'r') as f_in:
		with open(out_file,'a') as f_out:
			for line in f_in:
				sample = json.loads(line)
				if sample['question_id'] in subset_data:
					sample['paragScore_recall_a']=subset_data[sample['question_id']]['paragScore_recall_a']
					sample['multi_spanScore_f1']=subset_data[sample['question_id']]['multi_spanScore_f1']
					f_out.write(json.dumps(sample, ensure_ascii=False)+'\n')
	print('done 1')

# for fileidx, out_file in enumerate(out_files):
# 	line_num=0
# 	with open(out_file,'r') as f_in:
# 		for line in f_in:
# 			sample = json.loads(line)
# 			line_num+=1
# 	print('done1 ',line_num)