import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import os
import json

# #全部猜测Yes
# resultPath='/home/yhli/DuReader/DuReader-master_v1/out/06/results/test.predicted.json'
# out_file_path='/home/yhli/DuReader/DuReader-master_v1/out/06/results/test.predicted_class.json'
# with open(resultPath,'r') as f_in:
# 	with open(out_file_path, 'w') as f_out:
# 		for line in f_in:
# 			sample = json.loads(line)
# 			if sample['question_type']=='YES_NO':
# 				sample["yesno_answers"].append('Yes')
# 			f_out.write(json.dumps(sample, ensure_ascii=False)+'\n')
# print('done')

#统计label
train_files=['../data/preprocessed/trainset_v1/search.train.json', '../data/preprocessed/trainset_v1/zhidao.train.json']

yesNum, noNum, dependNum=0,0,0
for train_file in train_files:
	with open(train_file,'r') as f_in:
		for line in f_in:
			sample = json.loads(line)
			if sample['question_type']=='YES_NO':
				for classlabel in sample['yesno_answers']:
					if classlabel=='Yes':
						yesNum+=1
					elif classlabel=='No':
						noNum+=1
					elif classlabel=='Depends':
						dependNum+=1
print(yesNum,noNum,dependNum)
						