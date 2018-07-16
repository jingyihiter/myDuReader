import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import json

qa_resultPath='/home/yhli/DuReader/DuReader-master_v1/out/09/results/test.predicted_09.json'
yesno_resultPath='/home/yhli/DuReader/DuReader-master_v1/out/07/results/test.predicted_09.YESNO.json'
out_file_path='/home/yhli/DuReader/DuReader-master_v1/out/07/results/test.predicted_09.class.json'

#首先载入YESNO部分的预测结果
yesno_records={}
with open(yesno_resultPath,'r') as f_in:
	for line in f_in:
		sample = json.loads(line)
		yesno_records[sample['question_id']]=line


with open(qa_resultPath,'r') as f_in:
	with open(out_file_path, 'w') as f_out:
		for line in f_in:
			sample = json.loads(line)
			if sample['question_id'] in yesno_records:
				line=yesno_records[sample['question_id']]
			f_out.write(line)

print('all done!')