# -*- coding: UTF-8 -*-
import sys
if sys.version[0] == '2':
	reload(sys)
	sys.setdefaultencoding("utf-8")

import os
import argparse
import json
import multiprocessing as multiprocess
from collections import Counter
import time

#优化效率后的通过f1找多fake span(为原数据集中的每个answer都找到对应的最匹配的fake span)
#处理最新训练集-27w

def parse_args():
	#usage python preprocess_yhl.py recall f1 a -n 8
	parser = argparse.ArgumentParser(description='Preprocess DuReader preprocessed dataset includs calculating parag scores and fake span scores.')
	
	path_settings = parser.add_argument_group('path settings')
	#, '../data/preprocessed/trainset_v1/zhidao.train.json'
	#6160负责跑search
	path_settings.add_argument('--train_files', nargs='+',
							   default=['../data/preprocessed/trainset_v1/search.train.json'],
							   help='list of files that contain the preprocessed train data')

	parser.add_argument('-n', '--n_processes', type=int, default=3,
						help="Number of processes (i.e., ) ")
	return parser.parse_args()

def scoreSpan(line):
	sample=json.loads(line)
	answer_num = len(sample['segmented_answers'])
	bestK_match = [(-1,-1,[-1,-1],None,0.0)] * answer_num#slot:(d_idx, p_idx, span_idxs, span_text, matchScore)
	
	for d_idx, doc in enumerate(sample['documents']):
		for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):#debug每个篇章选一个fake_span出来
			for a_idx, seg_answer_tokens in enumerate(sample['segmented_answers']):#为每个answer找到最匹配的fake_span
				answer_len=len(seg_answer_tokens)#为了计算recall

				answer_tokens = set([token for token in seg_answer_tokens if token not in ['，','。']])#去停用词？
				
				best_f1_score_p=0.0
				best_start_idx, best_end_idx = -1, -1
				best_span_tokens=None

				f1_score = 0#一旦找到全匹配，不再测试其他开始位置
				truncated_para_tokens = para_tokens[:1000]
				for start_tidx in range(len(truncated_para_tokens)):
					if truncated_para_tokens[start_tidx] not in answer_tokens:
						continue
					if f1_score==1.0:#在某个开始位置找到全匹配后，余下开始位置无需再探索...
						break
					#初始化第一个span
					end_tidx_init = len(truncated_para_tokens)-1
					span_len = end_tidx_init +1 - start_tidx#为了计算precision
					span_tokens = truncated_para_tokens[start_tidx: end_tidx_init+1]

					common = Counter(span_tokens) & Counter(seg_answer_tokens)
					num_same = sum(common.values())
					if num_same == 0:
						f1_score = 0
					else:
						p = 1.0 * num_same / span_len
						r = 1.0 * num_same / answer_len
						f1_score = (2 * p * r) / (p + r)#调和平均
					# print('针对answer token: ',seg_answer_tokens,'初始span token ', span_tokens,'num same: ',num_same,'p: ',p)
					if f1_score==0:#说明该开始位置对应的最长的可能span都没有相交部分
						break

					if f1_score > best_f1_score_p:
						# print('best_f1_score_p被更新【初始时】',f1_score)
						best_f1_score_p = f1_score
						best_start_idx, best_end_idx = start_tidx, end_tidx_init
						best_span_tokens = span_tokens
					if f1_score==1.0:#该开始位置，找到了最匹配span，结束位置再继续缩小无意义
							break
					#遍历其余span，参数在第一个span基础上变动
					for end_tidx in range(len(truncated_para_tokens) - 2, start_tidx - 1, -1):
						span_tokens = truncated_para_tokens[start_tidx: end_tidx + 1]
						# f1_score = f1_score(span_tokens, seg_answer_tokens)
						span_len-=1#参数变动_1
						if truncated_para_tokens[end_tidx+1] in seg_answer_tokens:
							num_same-=1#参数变动_2
						if num_same == 0:
							f1_score = 0
						else:
							p = 1.0 * num_same / span_len
							r = 1.0 * num_same / answer_len
							f1_score = (2 * p * r) / (p + r)#调和平均
						# print('针对answer token: ',seg_answer_tokens,'后续span token ', span_tokens,'num same: ',num_same,'p: ',p)
						if f1_score==0:
							break
						if f1_score > best_f1_score_p:
							# print('best_f1_score_p被更新【后续中】',f1_score)
							best_f1_score_p = f1_score
							best_start_idx, best_end_idx = start_tidx, end_tidx
							best_span_tokens = span_tokens
						if f1_score==1.0:#该开始位置，找到了最匹配span，结束位置再继续缩小无意义
							break
				
				#每个篇章下为每个answer选择一个最匹配的fake_span
				if best_f1_score_p > bestK_match[a_idx][-1]:
					bestK_match[a_idx]=(d_idx, p_idx, [best_start_idx, best_end_idx], ''.join(best_span_tokens), best_f1_score_p)#替换

	assert len(bestK_match) == answer_num, 'wrong with best k spans'
	# print(bestK_match)
	# print('\n')
	# sample['multi_spanScore_f1'] = bestK_match

	#用于返回的结果
	rst={}
	rst['question_id']=sample['question_id']
	rst['multi_spanScore_f1'] = bestK_match
	return json.dumps(rst, ensure_ascii=False)

if __name__ == '__main__':
	args=parse_args()
	print(args)

	print('开始计时....')
	start = time.time()

	for train_file in args.train_files:
		i = 0
		work = list()
		with open(train_file,'r') as f_in:
			(filepath, tempfilename) = os.path.split(train_file)
			(train_filename, extension) = os.path.splitext(tempfilename)
			out_filename = train_filename+'_multi_fakespan_opt'+extension
			out_file_path=os.path.join(filepath, out_filename)
			
			with open(out_file_path,'w') as f_out:
				with multiprocess.Pool(args.n_processes) as pool:
					for line in f_in:
						if i < 5000:
							work.append(line)#for trainset
							i += 1
						else:
							pool_res = pool.map(scoreSpan, work)
							f_out.write('\n'.join(pool_res)+'\n')

							work=list()
							work.append(line)
							i = 1
							
					if i > 0:#处理最后一批
						pool_res = pool.map(scoreSpan, work)
						f_out.write('\n'.join(pool_res))
		
		print('done 1...')
		time_elapsed = time.time()-start
		print('Training complete in {:.0f}min-{:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
	
	print('deal with devset all done')

#DEBUG
# sample={}
# sample["documents"]=[]
# doc1={}
# doc1['segmented_paragraphs']=[["手", "账", "，", "指", "用于", "记事", "的", "本子", "。", "干扰1", "干扰2"]]
# sample["documents"].append(doc1)
# doc2={}
# doc2['segmented_paragraphs']=[["就是", "日程簿","，","干扰1", "干扰2"],["日程表", "。","干扰1", "干扰2"]]
# sample["documents"].append(doc2)

# sample["segmented_answers"] = [["记事", "的", "本子", "。"], ["日程", "簿", "，", "日程表", "。"]]

# scoreSpan(sample)
