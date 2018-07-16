import sys
if sys.version[0] == '2':
	reload(sys)
	sys.setdefaultencoding("utf-8")

import os
import json
import re
import numpy as np
from collections import Counter

samples=[]

def load_jsonfiles(filedir):
	# files = os.listdir(filedir)
	# for filename in files:
	# 	with open(os.path.join(filedir,filename), "r") as f:
	# 		samples.append(json.loads(f.readline(), encoding='utf-8'))
	filenames=range(100)#有序
	for filename in filenames:
		with open(os.path.join(filedir,str(filename)+'.json'), "r") as f:
			samples.append(json.loads(f.readline(), encoding='utf-8'))

	print('loading done!')

def analysis(is_search):
	filtered_qids=[]
	if is_search:
		filtered_path='search.filtered.txt'
		with open(filtered_path,'r') as sf:
			filtered_qids=sf.readline().strip().split()
		with open('best_question_match_search.json') as bf:
			best_question_match_record=json.load(bf)
	else:
		with open('best_question_match_zhidao.json') as bf:
			best_question_match_record=json.load(bf)

	ruleA = r'\$\$(.*?)\$\$'
	ruleB = r'\*\*(.*?)\*\*'
	ruleA_pri = r'\%\%\$(.*?)\$\%\%'
	ruleB_pri = r'\%\%\*(.*?)\*\%\%'

	para_prs=[]
	quesMatch_para_prs=[]
	span_prs=[]
	model_scores=[]
	fakespan_modelscore=[]#记录每个样本fakespan得分与model结果得分关系
	for sam_idx,sample in enumerate(samples):
		if len(sample['answers'])==0:
			print('qid: {} is filtered coz no answer'.format(sample['question_id']))
			continue
		if str(sample['question_id']) in filtered_qids:
			print('qid: {} is filtered coz special no answer'.format(sample['question_id']))
			continue
		# paraSelec_records=[]#每个元素记录一个doc的baseline和manual的para选择
		# spanSelec_records=[]#每个元素记录一个doc的baseline和manual的span选择
		assert best_question_match_record[sam_idx][0]==sample['question_id']
		interParaA_nums,interParaB_nums,interParaC_nums, manualParaA_nums, manualParaB_nums, manualParaC_nums = 0, 0, 0, 0, 0 ,0
		interParaA_quesMatch_nums,interParaB_quesMatch_nums,interParaC_quesMatch_nums = 0, 0, 0
		interSpanA_nums, interSpanB_nums, manualSpanA_num, manualSpanB_num = 0, 0, 0, 0
		baselinePara_nums=0# for precision calc
		fakespan_level=0
		for d_idx,doc in enumerate(sample['documents']):
			# paraSelec_records.append({'baseline':[doc['most_related_para']],'manual':doc['fake_paras']})#添加一个doc记录--value为list
			
			## set_manualPara=set(sum(doc['fake_paras'], []))
			set_manualAPara=set(doc['fake_paras'][0])
			set_manualBPara=set(doc['fake_paras'][1])
			set_manualCPara=set(doc['fake_paras'][2])

			manualParaA_nums += len(set_manualAPara)
			manualParaB_nums += len(set_manualBPara)
			manualParaC_nums += len(set_manualCPara)

			set_baselinePara= set([doc['most_related_para']])
			
			interParaA_num=len(set_baselinePara & set_manualAPara)
			interParaB_num=len(set_baselinePara & set_manualBPara)
			interParaC_num=len(set_baselinePara & set_manualCPara)

			interParaA_nums += interParaA_num
			interParaB_nums += interParaB_num
			interParaC_nums += interParaC_num
			
			baselinePara_nums+=1

			##测试集定位most_related_para策略：ques and para based on recall
			set_quesMatchPara= set([best_question_match_record[sam_idx][1][d_idx]])
			
			interParaA_quesMatch_num=len(set_quesMatchPara & set_manualAPara)
			interParaB_quesMatch_num=len(set_quesMatchPara & set_manualBPara)
			interParaC_quesMatch_num=len(set_quesMatchPara & set_manualCPara)

			interParaA_quesMatch_nums += interParaA_quesMatch_num
			interParaB_quesMatch_nums += interParaB_quesMatch_num
			interParaC_quesMatch_nums += interParaC_quesMatch_num

			# a_level_spans={}
			# b_level_spans={}
			para_idx_withManual=[]
			paragraphs=doc['paragraphs']#dict
			for para_idx in paragraphs:
				para_text=paragraphs[para_idx]
				
				spansA=re.findall(ruleA, para_text)
				spansB=re.findall(ruleB, para_text)
				spansA_pri=re.findall(ruleA_pri, para_text)
				spansB_pri=re.findall(ruleB_pri, para_text)
				
				if len(spansA)!=0 or len(spansA_pri)!=0:
					# assert int(para_idx) in doc['fake_paras'][0], 'Assert A: sample idx: {} with q_id: {} doc_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx) #保证一致性
					if not int(para_idx) in doc['fake_paras'][0]: 
						print('Assert A: sample idx: {} with q_id: {} doc_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx)) #保证一致性
					# a_level_spans[para_idx].append([span.replace('%%','') for span in spansA]+spansA_pri)#注意相交去掉字符
					for span in spansA:
						if '%%' in span:
							interSpanA_nums+=1
							fakespan_level=2#相交
							break
					interSpanA_nums+=len(spansA_pri)
					if len(spansA_pri) > 0:
						fakespan_level = 2#强相交
					manualSpanA_num+=sum(map(len,[spansA,spansA_pri]))
					para_idx_withManual.append(int(para_idx))

				if len(spansB)!=0 or len(spansB_pri)!=0:
					if not (is_search and sam_idx==18 and sample['question_id']==182362 and d_idx==2 and int(para_idx)==1):
						# assert int(para_idx) in doc['fake_paras'][0]+doc['fake_paras'][1], 'Assert B: sample idx: {} with q_id: {} doc_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx)
						if not int(para_idx) in doc['fake_paras'][0]+doc['fake_paras'][1]: 
							print('Assert B: sample idx: {} with q_id: {} doc_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx))
						# b_level_spans[para_idx].append([span.replace('%%','') for span in spansB]+spansB_pri)
						for span in spansB:
							if '%%' in span:
								interSpanB_nums+=1
								fakespan_level = 1#相交
								break
						interSpanB_nums+=len(spansB_pri)
						if len(spansB_pri) > 0:
							fakespan_level = 1#强相交
						manualSpanB_num+=sum(map(len,[spansB,spansB_pri]))
						para_idx_withManual.append(int(para_idx))

			# assert len(set(sum(doc['fake_paras'], []))-set(para_idx_withManual))==0, 'span与para标注不一致sample idx: {} with q_id: {} doc_idx:{}'.format(sam_idx, sample['question_id'], d_idx)
			if not (len(set(sum(doc['fake_paras'], []))-set(para_idx_withManual))==0 and len(set(para_idx_withManual)-set(sum(doc['fake_paras'], [])))==0): 
				print('span与para标注不一致sample idx: {} with q_id: {} doc_idx:{}'.format(sam_idx, sample['question_id'], d_idx))
			# spanSelec_records.append({'a_level':a_level_spans,'b_level':b_level_spans})#添加一个doc的记录，只记录人工
		# recall=float(interPara_nums)/manualPara_nums#当前样本的para的recall
		if manualParaA_nums+manualParaB_nums==0:
			print('人工标注相关段落为0',sam_idx,'-----',sample['question_id'])
		para_prs.append(((interParaA_nums, interParaB_nums, interParaC_nums), (manualParaA_nums, manualParaB_nums, manualParaC_nums), baselinePara_nums))#记录相交
		quesMatch_para_prs.append(((interParaA_quesMatch_nums, interParaB_quesMatch_nums, interParaC_quesMatch_nums), (manualParaA_nums, manualParaB_nums, manualParaC_nums), baselinePara_nums))#记录相交
		span_prs.append(((interSpanA_nums, interSpanB_nums),(manualSpanA_num, manualSpanB_num)))
		model_scores.append(sample['model_score'])
		fakespan_modelscore.append((fakespan_level, sample['model_score']))#强相交
	
	total_para_interNum, total_para_totalNum=0,0
	paraRecalls, paraPrecisions=[], []
	for para_prRecords in para_prs:
		para_interNum=sum(para_prRecords[0][:2])
		para_totalNum=sum(para_prRecords[1][:2])
		total_para_interNum+=para_interNum
		total_para_totalNum+=para_totalNum
		paraRecalls.append(float(para_interNum)/para_totalNum)
		paraPrecisions.append(float(para_interNum)/para_prRecords[2])

	micro_final_para_recall=float(total_para_interNum)/total_para_totalNum
	macro_final_para_recall=np.mean(np.array(paraRecalls))
	macro_final_para_precision=np.mean(np.array(paraPrecisions))
	print('Micro-averaging para recall:{:%} '.format(micro_final_para_recall))
	print('Macro-averaging para recall:{:%} '.format(macro_final_para_recall))
	print('Macro-averaging para precison:{:%} '.format(macro_final_para_precision))

	total_span_interNum, total_span_totalNum=0,0
	spanRecalls,spanPrecisions=[], []
	for span_prRecords in span_prs:
		span_interNum=sum(span_prRecords[0])
		span_totalNum=sum(span_prRecords[1])
		total_span_interNum+=span_interNum
		total_span_totalNum+=span_totalNum
		spanRecalls.append(float(span_interNum)/span_totalNum)
		spanPrecisions.append(float(span_interNum)/1)
	micro_final_span_recall=float(total_span_interNum)/total_span_totalNum
	macro_final_span_recall=np.mean(np.array(spanRecalls))
	macro_final_span_precision=np.mean(np.array(spanPrecisions))
	
	print('Micro-averaging span recall:{:%} '.format(micro_final_span_recall))
	print('Macro-averaging span recall:{:%} '.format(macro_final_span_recall))
	print('Macro-averaging span precison:{:%} '.format(macro_final_span_precision))

	print('mean model score:{} '.format(np.mean(np.array(model_scores))*100))
	A2,A1,A0,B2,B1,B0,C2,C1,C0=0,0,0,0,0,0,0,0,0
	for item in fakespan_modelscore:
		if item[0]==2:
			if item[1]==2:
				A2+=1
			elif item[1]==1:
				A1+=1
			elif item[1]==0:
				A0+=1
		elif item[0]==1:
			if item[1]==2:
				B2+=1
			elif item[1]==1:
				B1+=1
			elif item[1]==0:
				B0+=1
		elif item[0]==0:
			if item[1]==2:
				C2+=1
			elif item[1]==1:
				C1+=1
			elif item[1]==0:
				C0+=1
	print('A2:{} ,A1:{} ,A0:{} ,B2:{} ,B1:{} ,B0:{} ,C2:{} ,C1:{} ,C0:{} '.format(A2,A1,A0,B2,B1,B0,C2,C1,C0))

	#select most_related_para: ques and para match based on Recall
	quesMatch_total_para_interNum, total_para_totalNum=0, 0
	quesMatch_paraRecalls,quesMatch_paraPrecisions=[], []
	for para_prRecords in quesMatch_para_prs:
		para_interNum=sum(para_prRecords[0][:2])
		para_totalNum=sum(para_prRecords[1][:2])
		quesMatch_total_para_interNum+=para_interNum
		total_para_totalNum+=para_totalNum

		quesMatch_paraRecalls.append(float(para_interNum)/para_totalNum)
		quesMatch_paraPrecisions.append(float(para_interNum)/para_prRecords[2])

	micro_final_quesMatch_para_recall=float(quesMatch_total_para_interNum)/total_para_totalNum
	macro_final_quesMatch_para_recall=np.mean(np.array(quesMatch_paraRecalls))
	macro_final_quesMatch_para_precision=np.mean(np.array(quesMatch_paraPrecisions))
	print('Micro-averaging para recall:{:%} '.format(micro_final_quesMatch_para_recall))
	print('Macro-averaging para recall:{:%} '.format(macro_final_quesMatch_para_recall))
	print('Macro-averaging para precison:{:%} '.format(macro_final_quesMatch_para_precision))


def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)#调和平均
    return p, r, f1


def recall(prediction, ground_truth):
    """
    This function calculates and returns the recall
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of recall
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[1]

def find_best_question_match(doc, question):
    """
    For each docment, find the paragraph that matches best to the question.--recall&测试集？
    Args:
        doc: The document object.
        question: The question tokens.
            otherwise False.
    Returns:
        The index of the best match paragraph
    """
    most_related_para = -1
    max_related_score = 0
    most_related_para_len = 0
    for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
        if len(question) > 0:
            related_score = recall(para_tokens, question)
        else:
            related_score = 0

        if related_score > max_related_score \
                or (related_score == max_related_score \
                and len(para_tokens) < most_related_para_len):
            most_related_para = p_idx
            max_related_score = related_score
            most_related_para_len = len(para_tokens)
    if most_related_para == -1:
        most_related_para = 0
    return most_related_para

def best_question_match(filepath,samplepath,is_search):
	import linecache
	sampled_qids = linecache.getline(samplepath, 2).strip().split()
	most_related_para_record=[]
	with open(filepath,'r') as fr:
		for line in fr:
			sample = json.loads(line)
			if str(sample['question_id']) in sampled_qids:
				most_related_idxs=[]
				for d_idx, doc in enumerate(sample['documents']):
					most_related_idx=find_best_question_match(doc, sample['segmented_question'])
					most_related_idxs.append(most_related_idx)
				most_related_para_record.append((sample['question_id'],most_related_idxs))
        #save
		if is_search:
			json.dump(most_related_para_record, open('best_question_match_search.json', "w"))
		else:
			json.dump(most_related_para_record, open('best_question_match_zhidao.json', "w"))
		print('best_question_match_record calculated done!')


if __name__ == "__main__":
	#问题和答案通过recall计算most_related_para
	# original_filedir='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset'
	
	# original_searchpath=os.path.join(original_filedir,'search.dev.json')
	# samplepath='100sample/100.search.dev.txt'
	# best_question_match(original_searchpath,samplepath,True)

	# original_zhidaopath=os.path.join(original_filedir,'zhidao.dev.json')
	# samplepath='100sample/100.zhidao.dev.txt'
	# best_question_match(original_zhidaopath,samplepath,False)



	print('start deal with search...')
	filedir='search'
	load_jsonfiles(filedir)
	analysis(True)

	print('start deal with zhidao...')
	samples=[]
	filedir='zhidao'
	load_jsonfiles(filedir)
	analysis(False)