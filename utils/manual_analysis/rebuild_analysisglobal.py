import sys
if sys.version[0] == '2':
	reload(sys)
	sys.setdefaultencoding("utf-8")

import os
import json
import re
import numpy as np
from collections import Counter
# import Levenshtein
import jieba

def load_manual_labeled_jsonfiles(jsonfiledir):
	manual_labeled_samples=[]
	filenames=range(100)#有序
	for filename in filenames:
		with open(os.path.join(jsonfiledir,str(filename)+'.json'), "r") as f:
			manual_labeled_samples.append(json.loads(f.readline(), encoding='utf-8'))
	print('loading manual labeled json files done!')
	return manual_labeled_samples

def load_global_a_json(jsonpath, samplepath, topK=None):
	import linecache
	sampled_qids = linecache.getline(samplepath, 2).strip().split()

	global_a_samples=[]
	with open(jsonpath,'r') as fr:
		for line in fr:
			sample = json.loads(line)
			if str(sample['question_id']) in sampled_qids:
				paragScoreRecords=[]			   
				# for k,v in sample['paragScore_recall_a'].items():
				for k,v in sample['paragScore_tfidf_q'].items():
					for item in v:
						paragScoreRecords.append((k,item[0],item[1]))
				sortedparagScoreRecords=sorted(paragScoreRecords, key=lambda record: record[-1],reverse=True)
				if topK==None:
					topK=len(sample['documents'])
				topparagScoreRecords={}
				for paragScoreRecord in sortedparagScoreRecords[:topK]:#取前x个
					if paragScoreRecord[0] not in topparagScoreRecords:#doc_id
						topparagScoreRecords[paragScoreRecord[0]]=[]
					topparagScoreRecords[paragScoreRecord[0]].append(paragScoreRecord[1])

				spanScoreRecord=sample['spanScore_f1'][0]

				global_a_samples.append((sample['question_id'], topparagScoreRecords,spanScoreRecord))
	return global_a_samples

def load_global_q_json(jsonpath, samplepath, topK=None):
	import linecache
	sampled_qids = linecache.getline(samplepath, 2).strip().split()

	global_q_samples=[]
	with open(jsonpath,'r') as fr:
		for line in fr:
			sample = json.loads(line)
			if str(sample['question_id']) in sampled_qids:
				paragScoreRecords=[]			   
				# for k,v in sample['paragScore_recall_q'].items():
				for k,v in sample['paragScore_tfidf_q'].items():
					for item in v:
						paragScoreRecords.append((k,item[0],item[1]))
				sortedparagScoreRecords=sorted(paragScoreRecords, key=lambda record: record[-1],reverse=True)
				if topK==None:
					topK=len(sample['documents'])
				topparagScoreRecords={}
				for paragScoreRecord in sortedparagScoreRecords[:topK]:#取前x个
					if paragScoreRecord[0] not in topparagScoreRecords:#doc_id
						topparagScoreRecords[paragScoreRecord[0]]=[]
					topparagScoreRecords[paragScoreRecord[0]].append(paragScoreRecord[1])

				global_q_samples.append((sample['question_id'], topparagScoreRecords))
	return global_q_samples

def load_multi_a_json(jsonpath, samplepath, topK=None):
	import linecache
	sampled_qids = linecache.getline(samplepath, 2).strip().split()

	multi_a_samples=[]
	with open(jsonpath,'r') as fr:
		for line in fr:
			sample = json.loads(line)
			if str(sample['question_id']) in sampled_qids:
				multispanScoreRecord=sample['multi_spanScore_f1']
				multi_a_samples.append((sample['question_id'], multispanScoreRecord))
	return multi_a_samples

def analysis_vsBaseline(manualjsonfiledir, global_a_jsonpath, global_q_jsonpath, samplepath, is_search):
	manual_labeled_samples = load_manual_labeled_jsonfiles(manualjsonfiledir)
	global_a_samples = load_global_a_json(global_a_jsonpath, samplepath)
	global_q_samples = load_global_q_json(global_q_jsonpath, samplepath)

	filtered_qids=[]
	if is_search:
		filtered_path='search.filtered.txt'
		with open(filtered_path,'r') as sf:
			filtered_qids=sf.readline().strip().split()

	ruleA = r'\$\$(.*?)\$\$'
	ruleB = r'\*\*(.*?)\*\*'
	ruleA_pri = r'\%\%\$(.*?)\$\%\%'
	ruleB_pri = r'\%\%\*(.*?)\*\%\%'

	globalA_para_prs=[]
	quesMatch_para_prs=[]
	globalA_span_prs=[]
	model_scores=[]
	fakespan_modelscore=[]#记录每个样本fakespan得分与model结果得分关系
	#开始遍历
	for sam_idx,sample in enumerate(manual_labeled_samples):
		if len(sample['answers'])==0:
			print('qid: {} is filtered coz no answer'.format(sample['question_id']))
			continue
		if str(sample['question_id']) in filtered_qids:
			print('qid: {} is filtered coz special no answer'.format(sample['question_id']))
			continue

		assert global_a_samples[sam_idx][0]==sample['question_id']
		assert global_q_samples[sam_idx][0]==sample['question_id']
		
		interParaA_globalA_nums,interParaB_globalA_nums,interParaC_globalA_nums, manualParaA_nums, manualParaB_nums, manualParaC_nums = 0, 0, 0, 0, 0 ,0
		interParaA_quesMatch_nums,interParaB_quesMatch_nums,interParaC_quesMatch_nums = 0, 0, 0
		interSpanA_nums, interSpanB_nums, manualSpanA_num, manualSpanB_num = 0, 0, 0, 0
		
		globalAPara_nums=len(sample['documents'])# for precision calc
		fakespan_level=0
		
		for d_idx,doc in enumerate(sample['documents']):
			## set_manualPara=set(sum(doc['fake_paras'], []))
			set_manualAPara=set(doc['fake_paras'][0])
			set_manualBPara=set(doc['fake_paras'][1])
			set_manualCPara=set(doc['fake_paras'][2])

			manualParaA_nums += len(set_manualAPara)
			manualParaB_nums += len(set_manualBPara)
			manualParaC_nums += len(set_manualCPara)

			if str(d_idx) in global_a_samples[sam_idx][1]:
				set_globalAPara= set(global_a_samples[sam_idx][1][str(d_idx)])
			else:
				set_globalAPara=set([])
			
			interParaA_globalA_num=len(set_globalAPara & set_manualAPara)
			interParaB_globalA_num=len(set_globalAPara & set_manualBPara)
			interParaC_globalA_num=len(set_globalAPara & set_manualCPara)

			interParaA_globalA_nums += interParaA_globalA_num
			interParaB_globalA_nums += interParaB_globalA_num
			interParaC_globalA_nums += interParaC_globalA_num
			
			##测试集定位most_related_para策略：ques and para based on recall
			if str(d_idx) in global_q_samples[sam_idx][1]:
				set_globalQPara = set(global_q_samples[sam_idx][1][str(d_idx)])
			else:
				set_globalQPara = set([])

			interParaA_quesMatch_num=len(set_globalQPara & set_manualAPara)
			interParaB_quesMatch_num=len(set_globalQPara & set_manualBPara)
			interParaC_quesMatch_num=len(set_globalQPara & set_manualCPara)

			interParaA_quesMatch_nums += interParaA_quesMatch_num
			interParaB_quesMatch_nums += interParaB_quesMatch_num
			interParaC_quesMatch_nums += interParaC_quesMatch_num

			# print('global A labeled answer span: d_idx and para_idx',global_a_samples[sam_idx][2][0],global_a_samples[sam_idx][2][1])
			para_idx_withManual=[]
			paragraphs=doc['paragraphs']#dict
			for para_idx in paragraphs:
				para_text=paragraphs[para_idx]
				if d_idx == int(global_a_samples[sam_idx][2][0]) and int(para_idx) == int(global_a_samples[sam_idx][2][1]):
					span_globalA=global_a_samples[sam_idx][2][3]
				
				spansA=re.findall(ruleA, para_text)
				spansB=re.findall(ruleB, para_text)
				spansA_pri=re.findall(ruleA_pri, para_text)
				spansB_pri=re.findall(ruleB_pri, para_text)
				if len(spansA)!=0 or len(spansA_pri)!=0:
					# assert int(para_idx) in doc['fake_paras'][0], 'Assert A: sample idx: {} with q_id: {} d_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx) #保证一致性
					if not int(para_idx) in doc['fake_paras'][0]: 
						print('Assert A: sample idx: {} with q_id: {} d_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx)) #保证一致性
					if span_globalA:
						for span in spansA:
							span=span.replace('%%','')
							if span in span_globalA :
								# print(span,'====',span_globalA)
								interSpanA_nums+=1
								fakespan_level=2#相交
								break
						for span in spansA_pri:
							if span in span_globalA :
								interSpanA_nums+=1
								fakespan_level=2#相交
								break
					manualSpanA_num+=sum(map(len,[spansA,spansA_pri]))
					para_idx_withManual.append(int(para_idx))

				if len(spansB)!=0 or len(spansB_pri)!=0:
					if not (is_search and sam_idx==18 and sample['question_id']==182362 and d_idx==2 and int(para_idx)==1):
						# assert int(para_idx) in doc['fake_paras'][0]+doc['fake_paras'][1], 'Assert B: sample idx: {} with q_id: {} d_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx)
						if not int(para_idx) in doc['fake_paras'][0]+doc['fake_paras'][1]: 
							print('Assert B: sample idx: {} with q_id: {} d_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx))
						if span_globalA:
							for span in spansB:
								span=span.replace('%%','')
								if span in span_globalA :
									interSpanB_nums+=1
									fakespan_level=1#相交
									break
							for span in spansB_pri:
								if span in span_globalA :
									interSpanB_nums+=1
									fakespan_level=1#相交
									break
						manualSpanB_num+=sum(map(len,[spansB,spansB_pri]))
						para_idx_withManual.append(int(para_idx))

			# assert len(set(sum(doc['fake_paras'], []))-set(para_idx_withManual))==0, 'span与para标注不一致sample idx: {} with q_id: {} d_idx:{}'.format(sam_idx, sample['question_id'], d_idx)
			if not (len(set(sum(doc['fake_paras'], []))-set(para_idx_withManual))==0 and len(set(para_idx_withManual)-set(sum(doc['fake_paras'], [])))==0): 
				print('span与para标注不一致sample idx: {} with q_id: {} d_idx:{}'.format(sam_idx, sample['question_id'], d_idx))
		if manualParaA_nums+manualParaB_nums==0:
			print('人工标注相关段落为0: ',sam_idx,'-----',sample['question_id'])
		globalA_para_prs.append(((interParaA_globalA_nums, interParaB_globalA_nums, interParaC_globalA_nums), (manualParaA_nums, manualParaB_nums, manualParaC_nums), globalAPara_nums))#记录相交
		quesMatch_para_prs.append(((interParaA_quesMatch_nums, interParaB_quesMatch_nums, interParaC_quesMatch_nums), (manualParaA_nums, manualParaB_nums, manualParaC_nums), globalAPara_nums))#记录相交
		globalA_span_prs.append(((interSpanA_nums, interSpanB_nums),(manualSpanA_num, manualSpanB_num)))
		model_scores.append(sample['model_score'])
		fakespan_modelscore.append((fakespan_level, sample['model_score']))#强相交
	
	total_para_interNum, total_para_totalNum=0,0
	paraRecalls, paraPrecisions=[], []
	for para_prRecords in globalA_para_prs:
		para_interNum=sum(para_prRecords[0][:2])
		para_totalNum=sum(para_prRecords[1][:2])
		total_para_interNum+=para_interNum
		total_para_totalNum+=para_totalNum
		paraRecalls.append(float(para_interNum)/para_totalNum)
		paraPrecisions.append(float(para_interNum)/para_prRecords[2])

	micro_final_para_recall=float(total_para_interNum)/total_para_totalNum
	macro_final_para_recall=np.mean(np.array(paraRecalls))
	macro_final_para_precision=np.mean(np.array(paraPrecisions))
	print('\nrecall and precision results of para selection based on global a')
	print('Micro-averaging para recall:{:%} '.format(micro_final_para_recall))
	print('Macro-averaging para recall:{:%} '.format(macro_final_para_recall))
	print('Macro-averaging para precison:{:%} \n'.format(macro_final_para_precision))

	total_span_interNum, total_span_totalNum=0,0
	spanRecalls,spanPrecisions=[], []
	for span_prRecords in globalA_span_prs:
		span_interNum=sum(span_prRecords[0])
		span_totalNum=sum(span_prRecords[1])
		total_span_interNum+=span_interNum
		total_span_totalNum+=span_totalNum
		spanRecalls.append(float(span_interNum)/span_totalNum)
		spanPrecisions.append(float(span_interNum)/1)
	micro_final_span_recall=float(total_span_interNum)/total_span_totalNum
	macro_final_span_recall=np.mean(np.array(spanRecalls))
	macro_final_span_precision=np.mean(np.array(spanPrecisions))
	print('recall and precision results of span selection based on global a')
	print('Micro-averaging span recall:{:%} '.format(micro_final_span_recall))
	print('Macro-averaging span recall:{:%} '.format(macro_final_span_recall))
	print('Macro-averaging span precison:{:%} \n'.format(macro_final_span_precision))

	# print('mean model score:{} '.format(np.mean(np.array(model_scores))*100))
	# A2,A1,A0,B2,B1,B0,C2,C1,C0=0,0,0,0,0,0,0,0,0
	# for item in fakespan_modelscore:
	# 	if item[0]==2:
	# 		if item[1]==2:
	# 			A2+=1
	# 		elif item[1]==1:
	# 			A1+=1
	# 		elif item[1]==0:
	# 			A0+=1
	# 	elif item[0]==1:
	# 		if item[1]==2:
	# 			B2+=1
	# 		elif item[1]==1:
	# 			B1+=1
	# 		elif item[1]==0:
	# 			B0+=1
	# 	elif item[0]==0:
	# 		if item[1]==2:
	# 			C2+=1
	# 		elif item[1]==1:
	# 			C1+=1
	# 		elif item[1]==0:
	# 			C0+=1
	# print('A2:{} ,A1:{} ,A0:{} ,B2:{} ,B1:{} ,B0:{} ,C2:{} ,C1:{} ,C0:{} \n'.format(A2,A1,A0,B2,B1,B0,C2,C1,C0))

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
	print('recall and precision results of para selection based on global q')
	print('Micro-averaging para recall:{:%} '.format(micro_final_quesMatch_para_recall))
	print('Macro-averaging para recall:{:%} '.format(macro_final_quesMatch_para_recall))
	print('Macro-averaging para precison:{:%} '.format(macro_final_quesMatch_para_precision))


def analysis_TOP5(manualjsonfiledir, global_a_jsonpath, global_q_jsonpath, multi_a_jsonpath, samplepath, is_search):
	manual_labeled_samples = load_manual_labeled_jsonfiles(manualjsonfiledir)
	global_a_samples = load_global_a_json(global_a_jsonpath, samplepath, 5)
	global_q_samples = load_global_q_json(global_q_jsonpath, samplepath, 5)
	# multi_a_samples = load_multi_a_json(multi_a_jsonpath, samplepath)

	filtered_qids=[]
	if is_search:
		filtered_path='search.filtered.txt'
		with open(filtered_path,'r') as sf:
			filtered_qids=sf.readline().strip().split()

	ruleA = r'\$\$(.*?)\$\$'
	ruleB = r'\*\*(.*?)\*\*'
	ruleA_pri = r'\%\%\$(.*?)\$\%\%'
	ruleB_pri = r'\%\%\*(.*?)\*\%\%'

	globalA_para_prs=[]
	quesMatch_para_prs=[]
	globalA_span_prs=[]
	model_scores=[]
	fakespan_modelscore=[]#记录每个样本fakespan得分与model结果得分关系
	#开始遍历
	for sam_idx,sample in enumerate(manual_labeled_samples):
		if len(sample['answers'])==0:
			print('qid: {} is filtered coz no answer'.format(sample['question_id']))
			continue
		if str(sample['question_id']) in filtered_qids:
			print('qid: {} is filtered coz special no answer'.format(sample['question_id']))
			continue

		assert global_a_samples[sam_idx][0]==sample['question_id']
		assert global_q_samples[sam_idx][0]==sample['question_id']
		# assert multi_a_samples[sam_idx][0]==sample['question_id']
		
		interParaA_globalA_nums,interParaB_globalA_nums,interParaC_globalA_nums, manualParaA_nums, manualParaB_nums, manualParaC_nums = 0, 0, 0, 0, 0 ,0
		interParaA_quesMatch_nums,interParaB_quesMatch_nums,interParaC_quesMatch_nums = 0, 0, 0
		interSpanA_nums, interSpanB_nums, manualSpanA_num, manualSpanB_num = 0, 0, 0, 0
		
		globalAPara_nums, globalQPara_nums = 0, 0
		spanSelected_nums = 0
		fakespan_level = 0
		

		for d_idx,doc in enumerate(sample['documents']):
			## set_manualPara=set(sum(doc['fake_paras'], []))
			set_manualAPara=set(doc['fake_paras'][0])
			set_manualBPara=set(doc['fake_paras'][1])
			set_manualCPara=set(doc['fake_paras'][2])

			manualParaA_nums += len(set_manualAPara)
			manualParaB_nums += len(set_manualBPara)
			manualParaC_nums += len(set_manualCPara)

			if str(d_idx) in global_a_samples[sam_idx][1]:
				set_globalAPara= set(global_a_samples[sam_idx][1][str(d_idx)])
				# beforelen=len(set_globalAPara)
				# print(type(global_a_samples[sam_idx][1][str(d_idx)][0]),'is int')
				if d_idx == int(global_a_samples[sam_idx][2][0]):#fake answer span所在的para也算入内
					set_globalAPara.add(int(global_a_samples[sam_idx][2][1]))
					# if beforelen!=len(set_globalAPara):
						# print('---------------after len',len(set_globalAPara))
			else:
				set_globalAPara = set([])

			globalAPara_nums += len(set_globalAPara)
			
			interParaA_globalA_num=len(set_globalAPara & set_manualAPara)
			interParaB_globalA_num=len(set_globalAPara & set_manualBPara)
			interParaC_globalA_num=len(set_globalAPara & set_manualCPara)

			interParaA_globalA_nums += interParaA_globalA_num
			interParaB_globalA_nums += interParaB_globalA_num
			interParaC_globalA_nums += interParaC_globalA_num
			
			##测试集定位most_related_para策略：ques and para based on recall
			if str(d_idx) in global_q_samples[sam_idx][1]:
				set_globalQPara = set(global_q_samples[sam_idx][1][str(d_idx)])
			else:
				set_globalQPara = set([])
			
			globalQPara_nums+=len(set_globalQPara)

			interParaA_quesMatch_num=len(set_globalQPara & set_manualAPara)
			interParaB_quesMatch_num=len(set_globalQPara & set_manualBPara)
			interParaC_quesMatch_num=len(set_globalQPara & set_manualCPara)

			interParaA_quesMatch_nums += interParaA_quesMatch_num
			interParaB_quesMatch_nums += interParaB_quesMatch_num
			interParaC_quesMatch_nums += interParaC_quesMatch_num

			# print('global A labeled answer span: d_idx and para_idx',global_a_samples[sam_idx][2][0],global_a_samples[sam_idx][2][1])
			para_idx_withManual=[]
			paragraphs=doc['paragraphs']#dict
			for para_idx in paragraphs:
				inter_flag=False
				para_text=paragraphs[para_idx]
				
				span_globalA = None
				if d_idx == int(global_a_samples[sam_idx][2][0]) and int(para_idx) == int(global_a_samples[sam_idx][2][1]):
					span_globalA=global_a_samples[sam_idx][2][3]
					spanSelected_nums += 1
				
				# multi_fakespans=[]
				# for multi_a_score_record in multi_a_samples[sam_idx][1]:
				# 	multi_d_idx, multi_p_idx = int(multi_a_score_record[0]), int(multi_a_score_record[1])
				# 	if d_idx == multi_d_idx and int(para_idx) == multi_p_idx and multi_a_score_record[-1]>0.8:#TODO设定阈值
				# 		multi_fakespans.append(multi_a_score_record[3])
				# spanSelected_nums+=len(multi_fakespans)

				spansA=re.findall(ruleA, para_text)
				spansB=re.findall(ruleB, para_text)
				spansA_pri=re.findall(ruleA_pri, para_text)
				spansB_pri=re.findall(ruleB_pri, para_text)
				
				if len(spansA)!=0 or len(spansA_pri)!=0:
					# assert int(para_idx) in doc['fake_paras'][0], 'Assert A: sample idx: {} with q_id: {} d_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx) #保证一致性
					if not int(para_idx) in doc['fake_paras'][0]: 
						print('Assert A: sample idx: {} with q_id: {} d_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx)) #保证一致性
					if span_globalA:
						for span in spansA:
							span=span.replace('%%','')
							if span in span_globalA or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(span_globalA)))[1]==1.0 or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(span_globalA)))[2]>=0.9:
								# print(span,'====',span_globalA)
								inter_flag=True
								interSpanA_nums+=1
								fakespan_level=2#相交
								break
						for span in spansA_pri:
							if span in span_globalA or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(span_globalA)))[1]==1.0 or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(span_globalA)))[2]>=0.9:
								inter_flag=True
								interSpanA_nums+=1
								fakespan_level=2#相交
								break

					# if len(multi_fakespans):
					# 	for span in spansA:
					# 		span=span.replace('%%','')
					# 		for multi_fakespan in multi_fakespans:
					# 			if span in multi_fakespan or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(multi_fakespan)))[1]==1.0 or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(multi_fakespan)))[2]>=0.9:
					# 				# print(span,'====',multi_fakespan)
					# 				inter_flag=True
					# 				interSpanA_nums+=1
					# 				break
					# 	for span in spansA_pri:
					# 		for multi_fakespan in multi_fakespans:
					# 			if span in multi_fakespan or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(multi_fakespan)))[1]==1.0 or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(multi_fakespan)))[2]>=0.9:
					# 				inter_flag=True
					# 				interSpanA_nums+=1
					# 				break

					manualSpanA_num+=sum(map(len,[spansA,spansA_pri]))
					para_idx_withManual.append(int(para_idx))

				if len(spansB)!=0 or len(spansB_pri)!=0:
					if not (is_search and sam_idx==18 and sample['question_id']==182362 and d_idx==2 and int(para_idx)==1):
						# assert int(para_idx) in doc['fake_paras'][0]+doc['fake_paras'][1], 'Assert B: sample idx: {} with q_id: {} d_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx)
						if not int(para_idx) in doc['fake_paras'][0]+doc['fake_paras'][1]: 
							print('Assert B: sample idx: {} with q_id: {} d_idx:{} para_idx:{}'.format(sam_idx,sample['question_id'],d_idx, para_idx))
						if span_globalA:
							for span in spansB:
								span=span.replace('%%','')
								if span in span_globalA or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(span_globalA)))[1]==1.0 or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(span_globalA)))[2]>=0.9:
									inter_flag=True
									interSpanB_nums+=1
									fakespan_level=1#相交
									break
							for span in spansB_pri:
								if span in span_globalA or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(span_globalA)))[1]==1.0 or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(span_globalA)))[2]>=0.9:
									inter_flag=True
									interSpanB_nums+=1
									fakespan_level=1#相交
									break
						
						# if len(multi_fakespans):
						# 	for span in spansB:
						# 		span=span.replace('%%','')
						# 		for multi_fakespan in multi_fakespans:
						# 			if span in multi_fakespan or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(multi_fakespan)))[1]==1.0 or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(multi_fakespan)))[2]>=0.9:
						# 				# print(span,'====',multi_fakespan)
						# 				inter_flag=True
						# 				interSpanB_nums+=1
						# 				break
						# 	for span in spansB_pri:
						# 		for multi_fakespan in multi_fakespans:
						# 			if span in multi_fakespan or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(multi_fakespan)))[1]==1.0 or precision_recall_f1(list(jieba.cut(span)), list(jieba.cut(multi_fakespan)))[2]>=0.9:
						# 				inter_flag=True
						# 				interSpanB_nums+=1
						# 				break
						manualSpanB_num+=sum(map(len,[spansB,spansB_pri]))
						para_idx_withManual.append(int(para_idx))
				
				#DEBUG输出
				# if d_idx == int(global_a_samples[sam_idx][2][0]) and int(para_idx) == int(global_a_samples[sam_idx][2][1]) and inter_flag==False:
				# 	print('\nspansA',spansA)
				# 	print('spansB',spansB)
				# 	print('spansA_pri',spansA_pri)
				# 	print('spansB_pri',spansB_pri)
				# 	print('span_globalA',span_globalA)
				# 	print('qid',sample['question_id'])
			# assert len(set(sum(doc['fake_paras'], []))-set(para_idx_withManual))==0, 'span与para标注不一致sample idx: {} with q_id: {} d_idx:{}'.format(sam_idx, sample['question_id'], d_idx)
			if not (len(set(sum(doc['fake_paras'], []))-set(para_idx_withManual))==0 and len(set(para_idx_withManual)-set(sum(doc['fake_paras'], [])))==0): 
				print('span与para标注不一致sample idx: {} with q_id: {} d_idx:{}'.format(sam_idx, sample['question_id'], d_idx))

		if manualParaA_nums+manualParaB_nums==0:
			print('人工标注相关段落为0: ',sam_idx,'-----',sample['question_id'])

		globalA_para_prs.append(((interParaA_globalA_nums, interParaB_globalA_nums, interParaC_globalA_nums), (manualParaA_nums, manualParaB_nums, manualParaC_nums), globalAPara_nums))#记录相交
		quesMatch_para_prs.append(((interParaA_quesMatch_nums, interParaB_quesMatch_nums, interParaC_quesMatch_nums), (manualParaA_nums, manualParaB_nums, manualParaC_nums), globalQPara_nums))#记录相交
		globalA_span_prs.append(((interSpanA_nums, interSpanB_nums),(manualSpanA_num, manualSpanB_num), spanSelected_nums))
		model_scores.append(sample['model_score'])
		fakespan_modelscore.append((fakespan_level, sample['model_score']))#强相交
	
	total_para_interNum, total_para_totalNum=0,0
	paraRecalls, paraPrecisions=[], []
	for para_prRecords in globalA_para_prs:
		if para_prRecords[2]!=0:
			para_interNum=sum(para_prRecords[0][:2])
			para_totalNum=sum(para_prRecords[1][:2])
			total_para_interNum+=para_interNum
			total_para_totalNum+=para_totalNum
			paraRecalls.append(float(para_interNum)/para_totalNum)
			paraPrecisions.append(float(para_interNum)/para_prRecords[2])

	print('len(paraRecalls)', len(paraRecalls))
	micro_final_para_recall=float(total_para_interNum)/total_para_totalNum
	macro_final_para_recall=np.mean(np.array(paraRecalls))
	macro_final_para_precision=np.mean(np.array(paraPrecisions))
	print('\nrecall and precision results of para selection based on global a')
	print('Micro-averaging para recall:{:%} '.format(micro_final_para_recall))
	print('Macro-averaging para recall:{:%} '.format(macro_final_para_recall))
	print('Macro-averaging para precison:{:%} \n'.format(macro_final_para_precision))

	total_span_interNum, total_span_totalNum=0,0
	spanRecalls,spanPrecisions=[], []
	for span_prRecords in globalA_span_prs:
		if span_prRecords[2]!=0:
			span_interNum=sum(span_prRecords[0])
			span_totalNum=sum(span_prRecords[1])
			total_span_interNum+=span_interNum
			total_span_totalNum+=span_totalNum
			spanRecalls.append(float(span_interNum)/span_totalNum)
			spanPrecisions.append(float(span_interNum)/span_prRecords[2])

	print('len(spanRecalls)', len(spanRecalls))
	micro_final_span_recall=float(total_span_interNum)/total_span_totalNum
	macro_final_span_recall=np.mean(np.array(spanRecalls))
	macro_final_span_precision=np.mean(np.array(spanPrecisions))
	print('recall and precision results of span selection based on global a')
	print('Micro-averaging span recall:{:%} '.format(micro_final_span_recall))
	print('Macro-averaging span recall:{:%} '.format(macro_final_span_recall))
	print('Macro-averaging span precison:{:%} \n'.format(macro_final_span_precision))

	# print('mean model score:{} '.format(np.mean(np.array(model_scores))*100))
	# A2,A1,A0,B2,B1,B0,C2,C1,C0=0,0,0,0,0,0,0,0,0
	# for item in fakespan_modelscore:
	# 	if item[0]==2:
	# 		if item[1]==2:
	# 			A2+=1
	# 		elif item[1]==1:
	# 			A1+=1
	# 		elif item[1]==0:
	# 			A0+=1
	# 	elif item[0]==1:
	# 		if item[1]==2:
	# 			B2+=1
	# 		elif item[1]==1:
	# 			B1+=1
	# 		elif item[1]==0:
	# 			B0+=1
	# 	elif item[0]==0:
	# 		if item[1]==2:
	# 			C2+=1
	# 		elif item[1]==1:
	# 			C1+=1
	# 		elif item[1]==0:
	# 			C0+=1
	# print('A2:{} ,A1:{} ,A0:{} ,B2:{} ,B1:{} ,B0:{} ,C2:{} ,C1:{} ,C0:{} \n'.format(A2,A1,A0,B2,B1,B0,C2,C1,C0))

	#select most_related_para: ques and para match based on Recall
	quesMatch_total_para_interNum, total_para_totalNum=0, 0
	quesMatch_paraRecalls,quesMatch_paraPrecisions=[], []
	for para_prRecords in quesMatch_para_prs:
		if para_prRecords[2]!=0:
			para_interNum=sum(para_prRecords[0][:2])
			para_totalNum=sum(para_prRecords[1][:2])
			quesMatch_total_para_interNum+=para_interNum
			total_para_totalNum+=para_totalNum

			quesMatch_paraRecalls.append(float(para_interNum)/para_totalNum)
			quesMatch_paraPrecisions.append(float(para_interNum)/para_prRecords[2])
	print('len(quesMatch_paraRecalls)', len(quesMatch_paraRecalls))
	micro_final_quesMatch_para_recall=float(quesMatch_total_para_interNum)/total_para_totalNum
	macro_final_quesMatch_para_recall=np.mean(np.array(quesMatch_paraRecalls))
	macro_final_quesMatch_para_precision=np.mean(np.array(quesMatch_paraPrecisions))
	print('recall and precision results of para selection based on global q')
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

if __name__ == "__main__":

	print('start deal with search...')
	manualjsonfiledir='search'
	
	# global_a_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/search.dev_recall_f1.json'
	# global_q_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/search.dev_recall.json'
	
	# global_a_removeStop_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/search.dev_recall_f1_100sample.json'
	# global_q_removeStop_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/search.dev_recall_100sample.json'
	
	global_a_tfidf_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/search.dev_tfidf_f1_100sample.json'
	global_q_tfidf_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/search.dev_tfidf_100sample.json'

	multi_a_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/search.dev_multi_fakespan.json'
	samplepath='100sample/100.search.dev.txt'
	is_search=True
	

	# print('start deal with zhidao...')
	# manualjsonfiledir='zhidao'
	# # global_a_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/zhidao.dev_recall_f1.json'
	# # global_q_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/zhidao.dev_recall.json'
	
	# # global_a_removeStop_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/zhidao.dev_recall_f1_100sample.json'
	# # global_q_removeStop_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/zhidao.dev_recall_100sample.json'
	
	# global_a_tfidf_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/zhidao.dev_tfidf_f1_100sample.json'
	# global_q_tfidf_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/zhidao.dev_tfidf_100sample.json'
	
	# multi_a_jsonpath='/home/yhli/DuReader/DuReader-master_v1/data/preprocessed/devset/zhidao.dev_multi_fakespan.json'
	# samplepath='100sample/100.zhidao.dev.txt'
	# is_search=False
	
	# analysis_vsBaseline(manualjsonfiledir, global_a_jsonpath, global_q_jsonpath, samplepath, is_search)
	analysis_TOP5(manualjsonfiledir, global_a_tfidf_jsonpath, global_q_tfidf_jsonpath, multi_a_jsonpath, samplepath, is_search)