# -*- coding: UTF-8 -*-
import sys
if sys.version[0] == '2':
	reload(sys)
	sys.setdefaultencoding("utf-8")

from collections import Counter
import json
import numpy as np
from sklearn.metrics import pairwise_distances
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords
from sklearn.externals import joblib

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

###########################################################################

def scoreParag_ml(sample, tfidfObj):
	#6个特征：f1,recall,is_first_para,tfidf,bleu,overlap

	#保存该样本形成的特征数据各行对应的d_idx和p_idx
	qid2featLidx={'qid':None,'lIdx2dIdx2pIdx':{}}#{qid:xx,lIdx2dIdx2pIdx:{lIdx1:(d_idx,p_idx)，lIdx2:(d_idx,p_idx)，...}
	qid2featLidx['qid']=sample['question_id']
	
	feature_datas=[]
	
	question=sample['segmented_question']
	stop_words=stopwords.words('chinese')
	q_words = {x for x in question if x not in stop_words}
	l_idx=0#featLidx
	for d_idx, doc in enumerate(sample['documents']):
		for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
			#feature: overlap
			found = set()
			for word in para_tokens:
				if word in q_words:
					found.add(word)
			word_match_feature=len(found)
			
			#feature: tfidf-cosine similarity
			try:
				para_features = tfidfObj.transform([' '.join(para_tokens)])
				q_features = tfidfObj.transform([' '.join(question)])
			except ValueError:
				pass
			tfidf_score = pairwise_distances(q_features, para_features, "cosine").ravel()[0]

			#feature: recall & f1
			recall_score, f1_score = precision_recall_f1(para_tokens,question)[1:]
			
			#feature: bleu
			bleu_score = sentence_bleu(question, para_tokens)
			
			#feature: first_para
			is_first_para= float(p_idx==0)

			#TODO---word_match_feature未做归一化...
			#slot:<target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
			# feature_line = str(l_idx)+' qid:'+str(sample['question_id'])+' 1:'+str(f1_score)+' 2:'+str(recall_score)+' 3:'+str(is_first_para)+' 4:'+str(tfidf_score)+' 5:'+str(bleu_score)+' 6:'+str(word_match_feature)

			feature_data=[l_idx, sample['question_id'], f1_score, recall_score, is_first_para, tfidf_score, bleu_score, word_match_feature]
			feature_datas.append(feature_data)
			
			
			qid2featLidx['lIdx2dIdx2pIdx'][l_idx]=(d_idx,p_idx)
			l_idx+=1
	return qid2featLidx, feature_datas

def lr_predict_forTOP5(feature_path, qid2featLidx_path, model_dir, out_path):

	with open(feature_path, "r") as fr1:
		with open(qid2featLidx_path, "r") as fr2:
			with open(out_path, "w") as fw:
				for line in fr1:#每行是一个样本的数据
					feature_list = json.loads(line)
					qid2featLidx = json.loads(fr2.readline())
					if len(feature_list)==0:
						print('qid2featLidx[\'qid\']',qid2featLidx['qid'])
						continue
						#没有篇章？？
						test1_search=[401621,466589,391912,430591,323995]
						test2_search=[395296,357864,325292,365231]

					# print('debug feature_list',feature_list)
					question_id=feature_list[0][1]
					assert question_id==qid2featLidx['qid']

					feature_array = np.array(feature_list)[:,2:]
					assert feature_array.shape[1]==6 #6个特征

					lIdx2dIdx2pIdx_list=[]#每行特征对应的d_id和p_id
					for i in range(len(feature_list)):#总篇章数
						lIdx2dIdx2pIdx_list.append(qid2featLidx['lIdx2dIdx2pIdx'][str(i)])

					#加载归一化模型并应用模型进行数据归一化
					sc=joblib.load(model_dir+'sc.model')
					feature_std=sc.transform(feature_array)
					# print('debug feature_std.shape',feature_std.shape)
					#加载LR模型并应用模型进行预测
					lr=joblib.load(model_dir+'lr.model')
					pred_probs=lr.predict_proba(feature_std)[:,-1]
					# print('pred_probs.shape',pred_probs.shape)

					top5Idx=np.argsort(-pred_probs)[:5]#为1的概率从高到低排序，取top5
					# print('debug top5Idx',top5Idx)
					# print('debug np.array(lIdx2dIdx2pIdx_list)',np.array(lIdx2dIdx2pIdx_list))
					top5_para_ids=np.array(lIdx2dIdx2pIdx_list)[top5Idx]
					# print('debug top5_para_ids', top5_para_ids)

					rst={'question_id':question_id, 'top5_para_ids':top5_para_ids.tolist()}
					fw.write(json.dumps(rst)+'\n')
