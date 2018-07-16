# -*- coding: UTF-8 -*-
import sys
if sys.version[0] == '2':
	reload(sys)
	sys.setdefaultencoding("utf-8")

from collections import Counter
import numpy as np
from sklearn.metrics import pairwise_distances
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords

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


def f1_score(prediction, ground_truth):
	"""
	This function calculates and returns the f1-score
	Args:
		prediction: prediction string or list to be matched
		ground_truth: golden string or list reference
	Returns:
		floats of f1
	Raises:
		None
	"""
	return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
	"""
	This function calculates and returns the precision, recall and f1-score
	Args:
		metric_fn: metric function pointer which calculates scores according to corresponding logic.
		prediction: prediction string or list to be matched
		ground_truth: golden string or list reference
	Returns:
		floats of (p, r, f1)
	Raises:
		None
	"""
	scores_for_ground_truths = []
	for ground_truth in ground_truths:#每次一个ground_truth
		score = metric_fn(prediction, ground_truth)
		scores_for_ground_truths.append(score)
	return max(scores_for_ground_truths)#返回最大的

###########################################################################
def scoreParag_recall(sample,args):
	#全局打分
	if args.parag_selectMode=='a':
		field_name='paragScore_recall_a'
		
	elif args.parag_selectMode=='q':
		field_name='paragScore_recall_q'
	sample[field_name]={}
	
	for d_idx, doc in enumerate(sample['documents']):
		# if not doc['is_selected']:
		# 	continue
		scoreRecords=[]
		for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
			if args.parag_selectMode=='a':#计算answer和parag之间的recall
				if len(sample['segmented_answers']) > 0:
					related_score = metric_max_over_ground_truths(recall,para_tokens,sample['segmented_answers'])
				else:
					related_score=0.0
			elif args.parag_selectMode=='q':
				related_score = recall(para_tokens,sample['segmented_question'])
			scoreRecords.append((p_idx, related_score))
		sample[field_name][d_idx]=scoreRecords


def scoreParag_tfidf(sample, args, tfidfObj):
	
	sample['paragScore_tfidf_q']={}	
	ques_text = " ".join(sample['segmented_question'])
	for d_idx, doc in enumerate(sample['documents']):
		# if not doc['is_selected']:
		# 	continue
		scoreRecords=[]
		for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
			parag_text = ' '.join(para_tokens)
			try:
				para_features = tfidfObj.transform([parag_text])
				q_features = tfidfObj.transform([ques_text])
			except ValueError:
				pass
			#计算余弦相似度
			dists = pairwise_distances(q_features, para_features, "cosine").ravel()#将多维数组降位一维，返回视图
			# if dists[0] < 1.0:
			# 	scoreRecords.append((p_idx, dists[0]))#debug dist==1?!
			scoreRecords.append((p_idx, dists[0]))#debug dist==1?!

		sample['paragScore_tfidf_q'][str(d_idx)]=scoreRecords

def scoreParag_ml(sample, args, tfidfObj):
	sample['paragScore_ml_q']={}
	TFIDF_W, WORDOVERLAP_W = 0.5, 0.5

	stop_words=stopwords.words('chinese')

	question=sample['segmented_question']
	q_words = {x for x in question if x not in stop_words}

	for d_idx, doc in enumerate(sample['documents']):
		# if not doc['is_selected']:
		# 	continue
		scoreRecords=[]
		for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
			found = set()
			for word in para_tokens:
				if word in q_words:
					found.add(word)
			word_match_feature=len(found)
			
			parag_text = ' '.join(para_tokens)
			try:
				para_features = tfidfObj.transform([parag_text])
				q_features = tfidfObj.transform([" ".join(question)])
			except ValueError:
				pass
			tfidf_score = pairwise_distances(q_features, para_features, "cosine").ravel()

			score = TFIDF_W* tfidf_score[0] + WORDOVERLAP_W * word_match_feature
			
			scoreRecords.append((p_idx, score))
		
		sample['paragScore_ml_q'][str(d_idx)]=scoreRecords

def scoreSpan(sample, args):
	bestK_match = [(-1,-1,[-1,-1],None,0.0)]#slot:(d_idx, p_idx, span_idxs, span_text, matchScore)
	answer_tokens = set()
	for segmented_answer in sample['segmented_answers']:
		answer_tokens = answer_tokens | set([token for token in segmented_answer])#去停用词？
	
	for d_idx, doc in enumerate(sample['documents']):
		# if not doc['is_selected']:
		# 	continue
		
		for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):#debug每个篇章选一个fake_span出来
			best_match_score_p=0.0
			best_start_idx, best_end_idx = -1, -1
			best_span_tokens=None

			truncated_para_tokens = para_tokens[:1000]
			for start_tidx in range(len(truncated_para_tokens)):
				if truncated_para_tokens[start_tidx] not in answer_tokens:
					continue
				for end_tidx in range(len(truncated_para_tokens) - 1, start_tidx - 1, -1):
					span_tokens = truncated_para_tokens[start_tidx: end_tidx + 1]

					if len(sample['segmented_answers']) > 0:
						if args.span_scoreFunc=='f1':
							match_score = metric_max_over_ground_truths(f1_score, span_tokens,sample['segmented_answers'])
						elif args.span_scoreFunc=='bleu':
							reference, candidate = sample['segmented_answers'], span_tokens
							match_score = sentence_bleu(reference, candidate)
					else:
						match_score = 0
					if match_score == 0:
						break

					if match_score > best_match_score_p:
						best_match_score_p = match_score
						best_start_idx, best_end_idx = start_tidx, end_tidx
						best_span_tokens = span_tokens
					
			#每个篇章选择一个fake_span
			for i in range(len(bestK_match)): #正序遍历
				if best_match_score_p > bestK_match[i][-1]:
					bestK_match.insert(i, (d_idx, p_idx, [best_start_idx, best_end_idx], ''.join(best_span_tokens), best_match_score_p))	
					if len(bestK_match) == args.k+1:
						bestK_match.pop()#删除最后一个元素,保持在k个内
					break

	assert len(bestK_match) <= args.k, 'wrong with best k spans'

	if args.span_scoreFunc=='f1':
		sample['spanScore_f1'] = bestK_match
	elif args.span_scoreFunc=='bleu':
		sample['spanScore_bleu'] = bestK_match
