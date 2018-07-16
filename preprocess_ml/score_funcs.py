import sys
if sys.version[0] == '2':
	reload(sys)
	sys.setdefaultencoding("utf-8")

from collections import Counter
import numpy as np
from sklearn.metrics import pairwise_distances
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
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
	for d_idx, doc in enumerate(sample['documents']):
		if not doc['is_selected']:
			continue
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

		if args.parag_selectMode=='a':
			if 'paragScore_recall_a' not in sample:
				sample['paragScore_recall_a']={}
			sample['paragScore_recall_a'][str(d_idx)]=scoreRecords
		elif args.parag_selectMode=='q':
			if 'paragScore_recall_q' not in sample:
				sample['paragScore_recall_q']={}
			sample['paragScore_recall_q'][str(d_idx)]=scoreRecords

def scoreParag_tfidf(sample, args, tfidfObj):
	ques_text = " ".join(sample['segmented_question'])
	for d_idx, doc in enumerate(sample['documents']):
		if not doc['is_selected']:
			continue
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
			if dists[0] < 1.0:
				scoreRecords.append((p_idx, dists[0]))#debug dist==1?!
		
		if 'paragScore_tfidf_q' not in sample:
			sample['paragScore_tfidf_q']={}
		sample['paragScore_tfidf_q'][str(d_idx)]=scoreRecords

def scoreParag_ml(sample, args, tfidfObj):
	#TFIDF_W, WORDOVERLAP_W = 0.5, 0.5  # wordoverlap 在测试的时候这一特征对结果没有影响
	# 不同数据集的权值不一样
	# search
	para_f1_W, para_recall_W = 0.020747168, 0.072683327
	first_para_W = 0.057613939
	tfidf_W = -0.04508809
	bleu_W = 0.0096270768
	# zhidao
	para_f1_W, para_recall_W = 0.017479911, 0.067206331
	first_para_W = 0.056296296
	tfidf_W = -0.041528691
	bleu_W = 0.0077238618

	stop_words=stopwords.words('chinese')

	question=sample['segmented_question']
	q_words = {x for x in question if x not in stop_words}

	for d_idx, doc in enumerate(sample['documents']):
		if not doc['is_selected']:
			continue
		scoreRecords=[]
		for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
			para_f1 = metric_max_over_ground_truths(f1_score, para_tokens, [question])
			para_recall = metric_max_over_ground_truths(recall, para_tokens, [question])
			# whether the paragraph was the first in document
			first_para = 1 if p_idx==0 else 0
			smoothie = SmoothingFunction().method4
			bleu_score = sentence_bleu(question, para_tokens, smoothing_function=smoothie)

			parag_text = ' '.join(para_tokens)
			try:
				para_features = tfidfObj.transform([parag_text])
				q_features = tfidfObj.transform([" ".join(question)])
			except ValueError:
				pass
			tfidf_score = pairwise_distances(q_features, para_features, "cosine").ravel()
			# 1:para_f1 2:para_recall 3:first_para 4:tfidf_score 5:bleu_score_para
			score = para_f1*para_f1_W + para_recall*para_recall_W +first_para*first_para_W+ tfdit_W* tfidf_score[0] + bleu_score*bleu_W
			
			scoreRecords.append((p_idx, score))
		
		if 'paragScore_ml_q' not in sample:
			sample['paragScore_ml_q']={}
		sample['paragScore_ml_q'][str(d_idx)]=scoreRecords

def scoreSpan(sample, args):
	bestK_match = [(-1,-1,[-1,-1],None,0.0)]#slot:(d_idx, p_idx, span_idxs, span_text, matchScore)
	stop_words=stopwords.words('chinese')
	answer_tokens = set()
	for segmented_answer in sample['segmented_answers']:
		answer_tokens = answer_tokens | set([token for token in segmented_answer if token not in stop_words])#去停用词？
	
	for d_idx, doc in enumerate(sample['documents']):
		if not doc['is_selected']:
			continue
		for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
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

					for i in range(len(bestK_match)-1, -1, -1): #逆序遍历
						if match_score <= bestK_match[i][-1] or (i==0 and match_score > bestK_match[i][-1]):#找到第一个比它大的
							if match_score > bestK_match[i][-1]:
								bestK_match.insert(i, (d_idx, p_idx, [start_tidx, end_tidx], ''.join(span_tokens), match_score))	
							else:
								bestK_match.insert(i+1, (d_idx, p_idx, [start_tidx, end_tidx], ''.join(span_tokens), match_score))
							if len(bestK_match) == args.k+1:
								bestK_match.pop()#删除最后一个元素,保持在k个内
							break

	assert len(bestK_match) <= args.k, 'wrong with best k spans'

	if args.span_scoreFunc=='f1':
		sample['spanScore_f1'] = bestK_match
	elif args.span_scoreFunc=='bleu':
		sample['spanScore_bleu'] = bestK_match
