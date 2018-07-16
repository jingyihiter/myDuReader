# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from collections import Counter

##改进v2: 训练集用全局选的para和span，但是开发集和测试集仍然按baseline方法
##改进multispan: 多篇章训练，预处理得到每个sample多个fake_span(至多3个)，形成训练数据时，1正，3top para，1顺次选，形成样本

class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_files=[], dev_files=[], test_files=[], multiSpan_files=[]):#TODO
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        
        # f1_thresholds=[0.75, 0.8]#f1_threshold for search and zhidao
        f1_thresholds=[0.85, 0.9]#f1_threshold for search and zhidao
        self.train_set, self.dev_set, self.test_set = [], [], []

        if train_files:
            for train_file, multiSpan_file, threshold in zip(train_files, multiSpan_files, f1_thresholds):
                self.train_set += self._load_dataset_train(train_file, multiSpan_file, threshold)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset_train(self, data_globalPara_path, data_globalMultiSpan_path, threshold=0.0):
        """
        Loads the dataset
        Args:
            data_globalPara_path: the data file to load
        """
        multiSpanRecords={}
        with open(data_globalMultiSpan_path) as fin_multiSpan:
            for line in fin_multiSpan:
                multiSpanRecord = json.loads(line.strip())
                multiSpanRecords[multiSpanRecord['question_id']]=multiSpanRecord
        # print('len(multiSpanRecords)', len(multiSpanRecords))#136208

        with open(data_globalPara_path) as fin_para:
            data_set = []
            for lidx, line in enumerate(fin_para):
                sample = json.loads(line.strip())
                multiSpanRecord=multiSpanRecords[sample['question_id']]
                idx_toremove=[]
                for idx, spanScore in enumerate(multiSpanRecord['multi_spanScore_f1']):
                    if spanScore[2][1]>= self.max_p_len or spanScore[-1]<threshold:
                        idx_toremove.append(idx)
                multiSpanRecord['multi_spanScore_f1']=[item for i,item in enumerate(multiSpanRecord['multi_spanScore_f1']) if i not in idx_toremove]
                
                ans_para_num=len(multiSpanRecord['multi_spanScore_f1'])#会形成的样本个数~

                if ans_para_num==0:
                    continue
                
                passages_group=[]
                # for_debug=[]
                fake_ans_record_group=[]
                for i in range(ans_para_num):
                    passages_group.append([])
                    # for_debug.append([])
                    fake_ans_record_group.append(())
                group_idx_pos=0#标示 找到的fake_span所在passage送入的group
                group_idx_neg=0#标示 找到的负例送入的group
                
                paragScoreRecords=[]
                for k,v in sample['paragScore_recall_a'].items():
                    for item in v:
                        paragScoreRecords.append((k,item[0],item[1]))
                
                sortedParagResult=sorted(paragScoreRecords, key=lambda record: record[-1],reverse=True)
                # print('len(sortedParagResult)',len(sortedParagResult))
                pos_num=0
                for r_idx, paragScoreRecord in enumerate(sortedParagResult):
                    is_pos=False#每篇passage初始化为不是正例
                    #首先判断是否是正例，若是，则放入指定的group，并不再作为负例
                    for psg_idx, spanScore in enumerate(multiSpanRecord['multi_spanScore_f1']):
                        if int(paragScoreRecord[0])==spanScore[0] and paragScoreRecord[1]==spanScore[1] and group_idx_pos<ans_para_num:
                            is_pos=True
                            if group_idx_pos<ans_para_num:#每组只会增加一个额外正例
                                passages_group[group_idx_pos].append(
                                            {'passage_tokens': sample['documents'][int(paragScoreRecord[0])]['segmented_paragraphs'][paragScoreRecord[1]],
                                             'is_selected': sample['documents'][int(paragScoreRecord[0])]['is_selected']}
                                        )
                                fake_ans_record_group[group_idx_pos]=(len(passages_group[group_idx_pos])-1, spanScore[2])
                                # for_debug[group_idx_pos].append((r_idx,paragScoreRecord[0],paragScoreRecord[1]))
                                group_idx_pos+=1
                    if is_pos:
                        pos_num+=1
                    #作为负例加入相应的group-(1)首先是否属于top3 负例
                    if is_pos==False:
                        if r_idx+1 - pos_num<3:#说明该篇章属于top3负例（当前总passage数-属于正例passage数）
                            for group_idx in range(ans_para_num):
                                passages_group[group_idx].append(
                                            {'passage_tokens': sample['documents'][int(paragScoreRecord[0])]['segmented_paragraphs'][paragScoreRecord[1]],
                                             'is_selected': sample['documents'][int(paragScoreRecord[0])]['is_selected']}
                                        )
                                # for_debug[group_idx].append((r_idx,paragScoreRecord[0],paragScoreRecord[1]))
                        else:
                            if group_idx_neg<ans_para_num:#每组只会增加一个额外负例
                                passages_group[group_idx_neg].append(
                                            {'passage_tokens': sample['documents'][int(paragScoreRecord[0])]['segmented_paragraphs'][paragScoreRecord[1]],
                                             'is_selected': sample['documents'][int(paragScoreRecord[0])]['is_selected']}
                                        )
                                # for_debug[group_idx_neg].append((r_idx,paragScoreRecord[0],paragScoreRecord[1]))
                                group_idx_neg+=1
                    if group_idx_pos==ans_para_num and group_idx_neg==ans_para_num:#每组已经找够了
                        break

                # if len(for_debug[0])==6:
                #     tt=[]
                #     for record in multiSpanRecord['multi_spanScore_f1']:
                #         tt.append((record[0],record[1]))
                #     print('multiSpanRecord[\'multi_spanScore_f1\']',tt)
                #     print('len(sortedParagResult)',len(sortedParagResult))
                #     print('fake_ans_record_group',fake_ans_record_group)
                #     print(for_debug)
                #     break#TODO--debug
                sample.pop('documents')
                for  p_group, fake_ans_record in zip(passages_group, fake_ans_record_group):#形成多个sample
                    sample['passages']=p_group
                    sample['fake_ans_record']=fake_ans_record
                    data_set.append(sample)
        return data_set

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path) as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue

                sample['passages'] = []
                
                for d_idx, doc in enumerate(sample['documents']):
                    if train:
                        most_related_para = doc['most_related_para']
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected']}
                        )
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            question_tokens = sample['segmented_question']
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]
                        sample['passages'].append({'passage_tokens': fake_passage_tokens})
                sample.pop('documents')
                data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        max_passage_num = self.max_p_num
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'fake_ans_record'in sample:#训练集全局
                gold_passage_offset = padded_p_len * sample['fake_ans_record'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['fake_ans_record'][1][0])
                batch_data['end_id'].append(gold_passage_offset + sample['fake_ans_record'][1][1])
            
            elif 'answer_docs' in sample and len(sample['answer_docs']):#开发集仍按照baseline
                gold_passage_offset = padded_p_len * sample['answer_docs'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            
            else:
                # fake span for some samples, only valid for testing-----TODO and evaluate...
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['segmented_question']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['segmented_question'])
                for passage in sample['passages']:
                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)
