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
##改进：全局选择most_related_paras（5个） 和 fake_span（1个）

class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

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
                bestspan_idx=-1
                if train:
                    if 'spanScore_f1' in sample:
                        bestspan_idx=0
                        if len(sample['spanScore_f1']) == 0:
                            continue
                        if sample['spanScore_f1'][bestspan_idx][0]==-1:
                            continue
                        if sample['spanScore_f1'][bestspan_idx][2][1] >= self.max_p_len:
                            continue
                    else:#9w数据f1选span优化了，故数据域有变化
                        assert 'multi_spanScore_f1' in sample
                        if len(sample['multi_spanScore_f1']) == 0:
                            continue
                        bestspan_matchscore=0.0
                        for r_idx,record in enumerate(sample['multi_spanScore_f1']):
                            if record[0][-1]>bestspan_matchscore:
                                bestspan_matchscore=record[0][-1]
                                bestspan_idx=r_idx
                        if bestspan_idx==-1:
                            continue
                        if sample['multi_spanScore_f1'][bestspan_idx][0][2][1] >= self.max_p_len:
                            continue

                sample['passages'] = []#全局选择
                
                if train:
                    score_field='paragScore_recall_a'
                else:
                    score_field='paragScore_recall_q'
                
                paragScoreRecords=[]
                
                
                for k,v in sample[score_field].items():
                    for item in v:
                        paragScoreRecords.append((k,item[0],item[1]))
                sortedParagResult=sorted(paragScoreRecords, key=lambda record: record[-1],reverse=True)
                if train:
                    if 'spanScore_f1' in sample:
                        spanScoreRecord=sample['spanScore_f1'][bestspan_idx]
                        fake_span_didx, fake_span_pidx=spanScoreRecord[0], spanScoreRecord[1]
                        sample['answer_spans']=[spanScoreRecord[2]]
                    else:
                        assert 'multi_spanScore_f1' in sample
                        spanScoreRecord=sample['multi_spanScore_f1'][bestspan_idx][0]
                        fake_span_didx, fake_span_pidx=spanScoreRecord[0], spanScoreRecord[1]
                        sample['answer_spans']=[spanScoreRecord[2]]

                odr=-1
                for r_idx, paragScoreRecord in enumerate(sortedParagResult[:5]):#取前5?
                    if train and int(paragScoreRecord[0])==fake_span_didx and paragScoreRecord[1]==fake_span_pidx:
                        odr=r_idx
                    if train:
                        sample['passages'].append(
                                {'passage_tokens': sample['documents'][int(paragScoreRecord[0])]['segmented_paragraphs'][paragScoreRecord[1]],
                                 'is_selected': sample['documents'][int(paragScoreRecord[0])]['is_selected']}
                            )
                    else:
                        sample['passages'].append(
                                {'passage_tokens': sample['documents'][int(paragScoreRecord[0])]['segmented_paragraphs'][paragScoreRecord[1]]})
                if odr==-1:
                    if train:
                        odr=5
                        sample['passages'].append(
                                {'passage_tokens': sample['documents'][fake_span_didx]['segmented_paragraphs'][fake_span_pidx],
                                 'is_selected': sample['documents'][fake_span_didx]['is_selected']}
                            )
                sample['fake_span_order']=odr
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
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)#TODO
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
            # if 'answer_docs' in sample and len(sample['answer_docs']):
            if sample['fake_span_order']!=-1:
                # gold_passage_offset = padded_p_len * sample['answer_docs'][0]
                gold_passage_offset = padded_p_len * sample['fake_span_order']
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
