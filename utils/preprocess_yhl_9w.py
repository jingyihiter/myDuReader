# -*- coding: UTF-8 -*-

###############################################################################
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
This module finds the most related paragraph of each document according to recall.
"""
import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import os
import argparse
import json
import multiprocessing as multiprocess
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from score_funcs import scoreParag_recall, scoreParag_tfidf, scoreParag_ml, scoreSpan

#处理训练集中新增的9w数据---全局选para和span，但鉴于span用f1选，所以做了优化，不仅选多answer span，并且为每个answer选对应的多个answer span
#处理结果为了减少I/O，仅保存结果

#usage python preprocess_yhl_9w.py recall f1 a -n 16

#全局变量
tfidfObj = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words=stopwords.words('chinese'))

def genet_tfidfObj(args):
    text = []#整个语料库组成文档集
    # for data_path in args.train_files + args.dev_files + args.test_files:
    #debug
    for data_path in args.demo_files:
        with open(data_path) as f_in:
            for line in f_in:
                sample = json.loads(line)
                text.append(' '.join(sample['segmented_question']))#加入问题
                for doc in sample['documents']:
                    for para_tokens in doc['segmented_paragraphs']:#加入篇章
                        text.append(' '.join(para_tokens))
    try:
        tfidfObj.fit(text)
        print('bag of words num:',len(tfidfObj.get_feature_names()))
    except ValueError:
        pass

def parse_args():
    
    parser = argparse.ArgumentParser(description='Preprocess DuReader preprocessed dataset includs calculating parag scores and fake span scores.')
    

    mode_settings = parser.add_argument_group('mode settings')
    #默认打分函数都是全局计算
    mode_settings.add_argument('parag_scoreFunc', choices=["recall", "tfidf", "ml"], default='recall', help='score function for parags')
    mode_settings.add_argument('span_scoreFunc', choices=["f1", "bleu"], default='f1', help='score function for spans')
    mode_settings.add_argument('parag_selectMode', choices=["q", "a"], default='a', help='score(v.) parag between q or a')
    
    path_settings = parser.add_argument_group('path settings')

    # path_settings.add_argument('--train_files', nargs='+',
    #                            default=['../data/preprocessed/trainset_v1/search.train.json', '../data/preprocessed/trainset_v1/zhidao.train.json'],
    #                            help='list of files that contain the preprocessed train data')
    
    #TODO只处理zhidao '../data/preprocessed/trainset_v1/search.train.diffQids.json',
    path_settings.add_argument('--train_Qids_toDeal_files', nargs='+',
                               default=[ '../data/preprocessed/trainset_v1/zhidao.train.diffQids.json'])
    parser.add_argument('--k', type=int, default=3,
                        help="Number of top scored spans to save")

    parser.add_argument('-n', '--n_processes', type=int, default=1,
                        help="Number of processes (i.e., ) ")
    return parser.parse_args()


def score_func(params):
    """
    For each sample, score every paragraph and every span under specified func in turn.
    Args:
        sample: a sample in the dataset
    Returns:
        None
    Raises:
        None
    """
    line, args, needScoreSpan = params
    sample = json.loads(line)
    
    #firstly score parag
    # print('start deal with parags...')
    if args.parag_scoreFunc=='recall':
        rst_paraRecall = scoreParag_recall(sample, args)
    elif args.parag_scoreFunc=="tfidf":
        scoreParag_tfidf(sample, args, tfidfObj)
    elif args.parag_scoreFunc=="ml":
        scoreParag_ml(sample, args, tfidfObj)
    else:
        raise NotImplementedError(args.parag_scoreFunc)

    #secondly score span
    if needScoreSpan:
        # print('start deal with spans')
        if args.span_scoreFunc=='f1' or "bleu":
            rst_scoreF1 = scoreSpan(sample, args)
        else:
            raise NotImplementedError(args.span_scoreFunc)

    assert rst_paraRecall['question_id']==rst_scoreF1['question_id']

    rst_paraRecall['multi_spanScore_f1']=rst_scoreF1['multi_spanScore_f1']#合并返回结果
    
    # #thirdly delete useless data
    # print('start delete useless data')
    # delete_useless(sample, needScoreSpan)

    return json.dumps(rst_paraRecall, ensure_ascii=False)#仅保存处理结果

def delete_useless(sample, needScoreSpan):
    """
    delete some unuseful data in sample
    :return:
    """
    sample.pop('question')
    for doc in sample['documents']:
        doc.pop('title')
        doc.pop('paragraphs')
    if needScoreSpan:
        sample.pop('answers')
        sample.pop('fake_answers')

if __name__ == '__main__':
    args=parse_args()
    print(args)

    print('开始计时....')
    start = time.time()

    #全数据集计算tfidf权重
    if args.parag_scoreFunc in ['tfidf', 'ml']:
        genet_tfidfObj(args)

    #trainset
    #从文件读入
    for train_file in args.train_Qids_toDeal_files:
        i = 0
        work = list()
        with open(train_file, 'r') as f_in:
            (filepath, tempfilename) = os.path.split(train_file)
            (train_filename, extension) = os.path.splitext(tempfilename)
            out_filename = train_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+extension
            out_file_path=os.path.join(filepath, out_filename)
            with open(out_file_path, 'w') as f_out:
                with multiprocess.Pool(args.n_processes) as pool:
                    for line in f_in:
                        if i < 5000:
                            work.append((line, args, True))
                            i += 1
                        else:
                            pool_res = pool.map(score_func, work)
                            f_out.write('\n'.join(pool_res)+'\n')

                            work=list()
                            work.append((line, args, True))
                            i = 1
                            
                    if i > 0:#处理最后一批
                        pool_res = pool.map(score_func, work)
                        f_out.write('\n'.join(pool_res))
        print('done 1')
        time_elapsed = time.time()-start
        print('Training complete in {:.0f}min-{:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    
    time_elapsed = time.time()-start
    print('Training complete in {:.0f}min-{:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    print('deal with trainset done')
