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
    #usage python preprocess_yhl.py 
    parser = argparse.ArgumentParser(description='Preprocess DuReader preprocessed dataset includs calculating parag scores and fake span scores.')
    

    mode_settings = parser.add_argument_group('mode settings')
    #默认打分函数都是全局计算
    mode_settings.add_argument('parag_scoreFunc', choices=["recall", "tfidf", "ml"], default='recall', help='score function for parags')
    mode_settings.add_argument('span_scoreFunc', choices=["f1", "bleu"], default='f1', help='score function for spans')
    mode_settings.add_argument('parag_selectMode', choices=["q", "a"], default='a', help='score(v.) parag between q or a')
    
    path_settings = parser.add_argument_group('path settings')

    path_settings.add_argument('--demo_files', nargs='+',
                               default=['../data/demo/trainset/search.train.json'],
                               help='list of files that contain the preprocessed demo baidu search train data')

    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/preprocessed/trainset/search.train.json', '../data/preprocessed/trainset/zhidao.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/preprocessed/devset/search.dev.json','../data/preprocessed/devset/zhidao.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/preprocessed/testset/search.test.json','../data/preprocessed/testset/zhidao.test.json'],
                               help='list of files that contain the preprocessed test data')

    parser.add_argument('--k', type=int, default=3,
                        help="Number of top scored spans to save")

    # parser.add_argument("-t", "--n_tokens", default=400, type=int,
    #                     help="Paragraph size")
    parser.add_argument('-n', '--n_processes', type=int, default=1,
                        help="Number of processes (i.e., ) ")
    return parser.parse_args()


def score_func(params):
    line, args = params
    sample = json.loads(line)
    """
    For each sample, score every paragraph and every span under specified func in turn.
    Args:
        sample: a sample in the dataset
    Returns:
        None
    Raises:
        None
    """
    #firstly score parag
    # print('start deal with parags...')
    if args.parag_scoreFunc=='recall':
        scoreParag_recall(sample, args)
    elif args.parag_scoreFunc=="tfidf":
        scoreParag_tfidf(sample, args, tfidfObj)
    elif args.parag_scoreFunc=="ml":
        scoreParag_ml(sample, args, tfidfObj)
    else:
        raise NotImplementedError(args.parag_scoreFunc)

    #secondly score span
    # print('start deal with spans')
    if args.span_scoreFunc=='f1' or "bleu":
        scoreSpan(sample, args)
    else:
        raise NotImplementedError(args.span_scoreFunc)

    #thirdly delect useless data
    # print('start delect useless data')
    delete_useless(sample)

    return json.dumps(sample, ensure_ascii=False)

def delete_useless(sample):
    """
    delete some unuseful data in sample
    :return:
    """
    sample.pop('question')
    sample.pop('answers')
    sample.pop('fake_answers')
    for doc in sample['documents']:
        doc.pop('title')
        doc.pop('paragraphs')

if __name__ == '__main__':
    args=parse_args()
    print(args)

    print('开始计时....')
    start = time.time()

    #全数据集计算tfidf权重
    if args.parag_scoreFunc=='tfidf' or 'ml':
        genet_tfidfObj(args)

    #从文件读入
    for demo_file in args.demo_files:
        i = 0
        work = list()
        with open(demo_file) as f_in:
            (filepath, tempfilename) = os.path.split(demo_file)
            (train_filename, extension) = os.path.splitext(tempfilename)
            out_filename = train_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+extension
            out_file_path=os.path.join(filepath, out_filename)
            with open(out_file_path, 'w') as f_out:
                with multiprocess.Pool(args.n_processes) as pool:
                    for line in f_in:
                        if i < 10000:
                            work.append((line,args))
                            i += 1
                        else:
                            pool_res = pool.map(score_func, work)
                            f_out.write('\n'.join(pool_res))

                            work=list()
                            work.append((line,args))
                            i = 1
                            
                    if i > 0:#处理最后一批
                        pool_res = pool.map(score_func, work)
                        f_out.write('\n'.join(pool_res))

    time_elapsed = time.time()-start
    print('Training complete in {:.0f}min-{:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    print('all done')

    # #从文件读入
    # for train_file in args.train_files:
    #     i = 0
    #     work = list()
    #     with open(train_file) as f_in:
    #         (filepath, tempfilename) = os.path.split(train_file)
    #         (train_filename, extension) = os.path.splitext(tempfilename)
    #         out_filename = train_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+extension
    #         out_file_path=os.path.join(filepath, out_filename)
    #         with open(out_file_path, 'w') as f_out:
    #             with multiprocess.Pool(4) as pool:
    #                 for line in f_in:
    #                     if i < 10000:
    #                         work.append((line,args))
    #                         i += 1
    #                     else:
    #                         pool_res = pool.map(score_func, work)
    #                         f_out.write('\n'.join(pool_res))

    #                         work=list()
    #                         work.append((line,args))
    #                         i = 1
                            
    #                 if i > 0:#处理最后一批
    #                     pool_res = pool.map(score_func, work)
    #                     f_out.write('\n'.join(pool_res))
            
    # #dev_file
    # #test_file