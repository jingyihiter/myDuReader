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

#score_funcs中的函数全局选篇章和span
from score_funcs import scoreParag_recall, scoreParag_tfidf, scoreParag_ml, scoreSpan

# #全局变量
tfidfObj = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words=stopwords.words('chinese'))

def genet_tfidfObj(args):
    text = []#整个语料库组成文档集
    #debug
    # for data_path in args.demo_files:
    for data_path in args.train_files + args.dev_files + args.test_files:
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
    #usage python preprocess_yhl.py recall f1 a -n 8
    parser = argparse.ArgumentParser(description='Preprocess DuReader preprocessed dataset includs calculating parag scores and fake span scores.')
    

    mode_settings = parser.add_argument_group('mode settings')
    #默认打分函数都是全局计算
    mode_settings.add_argument('parag_scoreFunc', choices=["recall", "tfidf", "ml"], default='recall', help='score function for parags')
    mode_settings.add_argument('span_scoreFunc', choices=["f1", "bleu"], default='f1', help='score function for spans')
    mode_settings.add_argument('parag_selectMode', choices=["q", "a"], default='a', help='score(v.) parag between q or a')
    
    path_settings = parser.add_argument_group('path settings')

    # path_settings.add_argument('--demo_files', nargs='+',
    #                            default=['../data/demo/trainset/search.train.json'],
    #                            help='list of files that contain the preprocessed demo baidu search train data')

    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/preprocessed/trainset_v1/search.train.json', '../data/preprocessed/trainset_v1/zhidao.train.json'],
                               help='list of files that contain the preprocessed train data')
    
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/preprocessed/devset_v1/search.dev.json','../data/preprocessed/devset_v1/zhidao.dev.json'],
                               help='list of files that contain the preprocessed dev data')

    # path_settings.add_argument('--test_files', nargs='+',
    #                            default=['../data/preprocessed/testset_v1/search.test.json','../data/preprocessed/testset_v1/zhidao.test.json', '../data/preprocessed/testset_v2/search.test.json','../data/preprocessed/testset_v2/zhidao.test.json'],
    #                            help='list of files that contain the preprocessed test data')

    path_settings.add_argument('--test_files_v1', nargs='+',
                               default=['../data/preprocessed/testset_v1/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    
    parser.add_argument('--k', type=int, default=3,
                        help="Number of top scored spans to save")

    # parser.add_argument("-t", "--n_tokens", default=400, type=int,
    #                     help="Paragraph size")
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
    line, args, is_train = params
    sample = json.loads(line)
    
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
    if is_train:
        # print('start deal with spans')
        if args.span_scoreFunc=='f1' or "bleu":
            scoreSpan(sample, args)
        else:
            raise NotImplementedError(args.span_scoreFunc)

    #thirdly delete useless data
    # print('start delete useless data')
    delete_useless(sample, is_train)

    return json.dumps(sample, ensure_ascii=False)

def delete_useless(sample, is_train):
    """
    delete some unuseful data in sample
    :return:
    """
    sample.pop('question')
    for doc in sample['documents']:
        doc.pop('title')
        doc.pop('paragraphs')
    if is_train:
        sample.pop('answers')
        sample.pop('fake_answers')

'''
def _load_dataset(data_path, train=False):
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
                if sample['answer_spans'][0][1] >= 500:
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
            # if len(sortedParagResult)<5:
            #     print('sample[question_id]',sample['question_id'])
            #     print(paragScoreRecords)
            if train:
                spanScoreRecord=sample['spanScore_f1'][0]
                fake_span_didx, fake_span_pidx=spanScoreRecord[0], spanScoreRecord[1]
            odr=-1
            # print('\n')
            for r_idx, paragScoreRecord in enumerate(sortedParagResult[:5]):#取前5?
                # print(paragScoreRecord)
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
            # print(odr)
            # data_set.append(sample)
    return data_set
'''

if __name__ == '__main__':
    args=parse_args()
    print(args)

    print('开始计时....')
    start = time.time()

    #全数据集计算tfidf权重
    if args.parag_scoreFunc in ['tfidf', 'ml']:
        genet_tfidfObj(args)

    # #从文件读入
    # for demo_file in args.demo_files:
    #     i = 0
    #     work = list()
    #     with open(demo_file) as f_in:
    #         (filepath, tempfilename) = os.path.split(demo_file)
    #         (train_filename, extension) = os.path.splitext(tempfilename)
    #         out_filename = train_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+extension
    #         out_file_path=os.path.join(filepath, out_filename)
    #         with open(out_file_path, 'w') as f_out:
    #             with multiprocess.Pool(args.n_processes) as pool:
    #                 for line in f_in:
    #                     if i < 10:
    #                         work.append((line,args, True))#for trainset
    #                         i += 1
    #                     else:
    #                         pool_res = pool.map(score_func, work)
    #                         f_out.write('\n'.join(pool_res)+'\n')

    #                         work=list()
    #                         work.append((line,args, True))
    #                         i = 1
                            
    #                 if i > 0:#处理最后一批
    #                     pool_res = pool.map(score_func, work)
    #                     f_out.write('\n'.join(pool_res))

    # time_elapsed = time.time()-start
    # print('Training complete in {:.0f}min-{:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    # print('deal with demo trainset done')
    
    # for demo_file in args.demo_files:
    #     (filepath, tempfilename) = os.path.split(demo_file)
    #     (train_filename, extension) = os.path.splitext(tempfilename)
    #     out_filename = train_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+extension
    #     out_file_path=os.path.join(filepath, out_filename)
    #     train_set = _load_dataset(out_file_path, train=False)
    # print('test load done!')

    # #trainset
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
    #             with multiprocess.Pool(args.n_processes) as pool:
    #                 for line in f_in:
    #                     if i < 5000:
    #                         work.append((line,args, True))
    #                         i += 1
    #                     else:
    #                         pool_res = pool.map(score_func, work)
    #                         f_out.write('\n'.join(pool_res)+'\n')

    #                         work=list()
    #                         work.append((line,args, True))
    #                         i = 1
                            
    #                 if i > 0:#处理最后一批
    #                     pool_res = pool.map(score_func, work)
    #                     f_out.write('\n'.join(pool_res))
    
    # time_elapsed = time.time()-start
    # print('Training complete in {:.0f}min-{:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    # print('deal with trainset done')

    #dev_file
    # for dev_file in args.dev_files:
    #     i = 0
    #     work = list()
    #     with open(dev_file) as f_in:
    #         (filepath, tempfilename) = os.path.split(dev_file)
    #         (dev_filename, extension) = os.path.splitext(tempfilename)
    #         if args.parag_selectMode=='a':
    #             out_filename = dev_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+extension
    #         else:
    #             out_filename = dev_filename+'_'+args.parag_scoreFunc+extension
    #         out_file_path=os.path.join(filepath, out_filename)
    #         with open(out_file_path, 'w') as f_out:
    #             with multiprocess.Pool(args.n_processes) as pool:
    #                 for line in f_in:
    #                     if i < 5000:
    #                         work.append((line, args, False))#倒数第一参数，决定是否计算fake span
    #                         i += 1
    #                     else:
    #                         pool_res = pool.map(score_func, work)
    #                         f_out.write('\n'.join(pool_res)+'\n')

    #                         work=list()
    #                         work.append((line, args, False))
    #                         i = 1
                            
    #                 if i > 0:#处理最后一批
    #                     pool_res = pool.map(score_func, work)
    #                     f_out.write('\n'.join(pool_res))

    # time_elapsed = time.time()-start
    # print('Training complete in {:.0f}min-{:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    # print('deal with devset done')

    #test_file
    for test_file in args.test_files_v1:
        print('start dealing with ',test_file)
        i = 0
        work = list()
        total_line=0
        write_line=0
        with open(test_file) as f_in:
            (filepath, tempfilename) = os.path.split(test_file)
            (test_filename, extension) = os.path.splitext(tempfilename)
            out_filename = test_filename+'_'+args.parag_scoreFunc+extension
            out_file_path=os.path.join(filepath, out_filename)
            with open(out_file_path, 'w') as f_out:
                with multiprocess.Pool(args.n_processes) as pool:
                    for line in f_in:
                        total_line+=1
                        if i < 5000:
                            work.append((line,args, False))
                            i += 1
                        else:
                            pool_res = pool.map(score_func, work)
                            write_line+=len(pool_res)
                            print('one pool later ',write_line)
                            f_out.write('\n'.join(pool_res)+'\n')

                            work=list()
                            work.append((line,args, False))
                            i = 1
                            
                    if i > 0:#处理最后一批
                        pool_res = pool.map(score_func, work)
                        write_line+=len(pool_res)
                        print('the last one pool later ',write_line)
                        f_out.write('\n'.join(pool_res))
        
        time_elapsed = time.time()-start
        print('Training complete in {:.0f}min-{:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)) # 打印出来时间

    time_elapsed = time.time()-start
    print('Training complete in {:.0f}min-{:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    print('deal with testset done')
