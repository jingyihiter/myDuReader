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
import re


#全局变量
tfidfObj = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words=stopwords.words('chinese'))

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

    # path_settings.add_argument('--train_files', nargs='+',
    #                            default=['../data/preprocessed/trainset/zhidao.train.json'],
    #                            help='list of files that contain the preprocessed train data')
    
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



if __name__ == '__main__':
    args=parse_args()
    print(args)
    rule_train= r'\{"documents":'
    rule_untrain= r'\{"documents":(.*?)\}\}'
    print('开始计时....')
    start = time.time()

    # #从文件读入 {"documents":
    # for demo_file in args.demo_files:
    #     (filepath, tempfilename) = os.path.split(demo_file)
    #     (train_filename, extension) = os.path.splitext(tempfilename)
    #     out_filename = train_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+extension
    #     reviewd_out_filename = train_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+'_v1'+extension
    #     out_file_path=os.path.join(filepath, out_filename)
    #     reviewd_out_file_path=os.path.join(filepath, reviewd_out_filename)
    #     with open(out_file_path, 'r') as f_in:
    #         with open(reviewd_out_file_path, 'w') as f_out:
    #             for line in f_in:
    #                 pos = [m.start() for m in re.finditer(rule_train, line)]
    #                 if len(pos)==2:
    #                     f_out.write(line[:pos[1]]+'\n')
    #                     f_out.write(line[pos[1]:])
    #                 else:
    #                     f_out.write(line)

    # time_elapsed = time.time()-start
    # print('Training complete in {:.0f}min-{:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    # print('all done')
    
    # for demo_file in args.demo_files:
    #     (filepath, tempfilename) = os.path.split(demo_file)
    #     (train_filename, extension) = os.path.splitext(tempfilename)
    #     out_filename = train_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+extension
    #     out_file_path=os.path.join(filepath, out_filename)
    #     train_set = _load_dataset(out_file_path, train=False)
    # print('test load done!')

    #trainset
    #从文件读入
    for train_file in args.train_files:
        (filepath, tempfilename) = os.path.split(train_file)
        (train_filename, extension) = os.path.splitext(tempfilename)
        out_filename = train_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+'_v0'+extension
        reviewd_out_filename = train_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+extension
        out_file_path=os.path.join(filepath, out_filename)
        reviewd_out_file_path=os.path.join(filepath, reviewd_out_filename)
        l_idx=0
        with open(out_file_path, 'r') as f_in:
            with open(reviewd_out_file_path, 'w') as f_out:
                for line in f_in:
                    pos = [m.start() for m in re.finditer(rule_train, line)]
                    if len(pos)==2:
                        f_out.write(line[:pos[1]]+'\n')
                        f_out.write(line[pos[1]:])
                    else:
                        f_out.write(line)
    
    time_elapsed = time.time()-start
    print('Training complete in {:.0f}min-{:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    print('deal with trainset done')

    #dev_file
    for dev_file in args.dev_files:
        (filepath, tempfilename) = os.path.split(dev_file)
        (dev_filename, extension) = os.path.splitext(tempfilename)
        out_filename = dev_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+'_v0'+extension
        reviewd_out_filename = dev_filename+'_'+args.parag_scoreFunc+'_'+args.span_scoreFunc+extension
        out_file_path=os.path.join(filepath, out_filename)
        reviewd_out_file_path=os.path.join(filepath, reviewd_out_filename)
        with open(out_file_path, 'r') as f_in:
            with open(reviewd_out_file_path, 'w') as f_out:
                for line in f_in:
                    pos = [m.start() for m in re.finditer(rule_train, line)]
                    if len(pos)==2:
                        f_out.write(line[:pos[1]]+'\n')
                        f_out.write(line[pos[1]:])
                    else:
                        f_out.write(line)

    time_elapsed = time.time()-start
    print('Training complete in {:.0f}min-{:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    print('deal with devset done')

    # #test_file
    # for test_file in args.test_files:
        # (filepath, tempfilename) = os.path.split(test_file)
        # (test_filename, extension) = os.path.splitext(tempfilename)
        # out_filename = test_filename+'_'+args.parag_scoreFunc+extension
        # reviewd_out_filename = test_filename+'_'+args.parag_scoreFunc+'_v1'+extension
        # out_file_path=os.path.join(filepath, out_filename)
        # reviewd_out_file_path=os.path.join(filepath, reviewd_out_filename)
        # with open(out_file_path, 'r') as f_in:
        #     with open(reviewd_out_file_path, 'w') as f_out:
        #         for line in f_in:
        #             jsonStrs=re.findall(rule_untrain, line)
        #             if len(jsonStrs)==2:
        #                 f_out.write('{\"documents\":'+jsonStrs[0]+'}}'+'\n')
        #                 f_out.write('{\"documents\":'+jsonStrs[1]+'}}'+'\n')
        #             else:
        #                 f_out.write(line)

    # time_elapsed = time.time()-start
    # print('Training complete in {:.0f}min-{:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    # print('deal with testset done')
