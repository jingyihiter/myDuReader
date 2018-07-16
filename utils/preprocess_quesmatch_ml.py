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

from score_func_quesmatch_feat_extract_ml import scoreParag_ml, lr_predict_forTOP5

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
        print('one file loaded...')
    try:
        tfidfObj.fit(text)
        print('bag of words num:',len(tfidfObj.get_feature_names()))
    except ValueError:
        pass

def parse_args():
    #usage python preprocess_yhl.py ml q -n 8
    parser = argparse.ArgumentParser(description='Preprocess DuReader preprocessed dataset includs calculating parag scores and fake span scores.')
    

    mode_settings = parser.add_argument_group('mode settings')
    #默认打分函数都是全局计算
    mode_settings.add_argument('parag_scoreFunc', choices=["ml"], default='ml', help='score function for parags')
    mode_settings.add_argument('parag_selectMode', choices=["q"], default='q', help='score(v.) parag between q or a')
    
    path_settings = parser.add_argument_group('path settings')

    #TODO--for 232
    # path_settings.add_argument('--demo_files', nargs='+',
    #                            default=['../data/preprocessed/demo/trainset/search.train.json'],
    #                            help='list of files that contain the preprocessed demo baidu search train data')

    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/preprocessed/trainset/search.train.json', '../data/preprocessed/trainset/zhidao.train.json'],
                               help='list of files that contain the preprocessed train data')
    
    # path_settings.add_argument('--dev_files', nargs='+',
    #                            default=['../data/preprocessed/devset_v1/search.dev.json','../data/preprocessed/devset_v1/zhidao.dev.json'],
    #                            help='list of files that contain the preprocessed dev data')
    
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/preprocessed/devset/search.dev.json','../data/preprocessed/devset/zhidao.dev.json'],
                               help='list of files that contain the preprocessed dev data')

    path_settings.add_argument('--dev_sample_path', nargs='+',
                               default=['manual_analysis/100sample/100.search.dev.txt','manual_analysis/100sample/100.zhidao.dev.txt'])

    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/preprocessed/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')

    ## paths for LR predicting
    path_settings.add_argument('--dev_featurePaths', nargs='+',
                               default=['../data/preprocessed/devset/search.dev_ml_feature_data.json'])
    
    path_settings.add_argument('--dev_qid2featLidxPaths', nargs='+',
                               default=['../data/preprocessed/devset/search.dev_ml_qid2featLidx.json'])

    path_settings.add_argument('--dev_quesMatchML_top5Paths', nargs='+',
                               default=['../data/preprocessed/devset/search.dev_quesMatchML_top5Pids.json'])

    path_settings.add_argument('--test_featurePaths', nargs='+',
                               default=['../data/preprocessed/testset/search.test_ml_feature_data.json'])
    
    path_settings.add_argument('--test_qid2featLidxPaths', nargs='+',
                               default=['../data/preprocessed/testset/search.test_ml_qid2featLidx.json'])

    path_settings.add_argument('--test_quesMatchML_top5Paths', nargs='+',
                               default=['../data/preprocessed/testset/search.test_quesMatchML_top5Pids.json'])

    path_settings.add_argument('--LR_modeldir', default='manual_analysis/quesMatch_ml_models/', help='containing SC model and LR model')

    parser.add_argument('-n', '--n_processes', type=int, default=8,
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
    line = params
    sample = json.loads(line)
    
    #firstly score parag
    # print('start deal with parags...')
    qid2featLidx, feature_datas=scoreParag_ml(sample, tfidfObj)

    return json.dumps(qid2featLidx, ensure_ascii=False), json.dumps(feature_datas)

if __name__ == '__main__':
    args=parse_args()
    print(args)

    print('开始计时....')
    start = time.time()

    print('LR 预测不需要构建TF-IDF词典')
    # #全数据集计算tfidf权重
    # if args.parag_scoreFunc in ['tfidf', 'ml']:
    #     genet_tfidfObj(args)
    #     print('tf-idf vocabulary build done...')

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

    # #dev_file
    # print('start deal with devset...')
    # for dev_file, sample_path in zip(args.dev_files, args.dev_sample_path):
    #     import linecache
    #     sampled_qids = linecache.getline(sample_path, 2).strip().split()
        
    #     total_sampled=0
        
    #     i = 0
    #     work = list()
    #     with open(dev_file) as f_in:
    #         (filepath, tempfilename) = os.path.split(dev_file)
    #         (dev_filename, extension) = os.path.splitext(tempfilename)
    #         out_filename_qid2featLidx = dev_filename+'_'+args.parag_scoreFunc+'_qid2featLidx'+extension
    #         out_filename_feature_data = dev_filename+'_'+args.parag_scoreFunc+'_feature_data'+extension
    #         out_file_qid2featLidx_path=os.path.join(filepath, out_filename_qid2featLidx)
    #         out_file_feature_data_path=os.path.join(filepath, out_filename_feature_data)
    #         with open(out_file_qid2featLidx_path, 'w') as f_out_qid2featLidx:
    #             with open(out_file_feature_data_path, 'w') as f_out_feature_data:
    #                 with multiprocess.Pool(args.n_processes) as pool:
    #                     for line in f_in:
    #                         sample=json.loads(line)
    #                         if str(sample['question_id']) not in sampled_qids:
    #                             continue
    #                         print(sample['question_id'],'in sampling...')
    #                         if i < 10:#TODO 5000
    #                             work.append(line)#倒数第一参数，决定是否计算fake span
    #                             i += 1
    #                         else:
    #                             pool_res = pool.map(score_func, work)
    #                             total_sampled+=len(pool_res)
    #                             f_out_qid2featLidx.write('\n'.join([item[0] for item in pool_res])+'\n')
    #                             f_out_feature_data.write('\n'.join([item[1] for item in pool_res])+'\n')

    #                             work=list()
    #                             work.append(line)
    #                             i = 1
                                
    #                     if i > 0:#处理最后一批
    #                         pool_res = pool.map(score_func, work)
    #                         total_sampled+=len(pool_res)
    #                         f_out_qid2featLidx.write('\n'.join([item[0] for item in pool_res]))
    #                         f_out_feature_data.write('\n'.join([item[1] for item in pool_res]))
    #     print('done with 1 file')
    #     print('total_sampled',total_sampled)
    #     time_elapsed = time.time()-start
    #     print('Training complete in {:.0f}min-{:.0f}s'.format(
    #         time_elapsed // 60, time_elapsed % 60)) # 打印出来时间

    # time_elapsed = time.time()-start
    # print('Training complete in {:.0f}min-{:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    # print('deal with devset done')

    # #test_file
    # print('start deal with testset...')
    # for test_file in args.test_files:
    #     i = 0
    #     work = list()
    #     with open(test_file) as f_in:
    #         (filepath, tempfilename) = os.path.split(test_file)
    #         (test_filename, extension) = os.path.splitext(tempfilename)
            
    #         out_filename_qid2featLidx = test_filename+'_'+args.parag_scoreFunc+'_qid2featLidx'+extension
    #         out_filename_feature_data = test_filename+'_'+args.parag_scoreFunc+'_feature_data'+extension
    #         out_file_qid2featLidx_path=os.path.join(filepath, out_filename_qid2featLidx)
    #         out_file_feature_data_path=os.path.join(filepath, out_filename_feature_data)
    #         with open(out_file_qid2featLidx_path, 'w') as f_out_qid2featLidx:
    #             with open(out_file_feature_data_path, 'w') as f_out_feature_data:
    #                with multiprocess.Pool(args.n_processes) as pool:
    #                     for line in f_in:
    #                         if i < 5000:
    #                             work.append(line)
    #                             i += 1
    #                         else:
    #                             pool_res = pool.map(score_func, work)
    #                             f_out_qid2featLidx.write('\n'.join([item[0] for item in pool_res])+'\n')
    #                             f_out_feature_data.write('\n'.join([item[1] for item in pool_res])+'\n')

    #                             work=list()
    #                             work.append(line)
    #                             i = 1
                                
    #                     if i > 0:#处理最后一批
    #                         pool_res = pool.map(score_func, work)
    #                         f_out_qid2featLidx.write('\n'.join([item[0] for item in pool_res]))
    #                         f_out_feature_data.write('\n'.join([item[1] for item in pool_res]))

    #     print('done with 1')
    #     time_elapsed = time.time()-start
    #     print('Training complete in {:.0f}min-{:.0f}s'.format(
    #         time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    
    # time_elapsed = time.time()-start
    # print('Training complete in {:.0f}min-{:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    # print('deal with testset done')

    ###*********************************************************************###
    #100个样本训练好的LR模型对开发集和测试集基于LR模型利用多个特征选择篇章【只用于选search数据】

    # #dev_file
    # print('start deal with devset...')
    # for dev_featurePath, dev_qid2featLidxPath, dev_quesMatchML_top5Path in zip(args.dev_featurePaths, args.dev_qid2featLidxPaths, args.dev_quesMatchML_top5Paths):
    #     lr_predict_forTOP5(dev_featurePath, dev_qid2featLidxPath, args.LR_modeldir, dev_quesMatchML_top5Path)

    #     print('done with 1 file')
    #     time_elapsed = time.time()-start
    #     print('Training complete in {:.0f}min-{:.0f}s'.format(
    #         time_elapsed // 60, time_elapsed % 60)) # 打印出来时间

    # time_elapsed = time.time()-start
    # print('Training complete in {:.0f}min-{:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    # print('deal with devset done')

    #test_file
    print('start deal with testset...')
    for test_featurePath, test_qid2featLidxPath, test_quesMatchML_top5Path in zip(args.test_featurePaths, args.test_qid2featLidxPaths, args.test_quesMatchML_top5Paths):
        lr_predict_forTOP5(test_featurePath, test_qid2featLidxPath, args.LR_modeldir, test_quesMatchML_top5Path)

        print('done with 1 file')
        time_elapsed = time.time()-start
        print('Training complete in {:.0f}min-{:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)) # 打印出来时间

    time_elapsed = time.time()-start
    print('Training complete in {:.0f}min-{:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    print('deal with testset done')
