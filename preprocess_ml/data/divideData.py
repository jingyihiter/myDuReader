#!/usr/bin/env python
# _*_ coding:utf-8 _*_

#五折交叉验证数据集划分
import sys

def diviede(filename, trainfile, testfile, ith):
    """

    :param filename:
    :param ith:
    :return:
    """
    with open(filename, "r", encoding="utf-8") as f:
        with open(trainfile, "w", encoding="utf-8") as trf:
            with open(testfile, "w", encoding="utf-8") as tef:
                for i,line in enumerate(f):
                    # if i<(ith)*20 and i>=(ith-1)*20: 1:4
                    if i < (ith) * 40 and i >= (ith - 1) * 40: # 4:6
                        tef.write(line)
                    else:
                        trf.write(line)


def main():
    #  python divideData.py 100.zhidao.dev.json zhidao_82/zhidao.train.json zhidao_82/zhidao.test.json
    tarfile = sys.argv[1]
    trafile = sys.argv[2]
    tesfile = sys.argv[3]
    [diviede(tarfile, trafile+"_"+str(i), tesfile+"_"+str(i), i) for i in range(1,6)]


if __name__=="__main__":
    main()
