#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path = [os.path.join(ROOTDIR, "lib")] + sys.path

# Set your own model path
MODELDIR=os.path.join(ROOTDIR, "ltp_data")

from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller

paragraph = '平素体质:健康状况:良,既往有“高血压病史”多年'

sentence = SentenceSplitter.split(paragraph)[0]

segmentor = Segmentor()
segmentor.load("/home/yhli/ltp_data/ltp_data_v3.4.0/cws.model")
words = segmentor.segment(sentence)
print("\t".join(words))

postagger = Postagger()
postagger.load("/home/yhli/ltp_data/ltp_data_v3.4.0/pos.model")
postags = postagger.postag(words)
# list-of-string parameter is support in 0.1.5
# postags = postagger.postag(["中国","进出口","银行","与","中国银行","加强","合作"])
print("\t".join(postags))

parser = Parser()
parser.load("/home/yhli/ltp_data/ltp_data_v3.4.0/parser.model")
arcs = parser.parse(words, postags)

print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
for arc in arcs:
    print(arc.head)
    print(arc.relation)
recognizer = NamedEntityRecognizer()
recognizer.load("/home/yhli/ltp_data/ltp_data_v3.4.0/ner.model")
netags = recognizer.recognize(words, postags)
print("\t".join(netags))


segmentor.release()
postagger.release()
parser.release()
recognizer.release()