#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: extract_tags.py 
@desc: 基于jieba的关键词提取
       参考github连接: https://github.com/charlesXu86/jieba
@time: 2017/10/27 
"""

import jieba
import jieba.analyse


'''
   基于 TF-IDF 算法的关键词抽取
'''

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:python extract_tags.py [file name] -k [top k]"

parser = OptionParser(USAGE)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()


if len(args) < 1:
    print(USAGE)
    sys.exit(1)

file_name = args[0]

if opt.topK is None:
    topK = 10
else:
    topK = int(opt.topK)

content = open(file_name, 'rb').read()

tags = jieba.analyse.extract_tags(content, topK=topK)

print(",".join(tags))