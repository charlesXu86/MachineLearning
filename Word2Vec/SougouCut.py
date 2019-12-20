#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: SougouCut.py 
@desc: 搜狗新闻分词
@time: 2017/11/07 
"""

import jieba
import jieba.analyse
import jieba.posseg as pseg


def cut_words(sentence):
    return " ".join(jieba.cut(sentence)).encode('utf-8')
