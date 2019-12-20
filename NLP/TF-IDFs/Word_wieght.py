#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Word_wieght.py 
@desc: 关键词权重计算  基于TF-IDF
@time: 2017/11/09 
"""

import jieba
import jieba.posseg as pseg
import sys
import os

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# words = pseg.cut("对这句话进行分词")
# for key in words:
#     print(key.word, key.flag)

if __name__=='__main__':
    corpus = ["我 来到 北京 清华大学",
              "我 现在 在 杭研 大厦 工作",
              "我 爱 北京 天安门"]
    vectorize = CountVectorizer() # 该类会将文本中的词语转换为词频矩阵，矩阵a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer() # 统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorize.fit_transform(corpus)) # 第一个是计算tf-idf，第二个是将文本转换为词频矩阵
    word = vectorize.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()   # 将tf-idf矩阵抽出来，元素a[i][j] 表示j词在i类文本中的tf-idf权重
    for i in range(len(weight)):
        print('这里输出第', i + 1, '类文本的权重。')
        for j in range(len(word)):
            print(word[j], weight[i][j])