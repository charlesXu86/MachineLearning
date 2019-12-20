#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: tags.py 
@desc: 基于结巴的TF-IDF关键词提取
@time: 2017/10/27 
"""

from jieba import analyse

tfidf = analyse.extract_tags  # 引入TF-IDF关键词抽取接口

# 原始文本

# text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
#         是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
#         线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\
#         线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\
#         同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"


with open('tags.txt', encoding='utf-8') as f:

    keywords = tfidf(f)
    print("Keywords By tf-idf")

    for keyword in keywords:
        print(keyword + "/")