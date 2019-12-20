#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: learpython1.py
@desc: python学习1 基础
@time: 2017/10/18 
"""
import pprint
import redis
import pprint
# from __future__ import division

from unittest import result

'''

jfskfhskfhdsfhdkdsf
fdsfsdfsfsfsdfsdfdsfsdfsdfds

'''
# print('hello world')
# for i in range(1,10, 2):
#     a = i
#     if a > 5:
#         a = 10
# a, b, c = 1,2,3
# print(a, b, c)

# a = "哈哈哈"
# pprint.pprint()

# dict = {}
# print(dict)

file = open('E:\py_workspace\MachineLearning\DeepLearning\Data\sensitive.txt', 'r', encoding='utf-8')

files = open('E:\py_workspace\MachineLearning\DeepLearning\Data\sensitives.txt', 'w')
for line in file.readlines():
    word = line.strip().split("\t")[-1]
    pprint.pprint(word)
    files.write(word + '\n')
files.close()





