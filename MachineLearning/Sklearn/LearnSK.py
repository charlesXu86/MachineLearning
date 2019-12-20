#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: LearnSK.py 
@desc: sklearn学习笔记
@time: 2017/12/12 
"""
'''
  sklearn文档:
      http://cwiki.apachecn.org/pages/viewpage.action?pageId=10030193
'''

from sklearn import datasets
import numpy as np

boston = datasets.load_boston()
# print(boston.DESCR)

housing = datasets.fetch_california_housing()
# print(housing.DESCR)

X, y = boston.data, boston.target  # 用data属性连接数据中包含自变量Numpy数组，用target属性连接数据中的因变量
# print(X, y)


path = datasets.get_data_home()   # 获取数据下载的位置
# print(path)


# 创建回归数据集
reg_data = datasets.make_regression()
a, b = reg_data[0].shape, reg_data[1].shape
print(a, b)