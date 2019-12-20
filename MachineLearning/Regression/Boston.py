#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Boston.py
@desc: 波士顿房价预测
       http://blog.csdn.net/weixin_36627946/article/details/70240328
@time: 2017/12/11 
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
# import excep
import sklearn.datasets

from sklearn.metrics import mean_squared_error
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
def not_empty(s):
    return s != ''

if __name__=='__main__':
    warnings.filterwarnings(action='ignore')
    np.set_printoptions(suppress=True)  # 设置打印选项，这些选项确定显示浮点数，数组和其他Numpy对象的方式
                                        # suppress 是否使用科学计数法抑制小浮点值的打印，默认为False
    file_data = pd.read_csv('F:\dataset\MachineLearning\MachineLearning\Data\Regression\housing.csv', header=None)
    prices = file_data['MEDV']
    features = file_data.drop('MEDV', axis=1)
    print('Boston housing dataset has {} data points with {} variables each.'.format(*file_data.shape))
    # data = np.empty((len(file_data), 14))
    # print(data)
    # for i, d in enumerate(file_data.values):
    #     d = map(float, filter(not_empty, d[0].split(' ')))
    #     data[i] = d
    # x, y = np.split(data, (13, ), axis=1)
    # print('样本个数: %d, 特征个数: %d' % x.shape)
    # print(y.shape)