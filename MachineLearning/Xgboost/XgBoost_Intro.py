#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: XgBoost_Intro.py 
@desc: XgBoost学习笔记，入门
官网文档：http://xgboost.readthedocs.io/en/latest/get_started/index.html
@time: 2017/12/18 
"""

import xgboost as xgb
import numpy as np

# XgBoost的基本使用
# 自定义损失函数的梯度和二阶导,其损失函数是泰勒展开式的二项逼近
# 对树的结构进行了正则化约束，防止模型过度复杂，降低了过拟合的可能性
# xgboost的并行是在特征粒度上的。
# 我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），
# xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。
# 这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，
# 那么各个特征的增益计算就可以开多线程进行。


def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h

def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)

if __name__=='__main__':
    # 读取数据
    data_train = xgb.DMatrix('F:\project\MachineLearning\MachineLearning\Data\XgBoost\\agaricus_train.txt')
    data_test = xgb.DMatrix('F:\project\MachineLearning\MachineLearning\Data\XgBoost\\agaricus_test.txt')
    # print(data_train)
    # print(type(data_train))

    # 设置参数
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_around = 3
    bst = xgb.train(param, data_train, num_boost_round=n_around, evals=watchlist, obj=log_reg, feval=error_rate)

    # 计算错误率
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print(y_hat)
    print(y)

    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print('样本总数: \t', len(y_hat))
    print('错误总数: \t%4d' % error)
    print('错误率: \t%.5f%%' % (100* error_rate))