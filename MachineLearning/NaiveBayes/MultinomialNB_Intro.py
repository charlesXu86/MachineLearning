#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: MultinomialNB_Intro.py 
@desc: 多项式朴素贝叶斯实现
@time: 2017/12/24 
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

if __name__=='__main__':
    np.random.seed(0)
    M = 200
    N = 1000
    x = np.random.randint(2, size=(M,N))    # [Low, High]
    x = np.array(list(set([tuple(t) for t in x])))
    M = len(x)
    y = [0, 1, 2] * (int)((float(M) / 3) + 1)
    y = np.array(y[0:M])
    print('样本个数：%d，特征数目：%d' % x.shape)
    print('样本：\n', x)
    # mnb = MultinomialNB(alpha=1)  # 动手：换成GaussianNB()试试预测结果？
    mnb = GaussianNB()
    mnb.fit(x, y)
    y_hat = mnb.predict(x)
    print('预测类别：', y_hat)
    print('准确率：%.2f%%' % (100 * np.mean(y_hat == y)))
    print('系统得分：', mnb.score(x, y))