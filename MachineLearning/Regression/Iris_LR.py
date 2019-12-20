#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Iris_LR.py 
@desc: 鸢尾花数据回归分类
@time: 2017/12/12 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

if __name__=='__main__':
    path = 'F:\dataset\MachineLearning\MachineLearning\Data\Regression\iris.data'
    data = pd.read_csv(path, header=None)
    # print(data)
    iris_types = data[4].unique()
    for i, type in enumerate(iris_types):
        data.set_value(data[4] == type, 4, i)    # 数据预处理，将鸢尾花的种类转换成对应的数字
    x, y = np.split(data.values, (4, ), axis=1)
    x = x.astype(np.float)
    y = y.astype(np.int)
    x = x[:, :2]   # 仅使用前两列特征
    lr = Pipeline([('sc', StandardScaler()),
                   ('clf', LogisticRegression())])
    lr.fit(x, y.ravel())
    y_hat = lr.predict(x)
    y_hat_prob = lr.predict_proba(x)
    np.set_printoptions(suppress=True)
    print('y_hat = \n', y_hat)
    print('y_hat_prob = \n', y_hat_prob)
    print('准确度: %.2f%%' % (100 * np.mean(y_hat == y.ravel())))

    # 绘图
    N, M = 500, 500  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()   # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()   # 第一列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)   # np.linspace() 在指定的间隔内返回均匀间隔的数字，返回在间隔[开始，停止]上计算的num个均匀间隔的样本
    x1, x2 = np.meshgrid(t1, t2)          # np.meshgrid() 从坐标向量返回坐标矩阵
    x_test = np.stack((x1.flat, x2.flat), axis=1)    # np.stack沿着新轴连接数组的序列，

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    y_hat = lr.predict(x_test)     # 预测值
    y_hat = y_hat.reshape(x1.shape)   # 使之与输入的形状相同

    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
    plt.xlabel(u'花萼长度', fontsize=14)
    plt.ylabel(u'花萼宽度', fontsize=14)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.title(u'鸢尾花Logistic回归分类效果 - 标准化', fontsize=17)
    plt.show()
