#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Sklearn_SVM_Demo.py 
@desc: Sklearn_SVM_Demo
      Sklearn_SVM 译文链接: http://cwiki.apachecn.org/pages/viewpage.action?pageId=10031359
@time: 2018/01/01 
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

print(__doc__)

# 创建40个分离点
np.random.seed(0)


def loadDataSet(fileName):
    '''
      对文件进行逐行解析，得到行的类标签和整个数据矩阵
    :param fileName: 文件名
    :return:
          dataMat: 数据矩阵
          labelMat: 类标签
    '''
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

X, Y = loadDataSet('F:\project\MachineLearning\MachineLearning\Data\SVM\\testSet.txt')
X = np.mat(X)   # np.mat(data, dtype=None)，输入解释为矩阵。与matrix不同，如果输入已经是矩阵，则matrix不会进行复制。

print('X=', X)
print('Y=', Y)

# 拟合一个SVM模型
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 获取分割超平面
'''
  coef_ 存放回归系数
  intercept_ 存放截距
'''
w = clf.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-2, 10)    # 在指定的间隔内返回均匀间隔的数字。
# 二维的直线方程
yy = a * xx - (clf.intercept_[0]) / w[1]
print('yy=', yy)

# 通过支持向量绘制分割超平面
print('Support Vectors_=', clf.support_vectors_)
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
plt.scatter([X[:, 0]], [X[:, 1]], c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()

