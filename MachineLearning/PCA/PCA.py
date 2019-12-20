#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: PCA.py
@desc: pca实现
@time: 2018/01/09 
"""
'''
 pca编程模型:
     1,去均值
     2,计算协方差矩阵及特征值和特征向量
     3,计算协方差矩阵的特征值大于阈值的个数
     4,降序排列特征值，去掉较小的特征值
     5,合并选择的特征值
     6,选择相应的特征值和特征向量
     7,计算白化矩阵
     8,提取主分量
 
'''
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    '''
      加载数据
    :param fileName: 文件名
    :param delim:  分隔符
    :return:
    '''
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=True)  # 估计协方差矩阵，给定数据和权重。协方差表示两个变量一起变化的水平。
    eigVals, eigVects = linalg.eig(mat(covMat))  # 计算特征值和特征向量
    eigValInd = argsort(eigVals)  # 特征值排序 从小到大
    eigValInd = eigValInd[: -(topNfeat + 1): -1] # 去掉不要的维度
    redEigVects = eigVects[:, eigValInd]  # 从大到小
    lowDataMat = meanRemoved * redEigVects # 将数据转换成新的维度
    reconMat = (lowDataMat * redEigVects.T) + meanVals
    return lowDataMat, reconMat


def plotBestFit(dataSet1, dataSet2):
    '''
     绘图
    :param dataSet1:
    :param dataSet2:
    :return:
    '''
    dataArr1 = array(dataSet1)
    dataArr2 = array(dataSet2)
    n = shape(dataArr1)[0]
    m = shape(dataArr2)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    xcord3 = []; ycord3 = []
    j = 0
    for i in range(n):
        xcord1.append(dataArr1[i, 0]); ycord1.append(dataArr1[i, 1])
        xcord2.append(dataArr2[i, 0]); ycord2.append(dataArr2[i, 1])

    fit = plt.figure()
    ax = fit.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()




if __name__=='__main__':
    data = loadDataSet('F:\project\MachineLearning\MachineLearning\Data\PCA\\testSet.txt')
    a, b = pca(data, 2)
    plotBestFit(a, b)