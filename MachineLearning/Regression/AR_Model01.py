#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: AR_Model01.py 
@desc:
@time: 2017/12/07 
"""
import numpy as np
import matplotlib.pylab as plt

if __name__=='__main__':
    price = np.loadtxt('F:\project\MachineLearning\MachineLearning\Data\SH600000.txt', delimiter='\t',skiprows=2, usecols=(4,))
    print('原始价格: \n', price)
    n = 100       # 阶数
    y = price[n:]
    m = len(y)    # 样本个数
    print('预测价格: \n',y)
    x = np.zeros((m, n+1))
    for i in range(m):
        x[i] = np.hstack((price[i:i+n], 1))   # 水平（按列顺序）堆叠数组。
    print('自变量: \n',x)
    theta = np.linalg.lstsq(x, y)[0]                 # 将最小二乘解返回到线性矩阵方程。
    print(theta)
    plt.show(theta,x,y)
