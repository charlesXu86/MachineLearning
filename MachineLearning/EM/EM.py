#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: EM.py 
@desc: EM 算法实现
@time: 2017/11/25 
"""

import numpy as np
import math
import copy
import matplotlib.pyplot as plt

isdebug = True

# 参考文献：机器学习TomM.Mitchell P.137
#          李航 统计学习方法EM算法章节
# 代码参考http://blog.csdn.net/chasdmeng/article/details/38709063


# 指定k个高斯分布参数， 这里指定 k = 2， 这两个高斯分布具有相同的方差sigmma，均值分别为mu1, mu2
def init_data(Sigma, Mu1, Mu2, k, N):
    global X
    global Mu
    global Expectations
    X = np.zeros((1, N))
    Mu = np.random.random(k)
    Expectations = np.zeros((N,k))

    for i in range(0, N):
        if np.random.random(1) > 0.5:
            X[0,i] = np.random.normal(Mu1, Sigma)
        else:
            X[0,i] = np.random.normal(Mu2, Sigma)
    if isdebug:
        print("**************")
        print("初始化观测数据X:")
        print(X)

# EM 算法： E步： 计算E
def e_step(Sigma, k, N):
    global Expectations
    global Mu
    global X
    for i in range(0, N):
        Denom = 0
        Numer = [0.0] * k
        for j in range(0,k):
            Numer[j] = math.exp((-1/(2*(float(Sigma * 2)))) * (float(X[0,i] - Mu[j])) ** 2)
            Denom += Numer[j]
        for j in range(0,k):
            Expectations[i,j] = Numer[j] / Denom
    if isdebug:
        print("******************")
        print("隐藏变量E(Z):")
        print(Expectations)

#　EM算法： M步： 求最大化参数Mu
def m_step(k, N):
    global Expectations
    global X
    for j in range(0, k):
        Numer = 0
        Denom = 0
        for i in range(0, N):
            Numer += Expectations[i, j] * X[0,i]
            Denom += Expectations[i,j]
        Mu[j] = Numer / Denom

# 迭代
def run(Sigma, Mu1, Mu2, k, N, iter_num, epsilon):
    init_data(Sigma, Mu1, Mu2, k, N)
    print("初始化<u1,u2>:", Mu)
    for i in range(iter_num):
        Old_Mu = copy.deepcopy(Mu)
        e_step(Sigma, k, N)
        m_step(k, N)
        print(i, Mu)
        if sum(abs(Mu - Old_Mu)) < epsilon:
            break

if __name__=='__main__':
    sigma = 6   # 高斯分布具有相同的方差
    mu1 = 40    # 第一个高斯分布的均值
    mu2 = 20
    k = 2       # 高斯分布的个数
    N = 10000   # 样本个数
    iter_num = 1000   # 最大迭代次数
    epsilon = 0.0001  # 当两次误差小于这个数时退出
    run(sigma, mu1, mu2, k, N, iter_num, epsilon)

    plt.hist(X[0, :], 50)
    plt.show()