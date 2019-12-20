#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: OLS.py 
@desc: 最小二乘法python实现， 博客地址:  http://blog.csdn.net/yhao2014/article/details/51491413
@time: 2017/08/30 
"""

import numpy as np
import scipy as sp
import pylab as pl
from scipy.optimize import leastsq  #引入最小二乘函数

n = 9 #多项式次数

#目标函数
def real_func(x):
    return np.sin(2 * np.pi * x)

#多项式函数
def fit_func(p, x):
    f = np.poly1d(p)   # 一维多项式类，用于封装对多项式的操作，多项式的系数，以降低的功率，或者如果第二个参数的值为True，多项式的根（多项式计算结果为0的值）。例如，poly1d（[1， 2， 3]）返回表示x^2 + 2x + 3 ，而poly1d（[1， 2， 3]， True）一个代表(x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x -6。
    return f(x)

#残差函数
regularization = 0.005 #正则化系数lambda, 需要慎重的选择正则化参数的选择
def residuals_func(p, y, x):
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(regularization) * p)
    return ret

x = np.linspace(0, 1, 9) #随机选择9个点作为x
x_points = np.linspace(0, 1, 1000)

y0 = real_func(x)
y1 = [np.random.normal(0, 0.1) + y for y in y0] #加大正太分布噪声后的分布函数

p_init = np.random.rand(n) #随机初始化多项式参数

plsq = leastsq(residuals_func, p_init, args=(y1, x))

print('Fittinf Parameters:', plsq[0])  #输出拟合参数

pl.plot(x_points, real_func(x_points), label='real')
pl.plot(x_points, fit_func(plsq[0], x_points), label='fitted curve')
pl.plot(x, y1, 'bo', label='with noise')
pl.legend()
pl.show()

