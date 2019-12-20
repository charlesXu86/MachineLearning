# -*- coding: utf-8 -*-

import numpy as np
# import pylab


'''
  Numpy 官方文档 ：https://yiyibooks.cn/xx/NumPy_v111/genindex.html
'''

'''
  Numpy中的histogram函数应用到一个数组返回一对变量: 直方图数组和箱式向量。注意：matplotlib也有一个hist。
  主要的差别就是pylab是自动绘制直方图，而numpy.histigram仅仅产生数据。
'''
# mu, sigma= 2, 0.5
# v = np.random.normal(mu, sigma, 10000)
# # Plot a normalized histogram with 50 bins
# pylab.hist(v, bins = 50, normed=1)
# pylab.show()
#
# # Compute the histogram with numpy and then plot it
# (n, bins) = np.histogram(v, bins=50, normed=True)
#
# pylab.plot(.5*(bins[1:] + bins[:-1]), n)
# pylab.show()

# =========================
'''
  numpy.transpose(a,axis=None)
  @:param: a 输入数组
          axes : ints列表，可选，默认情况下，反转尺寸，否则根据给定的值替换轴。
  @:return : ndarray

'''
'''
 Numpy 的函数方法操作
   创建数组：arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace,
            logspace, mgrid, ogrid, ones, ones_like, r, zeros, zeros_like

   转化：astype, atleast 1d, atleast 2d, atleast 3d, mat

   操作：array split, column stack, concatenate, diagonal, dstack, hsplit, hstack, item, neaxis, ravel,
        repeat， reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack

   询问：all, any, nonzero, where

   排序：argmax, argmin, argsort, max, min, ptp, searchsorted, sort

   运算：choose, compress, cumprod, cumsum, inner, fill, imag, prod, put, putmask, real, sum

   基本统计: cov, mean, std, var

   基本线性代数：cross, dot, outer, svd, vdot
'''

# 1, empty_like ： 返回具有与给定数组相同的形状和类型的新数组
#  此函数不返回初始化的数组；可以使用zero_like 或 ones_like来代替。他可能稍微快于设置数组值的函数

# a = ([1,2,3],[4,5,6])
# b = np.empty_like(a)
# print(b)

# 2, zero_like: 返回具有与给定数组相同形状和类型相同的零数组
# x = np.arange(6)
# x = x.reshape((2,3))
# print(x)
# y = np.zeros_like(x)
# print(y)

# 3, ones_like: 返回具有与给定数组相同形状和类型相同的数组 (全为1)

# 4, eye(M, N, K, dtype) 返回一个2-D数组，其中一个在对角线上，零在其他地方.
#      M 输出中的行数。 N列数  K对角线索引:0(默认)是指主对角线，正值是指上对角线， 负值是指下对角线
# x = np.eye(4,k=2)
# print(x)

# 5, identity 返回一个正方形数组，在主对角线上有一个数组
# x = np.identity(4)
# print(x)

# 6,logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None) 返回以对数刻度均匀分布的数字
# x = np.logspace(2.0, 3.0, num=5)

# 7,np.random.randint  将随机正式从低(包括)返回到高(不包括)
# x = np.random.randint(10, size=[3,3])
# print(x)

# 8,np.random.random_integers 底和高之间的np.int类型的随机整数(含)

# 9, np.random.randn 从标准正太分布返回样本  对于N(\mu, \sigma^2)的随机样本，使用：
#                                        sigma * np.random.randn（...） + t5>
#    如果想要一个以元组作为第一个参数的接口，使用standard_normal
#  来自N(3, 6.25)的2×4数组样本：
# x = 2.5*np.random.randn(2, 4) + 3
# y = np.random.randn()
# print(x, y)

# 10, np.random.standard_normal


# 11, np.squeeze(a, axis=None) 从数组的形状中删除单维条目
# x = np.array([[[[1], [1], [2], [3]]]])
# print(x.shape)

# 12, np.rollaxis(a, axis, start=0) 向后滚动指定的轴，直到他位于给定位置
# a = np.ones((3,4,5,6))
# m = np.rollaxis(a, 3, 0).shape
# n = np.rollaxis(a,2).shape
# print(m,n)

# 13, np.concatenate((a1,a2,.....), axis=0)   沿现有轴连接数组序列
# a = np.array([[1,2], [3,4]])
# b = np.array([[5,6]])
# x = np.concatenate((a, b), axis=0)
# y = np.concatenate((a,b.T), axis=1)
# print(x, y)

# 14,np.random.permutation(x) 随机置换序列，或返回置换范围， 如果x是多维数组，则他只沿其第一个索引进行重排。
#     参数：x    如果x是整数，则随机置换np.array(x). 如果x是数组，请复制并随机移动元素

# 15, np.random.choice(a, size=None, repace=True, p=None)  从给定的1-D数组生成随机样本
# x = np.random.choice(10,4)
# print(x)

# 16, np.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None)       设置打印选项，这些选项确定显示浮点数，数组和其他Numpy对象的方式
#     参数:

# 17, np.empty()   返回给定形状和类型的新数组，而不初始化条目
#      参数:  shape   空数组的形状
#            dtype:   数据类型
#            order   C  F   是否在存储器中以行为主(C) 或列主(Fortran风格)顺序存储多维风格
# y = np.empty([2,2])
# print(y)

# 18 np.linspace()
#    np.logspace()   样本均匀分布在对数空间中

# 19 np.meshgrid()    返回网格矩阵

# 20 np.stack(arrays, axis=0)   沿着新轴连接数组的序列， axis参数指定在结果尺寸中的索引，例如 axis=0,他将是第一个维度，
#                                                                                       axis=-1,他将是最后一个维度

# 21 np.random.binomial(n,p,size=None)  从二项分布绘制样本
#                        样本从具有指定参数，n个实验和p成功概率的二项分布中得到，其中n是>=0的整数，p在[0,1]之间（n可以作为float输入，但在使用中被截断为整数）

# 22 np.split(ary, indices_or_sections, axis=0) 将数组拆分为多个子数组， 相关的方法： indices_or_sections如果是一个数组，则按数组的索引切分
#    array_split   将数组拆分成等于或接近相等大小的多个子数组，如果不能进行等分，不会引发异常
#    hsplit   水平(逐列)将数组拆分成多个子数组
#    vsplit   垂直
#    dsplit   将数组沿着第3轴(深度)拆分成多个子数组
#  ==========
#  concatenate  沿现有轴连烈数组序列
#  stack  沿着新轴连接数组的序列        hstack    vstack    dstack
# x = np.arange(9.0)   在给定范围内返回均匀间隔的值
# y = np.split(x, [4,8])
# print(x, y)

# 23 np.eye(N, M=None, k=0, dtype=<type 'float'>)  返回一个2-D数组, k是对角线索引

# 24 np.unique() 查找数组的唯一元素。
#             返回数组的排序的唯一元素。除了唯一元素外，还有三个可选输出: 提供唯一值的输入数组的索引
#                                                              重建输入数组的唯一数组的索引
#                                                              以及输入数组中每个唯一值的出现次数
# x = np.unique([1,1,2,2,3,3,4,5,4])
# print(x)

# 25 np.copy()  返回给定对象的数组副本
#   相当于 np.array(a, copy=True)

# 26 np.expm1()  对数组中的所有元素计算 exp(x) - 1
#    参考 log1p   log(1 + x)

# 27 np.percentile 沿指定轴计算数据的第q个百分位数 返回数组元素的第Q个百分位数

# 28 np.full(shape, fill_value, dtype=None, order='C') 返回一个给定的维度和类型，值均已fill_value填充
# x = np.full((2,2), np.inf)
# y = np.full((2,2), 10, dtype=np.int)
# print(x, y)

# 29 np.random.permutation



