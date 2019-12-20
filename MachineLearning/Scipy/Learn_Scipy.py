#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Learn_Scipy.py
@time: 2018/2/7 13:50
@desc:
"""

import scipy as sc
import numpy as np
from scipy.sparse import coo_matrix
from scipy.interpolate import UnivariateSpline

'''
   Scipy学习笔记 
   文档:
     http://python.usyiyi.cn/translate/scipy_lecture_notes/index.html
'''
# 1,稀疏矩阵
# 创建coo矩阵  协调格式
mtx = coo_matrix((3, 4), dtype=np.int8)
dense = mtx.todense()
print(dense)

# CSR 压缩稀疏行格式

# CSC 压缩稀疏列格式

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
# UnivariateSpline（x，y，w = None，bbox = [None，None]，k = 3，s = None，ext = 0，check_finite = False ）[source]
# 一维平滑样条拟合一组给定的数据点。
# 参数说明: x
