# -*- coding: utf-8 -*-

# Adaboost 工具类
import numpy as np
import scipy as sp
import warnings

def sign(x):
    q = np.zeros(np.array(x).shape)
    q[x>=0] = 1
    q[x<0] = -1
    return q
