#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: bnlstm.py 
@desc: Batch Normalization lstm
        https://github.com/OlavHN/bnlstm/blob/master/lstm.py
@time: 2017/10/26 
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

