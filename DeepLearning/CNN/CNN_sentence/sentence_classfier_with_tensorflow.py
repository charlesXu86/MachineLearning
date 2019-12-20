#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: sentence_classfier_with_tensorflow.py 
@desc:
@time: 2017/10/06 
"""
import pickle
import numpy as np
from collections import defaultdict, OrderedDict
import warnings
import re
import sys
import time
import os
import datetime
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.contrib import learn

"""
  使用tf构建cnn模型进行文本分类
"""
vector_size = 100
sentence_length = 100
hidden_layer_input_size = 100
filter_hs = [3,4,5]
num_filters = 128
img_h = sentence_length
img_w = vector_size
filter_w = img_w
batch_size = 100
word_idx_map_size = 75924

# 数据预处理
def get_idx_from_sent(sent, word_idx_map, max_i):
    '''
    
    :param sent: 
    :param word_idx_map: 
    :param max_i: 
    :return: 
    '''
