#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: TF_AlexNet.py 
@desc: 基于TF的AlexNet实现
参考代码:  https://github.com/charlesXu86/models/tree/master/tutorials/image/alexnet
@time: 2017/12/21 
"""

import math
import time
import tensorflow as tf

from datetime import datetime

batch_size = 32
num_batches = 100

# 定义一个函数，显示网格每一层结构，展示每一个卷积层或是池化层输出tensor的尺寸
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())