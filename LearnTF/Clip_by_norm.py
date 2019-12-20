#-*- coding:utf-8 _*-  
""" 
@Author:charlesXu
@File: Clip_by_norm.py 
@Desc: 
@Time: 2017/11/21 9:59 
"""

import tensorflow as tf
import numpy as np


'''
  clip_by_norm是指对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题，是一种比较常用的梯度规约方式
  
  他的作用在于将传入的梯度张量t的L2范数进行了上限约束，约束值为clip_norm,如果t的L2范数超过了clip_norm,
  则变换为 t*clip /l2norm(t),如此一来，变换后的t的L2范数便小于等于clip_norm了。
'''

# optimizer = tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.5)
# grads = optimizer.compute_gradients(optimizer)
# for i, (g, v) in enumerate(grads):
#     if g is not None:
#         grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
# train_op = optimizer.apply_gradients(grads)


# t = np.random.randint(low=0,high=5,size=10)


