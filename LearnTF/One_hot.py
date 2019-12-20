#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: One_hot.py 
@desc: tf.one_hot()函数
@time: 2017/11/18 
"""

import tensorflow as tf
import numpy as np

'''
  tensorflow中tf.one_hot()函数的作用是将一个值化为一个概率分布的向量，一般用于分类问题。
'''

SIZE = 6
CLASS = 8

label = tf.constant([0,1,2,3,4,5,6,7])
sess = tf.Session()
print('label:', sess.run(label))
b = tf.one_hot(label,CLASS,1,0, axis=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(b)
    print('After ont-hot',sess.run(b))