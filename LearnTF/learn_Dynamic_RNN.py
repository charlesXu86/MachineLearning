#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: learn_Dynamic_RNN.py 
@desc: 学习Dynamic_RNN
       http://blog.csdn.net/u010223750/article/details/71079036
@time: 2017/11/17 
"""

import tensorflow as tf
import numpy as np

X = np.random.randn(2, 10, 8)

X[1, 6:] = 0
X_lenghts = [10, 6]

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64, state_is_tuple=True)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lenghts,
    inputs=X
)
result = tf.contrib.learn.rnn_n(
    {"outputs": outputs, "last_states": last_states},
    n = 1,
    feed_dict = None
)

print(result[0])

assert result[0]["outputs"].shape == (2, 10, 64)
assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()