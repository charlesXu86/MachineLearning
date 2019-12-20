#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Exponential_decay.py 
@desc: 指数衰减法
@time: 2017/11/18 
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
在Tensorflow中，为解决设定学习率(learning rate)问题，提供了指数衰减法来解决。
通过tf.train.exponential_decay函数实现指数衰减学习率。
    步骤：1.首先使用较大学习率(目的：为快速得到一个比较优的解);
         2.然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定);

decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)  
  说明:decayed_learning_rate为每一轮优化时使用的学习率
      learning_rate:为事先设定好的初始学习率
      decay_rate:为衰减系数
      decay_step:为衰减速度。            
而tf.train.exponential_decay函数则可以通过staircase
 (默认值为False,当为True时，（global_step/decay_steps）则被转化为整数) ,选择不同的衰减方式。

'''

# global_step = tf.Variable(0)
# learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True) # 生成学习率
# learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(global_step=global_step) #使用指数衰减学习率



learning_rate = 0.1
decay_rate = 0.96
global_step = 1000
decay_step = 100

global_ = tf.Variable(tf.constant(0))
c = tf.train.exponential_decay(learning_rate, global_, decay_step, decay_rate, staircase=True)
d = tf.train.exponential_decay(learning_rate, global_, decay_step, decay_rate, staircase=False)

T_C = []
F_D = []

with tf.Session() as sess:
    for i in range(global_step):
        T_c = sess.run(c, feed_dict={global_: i})
        T_C.append(T_c)
        F_d = sess.run(d, feed_dict={global_: i})
        F_D.append(F_d)

plt.figure(1)
plt.plot(range(global_step), F_D, 'r-')
plt.plot(range(global_step), T_C, 'b-')

plt.show()





