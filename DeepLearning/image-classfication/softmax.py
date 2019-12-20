#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: softmax.py 
@desc:   
        https://github.com/wolfib/image-classification-CIFAR10-tf
@time: 2017/10/12 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import  print_function

import numpy as np
import tensorflow as tf
import time
from . import data_helpers


beginTime = time.time()

# 定义参数
batch_size = 100
learning_rate = 0.01
max_steps = 1000

# Prepare data
data_sets = data_helpers.load_data()

images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])  # 32*32*3=3072
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

weights = tf.Variable(tf.zeros([3072, 10]))
bias = tf.Variable(tf.zeros([10]))

logits = tf.matmul(images_placeholder, weights) + bias

# Define loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder))

# Define training operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # repeat input data batch
    for i in range(max_steps):
        indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
        images_batch = data_sets['images_train'][indices]
        labels_batch = data_sets['labels_train'][indices]

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                images_placeholder: images_batch, labels_placeholder:labels_batch
            })
            print('Step {:5d}: train_accuracy {:g}'.format(i, train_accuracy))

        sess.run(train_step, feed_dict={
            images_placeholder:images_batch, labels_placeholder:labels_batch
        })

        # evaluate
        test_accuracy = sess.run(accuracy, feed_dict={
            images_placeholder: data_sets['images_test'],
            labels_placeholder: data_sets['labels_test']
        })
        print('Test accuracy {:g}'.format(test_accuracy))

    endTime = time.time()
    print('Total time: {:5.2f}s'.format(endTime - beginTime))



