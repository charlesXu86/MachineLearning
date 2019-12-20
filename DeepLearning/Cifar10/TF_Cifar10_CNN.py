#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: TF_Cifar10_CNN.py 
@desc: 数据集为Cifar10的CNN实现
@time: 2017/12/21 
"""

# import cifar10.cifar10
# import cifar10.cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

from cifar10 import cifar10
# from cifar10 import cifar10_input
from cifar10.cifar10_input import distorted_inputs
from cifar10.cifar10_input import inputs
from cifar10.cifar10 import maybe_download_and_extract

max_steps = 3000
batch_size = 128
data_dir = 'F:\project\MachineLearning\DeepLearning\Data\cifar10'

def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

maybe_download_and_extract()   # 下载解压数据


# 使用distorted_inputs 产生训练需要的数据，包括特征及其对应的label, 该方法做了数据增强
images_train, labels_train = distorted_inputs(data_dir=data_dir, batch_size=batch_size)

# cifar10_input.inputs() 生成测试数据
image_test, labels_test = inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3]) # 3表示有RGB三种通道
label_holder = tf.placeholder(tf.int32, [batch_size])

# 创建卷积层，注意LRN的理解
# 使用 5x5的卷积核大小，3个颜色通道，64个卷积核
# ==============
# 卷积层对数据进行特征提取， 全连接层进行组合匹配

weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64],stddev=5e-2, w1=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME') # 最大池化的尺寸和步长不一致，可以增加数据的丰富性
# LRN层模仿了生物神经系统的“侧抑制”机制，对局部神经元的活动创建竞争环境，使得其中相应比较大的值变得更大，
# 并一直其他反馈较小的神经元，增强了模型的繁华能力。
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 创建第二个卷积层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1,1,1,1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME')

# Full connect
# 将两个卷积层的输出结果faltten，变成一位向量
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004) # 初始化
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)


weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

# 计算loss
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    # 将softmax的计算和cross_entropy loss 的计算合在一起
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')  # 实现一个列表的元素的相加
# 将logits节点和label_placeholder传入loss函数，得到最终的loss
loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# in_top_k 输出结果中top_k的准确率，默认使用 top 1
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

# 创建session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 启动图片增强的现成队列，，这里使用16个线程来加速，如果不启动这一步，后续无法进行
tf.train.start_queue_runners()

# 开始训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run(sess.run([train_op, loss],
                            feed_dict={image_holder:image_batch, label_holder:label_batch}))
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('Step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

# 评估模型在测试集上的准确率
num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([image_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)






