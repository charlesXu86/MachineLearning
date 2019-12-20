#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: TF_AutoEncoder.py 
@desc: tf实现自编码器，数据集为mnist
@time: 2017/12/13 
"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


# 1, 参数初始化，使用的是xavier， 他的特点是会根据某一层网络的输入和输出节点数自动调整最合适的分布。
#    在深度学习中，如果模型的权重初始化的太小，那信号将在每层的传递中主键缩小而难以产生作用，如果太大，
#    那信号将在珠层的传递中逐渐放大并导致发散和失效。
#    xavier的作用就是让权重被初始化的不大不小，正好合适。

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32) # 创建一个在此范围内的均匀分布

# 2, 定义一个去噪自编码的class，方便以后使用
class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        '''

        :param n_input: 输入变量数
        :param n_hidden: 隐含层节点数
        :param transfer_function: 隐含层激活函数，默认为softplus
        :param optimizer: 优化器
        :param scale: 高斯噪声系数，默认为0.1
        '''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 接下来定义网络结构。
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),self.weights['w1']),
                                           self.weights['b1']))
        # 经过隐含层后，需要在输出层进行数据复原，重建操作
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 接下来定义自编码器的损失函数，这里使用平方误差
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict= {self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 返回自编码器隐含层的输出结果，他的目的是提供一个接口来获取抽象后的特征，自编码器的隐含层的最主要功能就是学习出数据中的高阶特征。
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # 他将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # 定义reconstruct函数，他整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据
    # 输入数据是原始数据，输出数据是复原后的数据
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x : X, self.scale: self.training_scale})

    # 获取隐含层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取偏置
    def getBias(self):
        return self.sess.run(self.weights['b1'])

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义一个队训练，测试数据标准化的函数
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_test, X_train

# 定义一个随机获取block数据   不放回抽样
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

# 对训练集，测试集进行标准化转换
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epoches = 20
batch_size = 120
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoEncoder(
                                        n_input=784,
                                        n_hidden=200,
                                        transfer_function=tf.nn.softplus,
                                        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                        scale=0.01)
for epoch in range(training_epoches):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
