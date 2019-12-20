# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

print('>>>',tf.__version__)   # 打印Tensorflow的版本号

'''
   该类记录Tensorflow主要的方法和函数
   参见博客：http://blog.csdn.net/lenbow/article/details/52218551
            http://blog.csdn.net/UESTC_C2_403/article/details/72190282
'''

# 1,tf.random_uniform(shape,minval,maxval,dtype,seed,name) 产生一个随机矩阵

# with tf.Session() as sess:
#     print(sess.run(tf.random_uniform(
#         (4,4), minval=1, maxval=5,dtype=tf.float64
#     )))

# 2,tf中maximum,minimun,argmax,argmin对比
#     tf.maximum(a,b) 返回a，b之间的最小值
#     tf.minimum(a,b) 返回a，b之间的最大值
#     tf.argmax(a,dimension) 返回的是a中的某个维度最大值的索引。
#     tf.argmin(a,dimension) 返回的是a中的某个维度最小的索引。

# 3,tf.get_shape : 主要用于获取一个张量的维度，并且输出张量的值，如果是二维矩阵，也就是输出行和列的值。
# with tf.Session() as sess:
#     A = tf.random_normal(shape=[3,4])
# print(A.get_shape())
# print(A.get_shape)
# print(A.get_shape()[0])     # 表示第一个维度

# 4,tf把彩色图像转换为灰度图像：tf.image.rgb_to_grayscale
# with tf.Session() as sess:
#     image_raw_data_jpg = tf.gfile.FastGFile("100.jpg",'r').read()
#     image_data = tf.image.decode_jpeg(image_raw_data_jpg)
#     image_data = sess.run(tf.image.rgb_to_grayscale(image_data))
#     print(image_data.shape)
#     plt.imshow(image_data[:,:,0],cmap='gray')
#     plt.show()

# 5,tf.nn.conv2d_transpose(x,W,output_shape,strides,padding='SAME').推荐使用x的数据格式默认为
#   [batch,height,width,in_channels]. W是滤波器，输入格式为[height,width,output_channels].
#   output_shape就是输出数据的格式[batch,height,width,in_channels],Strides是滑动的步长
#   [1，h_l, w_l, 1].输出数据格式的height等于h_l乘以输入的height
# x = tf.ones([10,4,4,5])
# w = tf.ones([2,2,2,5])
# y = tf.nn.conv2d_transpose(x, w, output_shape=[10,12,12,2], strides=[1,3,3,1], padding='SAME')
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(y))

# 6,tf.unpack(A, axis) 这是一个解包函数。A 是一个需要解包的对象，axis是一个解包方式的定义。
#    默认是0，如果是0，返回的结果就是按行解包， 1则是按列解包。
#
# A = [[1,2,3],[4,5,6]]
# B = tf.unpack(A,axis=1)
# with tf.Session() as sess:
#     print(sess.run(B))
# tf.pack(values, axis=0, name='pack')  将一系列rank-R的tensor打包为为一个rank-(R+1)的tensor
#       例: 'x' is [1,4], 'y' is [2,5], 'z' is [3,6]
#         pack([x,y,z]) => [[1,4], [2,5],[3,6]]
#  沿着axis=1(按列)
#  pack([x,y,z], axis=1) => [[1,2,3], [4,5,6]]
#  tf.pack([x,y,z]) 等价于 np.asarray([x,y,z])

# 7,tf.add_n([p1,p2,p3...])函数实现一个列表的元素的相加。就是驶入的对象是一个列表，列表里的元素可以是向量，矩阵等
# input1 = tf.constant([1.0,2.0,3.0])
# input2 = tf.Variable(tf.random_uniform([3]))
# output = tf.add_n([input1, input2])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(input2 + input1))
#     print(sess.run(output))

# 8,tf里几个随机数的用法
#    tf.constant(value,dtype=None, shape=None) :创建一个常量tensor，按照给出value来赋值，用shape指定形状。
#                     value可以是一个数，，也可以是一个list
#    tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32) 正太分布随机数，mean是均值，stddev是标准差。
#    tf.truncated_normal(shape,mean,stddev,dtype) 截断正太分布随机数，不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数。
#    tf.random_uniform(shape,minval,maxval,dtype)均匀分布随机数，范围为[minval,maxval]

# 9, tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True,max_norm=None)
#       主要是选取一个张量里面索引对应的元素。tensor是输入的张量，id是张量对应的索引(还有其他参数)
#   参数说明: params是一些tensor组成的列表或单个的tensor。
#           ids是一个整形的tensor，每个元素将代表要在params中取的每个元素的第0维的逻辑index，这个逻辑index是由partiton_stategy来指定的。
#           partition_stategy用来设定ids的切分方式，目前有两种切分方式div和mod。其中mod的切分方式是如果对[1,2,3,4,5,6,7]进行切分则结果为
#                            [1,4,7], [2,5], [3,6],如果是div的切分方式则是[1,2,3],[4,5],[6,7].这两种切分方式在无法均匀切分的情况下都是
#                              将前(max_id + 1) % len(params)个切分多分配一个元素
# c = np.random.random([10,1])
# b = tf.nn.embedding_lookup(c,[1,3])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print("c++++++++",c)
#     print(sess.run(b))

# 10,tf.train.batch 和tf.train.shuffle_batch的用法

# 11,tf.concat(axis,[list1,list2]) 用于合并两个迭代器(比如列表)。axis表示合并的方向。0表示竖直，1表示水平
# t1 = [[1,2,3],[4,5,6]]
# t2 = [[7,8,9,10],[10,11,12,12]]
# x = tf.constant([[1,2,3], [4,5,6],[2,2,2]])
# y = tf.constant([[7,8,9],[11,12,13], [3,3,3]])
# print(tf.Session().run(tf.concat(0,[x,y])))
# print(tf.Session().run(tf.concat(1,[x,y])))

# 11,assert用于断言，也就是判断下一个表达式的对错，或者是一数字或者布尔类型的数据。如果是对的，那就继续执行下面程序，反正出现断言错误的标记，AssertionError

# 12,tf.add_to_collection:把变量放入一个集合，把很多变量组成一个列表
#    tf.get_collection :从一个集合中取出全部变量，是一个列表
# v1 = tf.get_variable(name='v1', shape=[1], initializer=tf.constant_initializer(0))
# tf.add_to_collection("loss",v1)
# v2 = tf.get_variable(name='v2', shape=[1], initializer=tf.constant_initializer(2))
# tf.add_to_collection("loss",v2)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(tf.get_collection('loss'))

# 13,tensorflow中关键字global_step:经常在滑动平均，学习速率变化的时候需要用到。
# 这个参数在tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)里面有，系统会自动更新这个参数的值，从1开始。
# x = tf.placeholder(tf.float32,shape=[None,1],name='x')
# y = tf.placeholder(tf.float32,shape=[None,1],name='y')
# w = tf.Variable(tf.constant(0.0))
# global_steps = tf.Variable(0,trainable=False)
# learning_rate = tf.train.exponential_decay(0.1,global_steps,10, 2, staircase=False)
# loss = tf.pow(w*x-y,2)
#
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_steps)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10):
#         sess.run(train_step, feed_dict={x:np.linspace(1,2,10).reshape([10,1]),
#                                         y:np.linspace(1,2,10).reshape([10,1])})
#         print(sess.run(learning_rate))
#         print(sess.run(global_steps))

# 14, tf.reduce_mean   求均值。
# A = np.array([[1.,1.],[2.,2.]])
# with tf.Session() as sess:
#     print(sess.run(tf.reduce_mean(A)))   #求总体的均值
#     print(sess.run(tf.reduce_mean(A,axis=0))) #按列求均值
#     print(sess.run(tf.reduce_mean(A,axis=1))) #按行求均值

# 15,tf.clip_by_value(A,min,max):输入一个张量，把A中的每一个元素的值都压缩在min和max之间。小于min的让他等于min，大于max的等于max。
# A = np.array([[1,1,2,4],[3,4,8,5]])
# with tf.Session() as sess:
#     print(sess.run(tf.clip_by_value(A, 2, 5)))

# 16,tf.cast:用于改变某个张量的数据类型
# A = tf.convert_to_tensor(np.array([[1,1,2,4],[3,4,8,5]]))
# with tf.Session() as sess:
#     print(A.dtype)
#     B = tf.cast(A,tf.float64)
#     print(B.dtype)

# 17,tf.train.exponential_decay(learning_rate, global_step,
# decay_steps, decay_rate, staircase=False, name=None) 退化学习率，对学习率进行指数衰退

# 18,tf.convert_to_tensor : 用于将不同数据转换为张量，比如可以让数组变成张量，也可以让列表变成张量


# 19 tf.trainable_variables: 返回的是需要训练的变量列表
#    tf.all_variable：返回的是所有变量的列表
# v = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), name='v')
# v1 = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='v1')
#
# globle_step = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='globle_step', trainable=False)
# ema = tf.train.ExponentialMovingAverage(0.99, globle_step)
#
# for ele1 in tf.trainable_variables():
#     print(ele1.name)
# for ele2 in tf.all_variables:
#     print(ele2.name)

# 20, tf.train.ExponentialMovingAverage(decay, steps) 这个函数用于更新参数，就是滑动平均的方法更新参数。
#     滑动平均操作的意义在于提高模型在测试数据上的健壮性
#     这个函数初始化需要提供一个衰减速率(decay)，用于控制模型的更新速度。这个函数还会维护一个影子变量(也就是更新参数后的参数值)，这个影子变量
#     的初始值就是这个变量的初始值，影子变量的值得更新方式如下：
#     shadow_variable = decay * shadow_variable + (1-decay) * variable
#    上述公式可知，decay控制着模型更新的速度，越大越趋于稳定。实际运用中，decay一般会设置为十分接近于1的常数(0.99或0.999)。为了使得模型在训练的
#    初始阶段更新的更快，ExponentialMovingAverage还提供了num_update参数来动态设置decay大小。
#        decay = min{decay, (1 + num_updates)/(10 + num_updates)}
# v1 = tf.Variable(dtype=tf.float32, initial_value=0.)
# decay = 0.99
# num_update = tf.Variable(0,trainable=False)
# ema = tf.train.ExponentialMovingAverage(decay=decay, num_updates=num_update)
#
# update_var_list = [v1]   # 定义更新变量列表
# ema_apply = ema.apply(update_var_list)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run([v1,ema.average(v1)]))
#
#     sess.run(tf.assign(v1, 5))
#     sess.run(ema_apply)
#     print(sess.run([v1, ema.average(v1)]))
#
#     sess.run(tf.assign(num_update, 100000))
#     sess.run(tf.assign(v1, 10))
#     sess.run(ema_apply)
#     print(sess.run([v1, ema.average(v1)]))
#
#     sess.run(ema_apply)
#     print(sess.run([v1, ema.average(v1)]))

# 21,tf.assign(A, new_number):把A的值变为new_number
# A = tf.Variable(tf.constant((0.0), dtype=tf.float32))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(A))
#     sess.run(tf.assign(A, 10))
#     print(sess.run(A))

# 22, tf.train.NewCheckPointReader('path'): path是保存的路径，这个函数可以得到保存的所有变量
# v = tf.Variable(0,dtype=tf.float32,name='v')
# v1 = tf.Variable(0,dtype=tf.float32,name='v1')
# result = v + v1
#
# x = tf.placeholder(tf.float32,shape=[1],name='x')
# test = result + x
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     saver.save(sess, "D:\model.ckpt")
#
# reader = tf.train.NewCheckpointReader("D:\model.ckpt")
# variable = reader.get_variable_to_shape_map()
# for ele in variable:
#     print("+++",ele)

# 23, tf.add_to_collection : 把变量放入一个集合，把很多变量变成一个列表
#     tf.get_collection : 从一个集合中取出全部变量，是一个列表
#     tf.add_n ：把一个列表的东西都一次加起来
# v1 = tf.get_variable(name='v1',shape=[1],initializer=tf.constant_initializer(0))
# tf.add_to_collection('loss', v1)
# v2 = tf.get_variable(name='v2',shape=[1], initializer=tf.constant_initializer(2))
# tf.add_to_collection('loss', v2)
#
# with tf.Session() as sess:
#     sess.run()

# ===========参数初始化
# 参见博客：http://www.cnblogs.com/denny402/p/6932956.html
# =============
# 24,tf.constant_initailizer也可以简写为tf.Constant(), 初始化为常数，这个非常有用，
#    通常偏置项就是用它来初始化的。由他衍生出的两个初始化方法
#     tf.zeros_initializer()

# tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True/False) 指数衰减法 解决设定学习率的问题，提供了这种方法
#        步骤: 1,首先使用较大的学习率(目的：为快速得到一个比较优的解)
#             2,然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定)
# decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
#



# 25, tf.reduce_max() 计算tensor中各个元素的最大值
#     reduce_max(t, 1) 是找出各行中最大值组成一个tensor
#     reduce_min(t, 0) 是找出各列中最大值组成一个tensor

# 26, tf.reduce_all() 计算tensor中各个元素的逻辑和(and运算)
# ‘x' is [[True, True],[False, False]]
#  tf.reduce_all(x)      =>  False
#  tf.reduce_all(x, 0)   =>  [False, False]
#  tf.reduce_all(x, 1)   =>  [True, False]

# 27, tf.reduce_any()    计算tensor中的各个元素的逻辑或(or运算)

# 28, tf.equal() 是对比两个矩阵或者向量的相等的元素，如果是相等的就返回True,反之则为False
# A = [[1,2,3,4,5,7]]
# B = [[1,2,4,4,5,5]]
# with tf.Session() as sess:
#     print(sess.run(tf.equal(A, B)))

# 29, tf.Variable() 创建，初始化，保存和加载，可以通过构造类Variable的实例向图中添加变量
#    如果要稍后更改变量的形状，则必须使用带有validate_shape = False的赋值操作。
# 如果需要创建一个取决于另一个variable的初始值的variable,要使用另外一个variable的initialized_value(),
#   这样可以确保以正确的顺序初始化变量。参照官网的例子:
#   Initialize 'v' with a random tensor
# v = tf.Variable(tf.truncated_normal([10, 40]))
#  use initialize_value to guarantee that 'v' has been initialized before
#  its value is used to initialize 'w', the random values are picked only ones
# w = tf.Variable(v.initialized_value() * 2.0)

# 30, tf.transpose(a, perm=None, name='transpose') 调换tensor的维度顺序，按照列表perm的维度排列调换tensor顺序（矩阵转置）
# 如未定义，则perm为(n-1, ....., 0)
# x = [[1, 2, 3],[4,5,6]]
# y = tf.transpose(x, perm=[1,0])
# print(y)

# 31,


# tf.gather(params,indices, validate_indices=None, name=None)合并索引所指示params中的切片

# 32, tf.nn.bidirectional_dynamic_rnn()