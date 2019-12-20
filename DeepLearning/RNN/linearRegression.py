'''
 linear Regression
 了解开发的基本步骤和一些基本的api和方法的调用
'''
import tensorflow as tf
import numpy as np
rng = np.random

# 预先设置模型参数
learning_rate = 0.02
training_epoches = 3000
display_step = 50

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
# shape 是numpy.core.fromnumeric 中的函数，他的功能是读取矩阵的长度。
# 比如shape[0]就是读取矩阵的第一维度的长度。他的输入参数可以使一个整数表示维度，也可以是一个矩阵。
n_samples = train_X.shape[0]

# tf graph Input
# tf.placeholder : 用于得到传递进来的真实的训练样本   与 tf.Variable比较。
#   可以不用指定初值，可以在运行时通过Session.run的函数的 feed_dict 参数指定
#   这也是其命名的原因所在，仅仅作为一种占位符。
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weight
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct linear model
pred = tf.add(tf.multiply(X, W), b)

# MSE
cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)

# 求梯度
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epoches):
        for(x,y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X:x, Y:y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print("Epoch:", '0%4d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    training_cost = sess.run(cost,feed_dict={X:train_X, Y:train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')