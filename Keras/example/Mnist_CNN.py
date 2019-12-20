#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Mnist_CNN.py 
@desc: 基于Keras实现的CNN手写体识别
@time: 2018/01/11 
"""

from __future__ import print_function

import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import  backend as K


batch_size = 128
num_classes = 10
epochs = 20

# 输入图片维度
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 参考keras文档: http://keras-cn.readthedocs.io/en/latest/backend/
if K.image_data_format() == 'channel_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape(0), 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 惯序模型是多个网络层的线性堆叠，也就是"一条路走到黑"

model = Sequential()   # 惯序模型   http://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 在训练模型之前，需要用compile来对学习过程进行配置，compile接受三个参数:
# 优化器:optimizer:该参数可指定为已预定义的优化器名，如rmsprop, adagrad, 或Optimizer
# 损失函数：该参数为模型试图最小化的目标函数
# 指标列表metrics：对分类问题，我们一般将该类问题设为metrics=['accuracy'].指标可以是一个预定义指标的名字，也可以是一个用户定制的函数
#                 指标函数应该返回单个张量，或一个完整的metric_name -> metric_value映射的字典。
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                                                           metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
           validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

