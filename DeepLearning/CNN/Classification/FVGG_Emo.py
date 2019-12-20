#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: FVGG_Emo.py 
@desc: 人脸表情识别
@time: 2017/10/15 
"""

import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os

def vgg16(input, num_class):
    # 在模型中，另trainable=False 来保证参数在训练过程中不会被更新
    x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1', trainable=False)
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpol1') # 最大池化操作

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1', trainable=False)
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpol2')  # 最大池化操作

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1', trainable=False)
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2', trainable=False)
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpol3')  # 最大池化操作

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1', trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2', trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpol4')  # 最大池化操作

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1', trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2', trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpol5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    # we changed the structure here to let the fc only have 2048, less parameter, enough for our task
    x = tflearn.fully_connected(x, 2048, activation='relu', scope='fc7', restore=False)
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8', restore=False)

    return x

# 加载数据
model_path = '.'
files_list = './data/train_fvgg_emo.txt'

from tflearn.data_utils import image_preloader  # 图片预加载，有两种模式， mode='file/folder

X, Y = image_preloader(files_list, image_shape=(224, 224), mode='file',
                       categorical_labels=True, normalize=False,
                       files_extension=['.jpg', '.png'], filter_channel=True)
print(X, Y)
num_classes = 7  # Num of your dataset

# VGG processing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean=[123.68, 116.779, 103.939], per_channel=True) # 确定数据是规范的
# img_prep.add_featurewise_stdnorm()

#  VGG Network
x = tflearn.input_data(shape=[None, 224, 224, 3], name='input', data_preprocessing=img_prep)
softmax = vgg16(x, num_classes)
regression = tflearn.regression(softmax, optimizer='adam', loss='categorical_crossentropy',
                                learning_rate=0.002, restore=False)
model = tflearn.DNN(regression, checkpoint_path='vgg-finetuning', max_checkpoints=3, tensorboard_verbose=2,
                   tensorboard_dir="./logs" )

model_file = os.path.join(model_path, "vgg16.tflearn")
model.load(model_file, weights_only=True)

# Start finetuning
model.fit(X, Y, n_epoch=100, validation_set=0.1, shuffle=True, show_metric=True, batch_size=64,
          snapshot_epoch=False, snapshot_step=200, run_id='vgg-finetuning')
model.save('My VGG16')
