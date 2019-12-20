#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Cifar10.py
@desc: 数据处理
@time: 2017/10/18 
"""

import os
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3

class Cifar10Dataset(object):

    def __init__(self, data_dir, subset='train', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion

    def get_filenames(self):
        if self.subset in ['train', 'validation', 'eval']:
            return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, serialized_example):
        '''
         Parses a single tf.Example into image ande label tensors
        :param serialized_example: 
        :return: 
        '''
        features = tf.parse_single_example(
            serialized_example,
            features= {
                'image':tf.FixedLenFeature([], tf.string),
                'label':tf.FixedLenFeature([], tf.int64),
            }
        )
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])

        # Reshape from [depth * height * width] to [depth, height, width]
        image = tf.cast(
            tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]), tf.float32
        )
        label = tf.cast(features['label'], tf.int32)

        image = self.preprocess(image)
        return image, label

    def make_batch(self, batch_size):
        '''
         Read the images and labels from ''filenames
        :param batch_size: 
        :return: 
        '''
        filenames = self.get_filenames()

        # Repeat records
        '''
         TextLineDataset: 从文本文件中读取各行内容
         TFRecordDataset: 从TfRecord文件中读取记录
         FixedLengthRecordDataset: 从二进制文件中读取固定大小的记录
        '''
        from tensorflow.contrib.data.python.ops import dataset_ops
        dataset = dataset_ops.TFRecordDataset(filenames).repeat()
        dataset = dataset.map(self.parser)
    #
    def preprocess(self, image):
        '''
         预处理
         Preprocess a single image in [height, width, depth] layout
        :param image: 
        :return: 
        '''
        if self.subset == 'train' and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
            image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
            image = tf.image.random_flip_left_right(image)
        return image

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 45000
        elif subset == 'validation':
            return 500
        elif subset == 'eval':
            return 10000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)
