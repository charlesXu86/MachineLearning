#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: text_cnn.py 
@desc: 基于CNN的文本分类
       https://github.com/Silencezjl/machine_learning/blob/master/classification/cnn/text_cnn/text_cnn.py
@time: 2017/10/30 
"""

import tensorflow as tf

class TextCNN(object):
    '''
     搭建一个用于文本数据的cnn模型，使用嵌入层(embedding layer)
                                   卷积层(convolutional)
                                   最大池化层(max-pooling)
                                   softmax层
    '''
