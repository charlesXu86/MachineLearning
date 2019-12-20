#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Embedding_Lookup.py
@desc:  https://github.com/xiholix/basicFunction/blob/master/embeddingLookup/embeddingLookup.py
@time: 2017/11/18 
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

def test_embedding_lookup():
    a = np.arange(8).reshape(2,4)
    b = np.arange(8,12).reshape(1,4)
    c = np.arange(12,20).reshape(2,4)
    print(a)
    print(b)
    print(c)

    a = tf.Variable(a)
    b = tf.Variable(b)
    c = tf.Variable(c)

    t = tf.nn.embedding_lookup([a,b,c], ids=[1,3])
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    m = sess.run(t)
    print(m)

if __name__=='__main__':
    test_embedding_lookup()