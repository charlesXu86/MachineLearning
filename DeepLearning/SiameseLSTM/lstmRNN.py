#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: lstmRNN.py 
@desc: 孪生网络 github: https://github.com/jx00109/siamese-lstm-for-sentence-similarity
               CSDN: http://blog.csdn.net/sinat_31188625/article/details/72675627
               论文原文：http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf
@time: 2017/09/30 
"""

import tensorflow as tf

class LSTMRNN(object):
    def singleRNN(self, x, scope, cell='lstm', reuse=None):
        if cell == 'gru':
            with tf.variable_scope('grucell' + scope, reuse=reuse, dtype=tf.float64):
                used_cell = tf.contrib.rnn.GruCell(self.hidden_neural_size)
