#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: pattern.py 
@desc: Define pattern of integer for seq2seq example
@time: 2017/10/24 
"""

import numpy as np
import tensorflow as tf

class SequencePattern(object):
    INPUT_SEQUENCE_LENGTH = 10
    OUTPUT_SEQUENCE_LENGTH = 10
    INPUT_MAX_INT = 9
    OUTPUT_MAX_INT = 9
    PATTERN_NAME = "sorted"

    def __init__(self, name=None, in_seq_len=None, out_seq_len=None):
        if name is not None:
            assert hasattr(self, "%s_sequence" % name)
            self.PATTERN_NAME = name
        if in_seq_len:
            self.INPUT_SEQUENCE_LENGTH = in_seq_len
        if out_seq_len:
            self.OUTPUT_SEQUENCE_LENGTH = out_seq_len

    def generate_output_sequence(self, x):
        '''
        For a given inout sequence, generate the output sequence.
        This produce defines the pattern which the seq2seq RNN will be trained to find
        :param x: x is a ID numpy array of a ID integers, with length INPUT_SEQUENCE_LENGTH
        :return: Returns a 1D numpy array of length OUTPUT_SEQUENCE_LENGTH
        '''
        return getattr(self, "$s_sequence" % self.PATTERN_NAME)(x)

    def maxmin_dup_sequence(self, x):
        '''
         Generate sequence with [max, min, rest of original entries]
        :param x:
        :return:
        '''
        x = np.array(x)
        y = [x.max(), x.min()] + list(x[2:])
        return np.array(y)[:self.OUTPUT_SEQUENCE_LENGTH]

    def reverse_sequence(self, x):
        '''
        Generate reversed version oforiginal sequence
        :param x:
        :return:
        '''
        return np.array(x[:-1])[:self.OUTPUT_SEQUENCE_LENGTH]