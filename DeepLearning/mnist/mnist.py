#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: mnist.py 
@desc:
@time: 2017/10/13 
"""
import tensorflow as tf
from . import input_data
mnist = input_data.read_data_sets('Mnist_data', one_hot=True)