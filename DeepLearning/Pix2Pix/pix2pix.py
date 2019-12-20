#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: pix2pix.py 
@desc: pix2pix
       https://github.com/affinelayer/pix2pix-tensorflow
@time: 2017/10/30 
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy
import os
import argparse
import json
import glob
import random
import collections
import math
import time

