#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: style-transfer.py 
@desc: 风格迁移
@time: 2017/11/09 
"""
import os
from six.moves import urllib
from scipy.io import loadmat
from scipy.misc import imresize

import tensorflow as tf

def download_hook(count, block_size, total_size):
    if count % 20 == 0 or count * block_size == total_size:
        percentage = 100.0 * count * block_size / total_size
        barstring = ["=" for _ in range(int(percentage / 2.0))] + ["," for _ in range(50 - int(percentage / 2.0))]
        barstring = "[" + "".join(barstring) + "]"
        outstring = '%02.02f%% (%02.02f of %02.02f MB)\t\t' + barstring
        print(outstring % (percentage, count * block_size / 1024.0 / 1024.0, total_size / 1024.0 / 1024.0), end='\r')


path = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
fname = "vgg-19.mat"
if not os.path.exists(fname):
    print("Downloading ...")
    filepath, _ = urllib.request.urlretrieve(path, filename=fname, reporthook=download_hook)
    print("Done.")

if not os.path.exists("content.jpg"):
    urllib.request.urlretrieve(
        "https://upload.wikimedia.org/wikipedia/commons/6/6b/Donald_Trump_by_Gage_Skidmore_5.jpg",
        filename="content.jpg")  # Attribution: Gage Skidmore
    urllib.request.urlretrieve("https://upload.wikimedia.org/wikipedia/commons/e/e8/Indischer_Maler_um_1690_001.jpg",
                               filename="style.jpg")

original_layers = loadmat(fname)["layers"](0)
original_layers.shape