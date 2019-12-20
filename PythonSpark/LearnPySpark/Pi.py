#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Pi.py 
@desc:
@time: 2017/12/29 
"""

import sys
from random import random
from operator import add

from pyspark.sql import SparkSession
from Util.Spark_config import SPARK_CONF

if __name__=='__main__':
    config = SPARK_CONF
    spark = SparkSession\
               .builder\
               .appName("PythonPi")\
               .getOrCreate()

    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n = 100000 * partitions

    def f(_):
        x = random() * 2 - 1
        y = random() * 2 - 1
        return 1 if x ** 2 + y ** 2 <= 1 else 0

    count = spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
    print("Pi is roughly %f" % (4.0 * count / n))
    spark.stop()
