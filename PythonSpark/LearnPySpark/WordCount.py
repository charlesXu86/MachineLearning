#-*- coding:utf-8 _*-
"""
@author:charlesXu
@file: WordCount.py
@desc: pyspark的HelloWorld
@time: 2017/12/27
"""

from __future__ import print_function

import sys
import os
from operator import add
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import SparkConf


spark_path = 'E:\spark-2.2.1-bin-hadoop2.7'
JAVA_HOME = 'E:\devtools\jdk1.8'
os.environ['JAVA_HOME'] = JAVA_HOME
os.environ['SPARK_HOME'] = spark_path

sys.path.append(spark_path + '/bin')
sys.path.append(spark_path + '/python')
sys.path.append(spark_path + '/python/pyspark')
sys.path.append(spark_path + '/python/lib')
sys.path.append(spark_path + '/python/lib/pyspark.zip')
sys.path.append(spark_path + '/python/lib/py4j-0.10.4-src.zip')

if __name__=='__main__':
    # if len(sys.argv) != 2:
    #     print("Usage: wordcount <file>", file=sys.stderr)
    #     exit(-1)

    # conf = SparkConf.setMaster('local').setAppName('WordCount')
    # sc = SparkContext(conf)
    # sc.textFile('E:\py_workspace\MachineLearning\PythonSpark\Data\wordcount.txt')
    # lines = sc.
    spark = SparkSession.builder\
                        .appName('pyWordCount')\
                        .getOrCreate()
    lines = spark.read.text(sys.argv[0]).rdd.map(lambda r: r[0])
    counts = lines.flatMap(lambda x: x.split(' '))\
                  .map(lambda x: (x, 1))\
                  .reduceByKey(add)
    output = counts.collect()
    for (word, count) in output:
        print('%s: %i' % (word, count))

    spark.stop()