#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Spark_config.py 
@desc:  spark相关配置文件
@time: 2017/12/29 
"""

import sys
import os

class SPARK_CONF:
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