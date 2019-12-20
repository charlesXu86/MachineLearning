# -*- coding:utf-8 _*-
""" 
@author:charlesXu
@file: SparkSQL.py 
@desc: 基于python的SparkSQL笔记
@time: 2017/12/27 
"""
from __future__ import print_function

import sys
import os

from Util.Spark_config import SPARK_CONF

from pyspark import sql
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession  # 初始化session
from pyspark import SparkConf
from pyspark.sql import Row
from pyspark.sql.types import *  # 数据类型

# spark_path = 'E:\spark-2.2.1-bin-hadoop2.7'
# JAVA_HOME = 'E:\devtools\jdk1.8'
# os.environ['JAVA_HOME'] = JAVA_HOME
# os.environ['SPARK_HOME'] = spark_path
#
# sys.path.append(spark_path + '/bin')
# sys.path.append(spark_path + '/python')
# sys.path.append(spark_path + '/python/pyspark')
# sys.path.append(spark_path + '/python/lib')
# sys.path.append(spark_path + '/python/lib/pyspark.zip')
# sys.path.append(spark_path + '/python/lib/py4j-0.10.4-src.zip')

'''
sql.SQLContext    #  DataFrame和SQL方法的入口
sql.DataFrame     #  将分布式数据集分组到指定列名的数据框中
sql.Column        #  Dataframe中的列
sql.Row           #  Dataframe中的行
sql.HiveContext   #  访问Hive数据的主入口
sql.GroupedData   #  由DataFrame.groupBy()创建的聚合方法类
sql.DataFrameNaFunctions  #  处理丢失数据(空数据)的方法
sql.DataFrameStatFunctions  # 统计功能的方法
sql.Window                  # 用于处理窗口函数
sql.DataFrameReader
'''


# sqlContext = SQLContext()
# l = [{'Alice', 1}]
# d = [{'name':'Alice', 'age':1}]
# x = SQLContext.createDataFrame(d).collect()
# print(x)


def basic_df_example(spark):
    df = spark.read.json('/opt/datas/project/MachineLearning/PythonSpark/Data/people.json')
    df.show()

    df.printSchema()

    df.select('name').show()

    # Select everybody, but increment the age by 1
    df.select(df['name'], df['age'] + 1).show()

    df.filter(df['age'] > 21).show()

    # Count people by age
    df.groupBy('age').count().show()

    # Register the DataFrame as a SQL temporaryView
    df.createOrReplaceTempView("people")

    sqlDF = spark.sql("SELECT * FROM people")
    sqlDF.show()

    # Register the DataFrame as a global temporaryView
    df.createGlobalTempView('people')

    spark.sql('SELECT * FROM global_temp.people')

    spark.newSession().sql('SELECT * FROM global_temp.people').show()


def schema_inference_example(spark):
    sc = spark.sparkContext

    # 加载文本文件，并且把每行转换成一个Row
    lines = sc.textFile('/opt/datas/project/MachineLearning/PythonSpark/Data/people.txt')
    parts = lines.map(lambda l: l.split(","))
    people = parts.map(lambda p: Row(name=p[0], age=int(p[1])))

    # Infer the schema, and register the DataFrame as a table
    schemaPeople = spark.createDataFrame(people)
    schemaPeople.createOrReplaceTempView('people')

    teenagers = spark.sql('SELECT name FROM people WHERE age >= 13 AND age <=21')

    teenNames = teenagers.rdd.map(lambda p: "Name:" + p.name).collect()
    for name in teenNames:
        print(name)


def programmatic_schema_example(spark):
    sc = spark.sparkContext
    # 加载文本文件，并且把每行转换成一个Row
    lines = sc.textFile('/opt/datas/project/MachineLearning/PythonSpark/Data/people.txt')
    parts = lines.map(lambda l: l.split(","))
    # Each Line is converted to a tuple
    people = parts.map(lambda p: (p[0], p[1].strip()))

    schemaString = "name age"
    fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]  # zhejuhua de yisi
    schema = StructType(fields)

    # Apply the schema to the RDD
    schemaPeople = spark.createDataFrame(people, schema)

    # Create a temporary View using the DataFrame
    schemaPeople.createOrReplaceTempView("people")

    results = spark.sql("SELECT name FROM people")
    results.show()


if __name__ == "__main__":
    config = SparkConf() \
        .set('spark.executor.instances', '2') \
        .set('spark.executors.cores', '2') \
        .set('spark.executor.memory', '2g') \
        .set('spark.driver.memory', '1g') \
        .set('spark.default.parallelism', '40') \
        .set('spark.yarn.executor.memoryOverhead', '1000')
    spark = SparkSession.builder \
        .appName("Python Spark SQL basic example") \
        .getOrCreate()

    basic_df_example(spark)
    schema_inference_example(spark)
    programmatic_schema_example(spark)
    spark.stop()

