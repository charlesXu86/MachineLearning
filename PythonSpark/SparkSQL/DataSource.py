#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: DataSource.py 
@desc: SparkSQl的各种数据源文件读取
@time: 2017/12/29 
"""

from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.sql import Row


def basic_datasource_example(spark):
    df = spark.read.load('F:\project\MachineLearning\PythonSpark\Data\\users.parquet')
    df.select("name", "favorite_color").write.save("namesAndFavColors.parquet")
    df.write.partitionBy("favorite_color").format("paequet").save("namesPartByColor.parquet")

    df = spark.read.parquet('F:\project\MachineLearning\PythonSpark\Data\\users.parquet')
    (df
            .write
            .partitionBy("favorite_color")
            .bucketBy(42, "name")
            .saveAsTable("people_partitioned_bucketed"))

    # Manual_load_options
    df = spark.read.load('F:\project\MachineLearning\PythonSpark\Data\people.json', format="json")
    df.select("name", "age").write.save("nameAndAges.parquet", format="parquet")

    df.write.bucketBy(42, "name").sortBy("age").saveAsTable("people_bucketed")

    df = spark.sql("SELECT FROM parquet. 'F:\project\MachineLearning\PythonSpark\Data\\users.parquet'")
    spark.sql("DROP TABLE IF EXISTS people_bucketed")
    spark.sql("DROP TABLE IF EXISTS people_partitioned_bucketed")

if __name__ == '__main__':
    spark = SparkSession.builder\
                        .appName("Python Spark SQL datasource Example")\
                        .getOrCreate()
    basic_datasource_example(spark)
    spark.stop()

