#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: pythonapp.py 
@desc:
@time: 2017/11/02 
"""

from pyspark import SparkContext
from pyspark import SparkFiles

# with open('text.txt', 'wb') as f:
#     _ = f.write("100")
sc = SparkContext("local[2]", "First Pyspark")
data = sc.textFile("data/UserPurchaseHistory.csv").map(lambda line : line.split(",")).map(lambda record: (record[0], record[1], record[2]))
numPurchases = data.count()
uniqueUsers = data.map(lambda record: record[0]).distinct().count()
totalRevenue = data.map(lambda record: float(record[2])).sum()
# 寻找最受欢迎的产品
products = data.map(lambda record: (record[1], 1.0)).reduceByKey(lambda a, b: a + b).collect()
mostPopular = sorted(products, key=lambda  x: x[1], reverse=True)[0]

print('Total purchases: %d' % numPurchases)
print('Unique users: %d' % uniqueUsers)
print('Total revenue: %2.2f' % totalRevenue)
print('Most popular product: %s with %d purchases' % (mostPopular[0], mostPopular[1]))

sc.stop()