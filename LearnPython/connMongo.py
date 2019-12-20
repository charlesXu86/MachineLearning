# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     connMongo.py
   Description :  连接mongoDB
   Author :       charl
   date：          2018/9/3
-------------------------------------------------
   Change Activity: 2018/9/3:
-------------------------------------------------
"""

from pymongo import MongoClient

# 建立mongo数据库连接
client = MongoClient('localhost', 27017)

# 连接所需的数据库
db = client.chatbot_cn

collection = db.chatbot_cn   # 表名

# 查询所有数据
for item in collection.find():
    print(item)

print(collection.find_one())

# 向集合中插入数据  insert_one()  or   insert_many()
# collection.insert_one({'name':'Tom', 'age': '25', 'desc': '你不是一个好人'})

# 更新数据
collection.update_one({'name': 'Tom'}, {'name':'Tom', 'age': '18'})

