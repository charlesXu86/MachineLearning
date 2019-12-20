#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Py_Json.py 
@desc: python的json操作
@time: 2018/02/25 
"""

'''
  json的操作函数: dumps，loads， dump，load
  
  json.dumps() 用于将dict类型的数据转换成str，
      因为如果直接将dict类型的数据写入json文件中会发生报错，因此在将数据写入时需要用到该函数。
    
  json.loads() 用于将str类型的数据转换成dict  
  
  json.dump()  用于将dict类型的数据转换成str，并且写入到json文件中
  
  json.load()  用于读取json文件中的数据
'''
import json

name_emb = {'a':'1111', 'b':'2222', 'c':'3333', 'd':'4444'}


jsObj = json.dumps(name_emb)
print(name_emb)
print(jsObj)

print(type(name_emb))
print(type(jsObj))

