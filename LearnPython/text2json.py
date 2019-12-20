# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:36:15 2018

@author: ESRI
"""


 
f = open('E:\\bigdata\\bigdata-common\data\SensitiveWord\敏感词库_All.txt','r',encoding='utf8')

 
list = []
i = 1
ask = ""
for line in f.readlines():
    v = str(line).strip('\n')
    if i%2==1:
        ask = v
    if i%2==0:
        list.append({"ask":ask,"answer":v})
    i = i+1
 
f.close()
 
v = {"result": list}
file_object = open('E:\\bigdata\\bigdata-common\data\SensitiveWord\sensitiveWord.json', 'w')
try:
    file_object.write(str(v) )
    file_object.close( )
except:
    print(str(v))
