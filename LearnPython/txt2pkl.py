#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: txt2pkl.py 
@desc: 读取txt文件，输出为pkl格式
@time: 2017/10/19 
"""

import pickle

f = open('tlbb.txt', encoding='utf-8')
line = f.read()

output = open('fiction.pkl', 'wb')
pickle.dump(line, output)
print(line,end='')
    # line = f.readline()

f.close()

