#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: learnPandas.py 
@desc:
      中文文档 ：http://python.usyiyi.cn/translate/Pandas_0j2/index.html
@time: 2017/10/06 
"""

import numpy as np
import pandas as pd

# 1,pd.read_csv()  读取csv(逗号分隔)文件到DatFrame
#                 具体说明参考博客: https://www.cnblogs.com/datablog/p/6127000.html
#                   参数说明: sep: str, default ',' 指定分隔符，如果不指定参数，则会尝试使用逗号分隔
#                           delimiter: 定界符,备选分隔符(如果指定该参数，则sep参数失效)
#                           quoting: 控制csv中的引号常量
#                           doublequoting: 双引号，当单引号已经被定义，并且quoting参数不是QUOTE_NONE的时候，使用双引号表示引号内的一个元素作为一个元素使用。
#
# 2, pd.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)
#       将分类变量转换为虚拟/指示符变量
# 参数说明: data: array-like，Series或DataFrame
# x = pd.Series(list('abcdb'))
# df = pd.DataFrame({'A': ['a','b','c'], 'B': ['b','a','c'], 'C':[1,2,3]})
# y = pd.get_dummies(x)
# y1 = pd.get_dummies(df, prefix=['col1', 'col2'])
# print(y1)

# pd.to_datetime(*args, **kwargs)  将参数转换为datetime
# df = pd.DataFrame({'year':[2015, 2016], 'month':[2,4], 'day':[4,5]})
# x = pd.to_datetime(df.split())
# print(x)

# pd.get_dummies()  将分类变量转换为虚拟/指示符变量
# s = pd.Series(list('abca'))
# y = pd.get_dummies(s)
# print(y)
#
# pd.strftime()

# pd.DateOffset

# python pandas.DataFrame选取、修改数据最好用.loc，.iloc，.ix
df = pd.DataFrame(np.arange(0, 60, 2).reshape(10, 3), columns=list('abc'))
df.loc[0, 'a']
# print(df.loc[0, 'a'])
# print(df.loc[0:3, ['a', 'b']])
print(df.loc[[1, 5], ['b', 'c']])