#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: split.py 
@desc: 基于jieba的分词
@time: 2017/10/27 
"""

import jieba

seg_list = jieba.cut("我来到杭州浙江大学", cut_all=True)
print("Full Mode:" + "/".join(seg_list))   #全模式

seg_list = jieba.cut("我来到杭州浙江大学", cut_all=False)
print("Default Mode:" + "/".join(seg_list))   #精确模式  默认是精确模式

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))



