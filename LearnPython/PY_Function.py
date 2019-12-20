# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     PY_Function
   Description :  python3的函数笔记
   Author :       charl
   date：          2018/8/6
-------------------------------------------------
   Change Activity:
                   2018/8/6:
-------------------------------------------------
"""

def printinfo(arg1, *vartuple ):
    print("输出：")
    print(arg1)
    print(vartuple)

printinfo(70, 60, 50)