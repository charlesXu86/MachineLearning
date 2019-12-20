# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     pandas_csv.py
   Description :  pandas操作scv
   Author :       charl
   date：          2018/9/7
-------------------------------------------------
   Change Activity: 2018/9/7:
-------------------------------------------------
"""

import pandas as pd

filename = 'D:\dataset\\en_community_content.csv'

data = df = pd.read_csv(filename, sep=",", encoding='latin-1', low_memory=False, error_bad_lines=False, delimiter='\t')

first_rows = data.head(n=5)

cols = data.columns  # 返回全部列名

dim = data.shape
print(dim)