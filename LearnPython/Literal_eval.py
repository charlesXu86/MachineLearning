#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Literal_eval.py 
@desc: eval() 和 ast.literal_eval() 的区别
      https://www.programcreek.com/python/example/5578/ast.literal_eval
@time: 2017/11/20 
"""

import ast

def test_literal_eval(self):
    self.assertEqual(ast.literal_eval('[1,2,3]'), [1,2,3])

if __name__ == '__main__':
    test_literal_eval()