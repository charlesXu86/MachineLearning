#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Reverse_integer.py
@desc: 翻转字符串(数字)
@time: 2018/01/29 
"""

class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        sign = x < 0 and -1 or 1
        x = abs(x)
        ans = 0
        while x:
            ans = ans * 10 + x % 10
            x = x / 10
        return sign * ans if ans <= 0x7fffffff else 0  # 0x7fffffff 的二进制表示除了首位是0，其他的都是1.
                                                # 就是说，这是最大的整形数。首位是符号位。0表示他是正数。
