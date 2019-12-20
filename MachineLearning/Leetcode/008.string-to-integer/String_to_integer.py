#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: String_to_integer.py
@time: 2018/3/12 18:52
@desc: String to Integer (atoi)
"""

'''
  思路:
    
'''

class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        str = str.strip()
        sign = 1
        if not str:
            return 0
        if str[0] in ['+', '-']:
            if str[0] == '-':
                sign = -1
            str = str[1:]
        ans = 0
        for c in str:
            if c.isdigit():
                ans = ans * 10 + int(c)
            else:
                break
        ans = ans * sign
        if ans > 2147483647:
            return 2147483647
        if ans < -2147483647:
            return -2147483648
        return ans


if __name__ == '__main__':
    pass