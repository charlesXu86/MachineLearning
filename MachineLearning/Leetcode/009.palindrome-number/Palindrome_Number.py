#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Palindrome_Number.py
@time: 2018/1/31 8:47
@desc: 回文数
"""
'''
  思考:
    1,负数没有回文数
    2、若你想转化为字符串做题，你要考虑到不能用额外空间的限制。
	3、你也可以反转整数来操作，但要注意翻转后可能会溢出
	
  思路:
    1,将整形数字转化为字符串来操作。，回文数是关于中间位置对称的
    
    2,将整形数字反转后与原数字判断即可。
'''

class Solution(object):
    # 常规方法
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        z = x
        y = 0
        while x > 0:
            y = y * 10 + x % 10
            x = x / 10
        return z == y


    # 快速方法
    def _isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0 or (x != 0 and x % 10 == 0):
            return False
        half = 0
        while x > half:
            half = half * 10 + x % 10
            x = x / 10
        return x == half or half / 10 == x








if __name__ == '__main__':
    pass