#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Length_Of_Last_Word.py
@time: 2018/2/6 14:09
@desc:
"""

'''
  Example:

Input: "Hello World"
Output: 5
'''
'''
 思路:
   最简单的办法就是:
    
'''

class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        b = s.strip().split(' ')

        if b[-1] == '':
            return 0
        else:
            c = len(b[-1])
        return c



if __name__ == '__main__':
    pass