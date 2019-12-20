#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Implement_strStr.py
@time: 2018/2/5 10:40
@desc:
"""

'''
Example 1:

Input: haystack = "hello", needle = "ll"
Output: 2
Example 2:

Input: haystack = "aaaaa", needle = "bba"
Output: -1
'''

class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if len(haystack) == len(needle):
            if haystack == needle:
                return 0
            else:
                return -1
        for i in range(len(haystack)):
            k = i
            j = 0
            while j < len(needle) and k < len(haystack) and haystack[k] == needle[j]:
                k += 1
                j += 1
            if j == len(needle):
                return i
        return -1 if needle else 0  # ?

if __name__ == '__main__':
    pass