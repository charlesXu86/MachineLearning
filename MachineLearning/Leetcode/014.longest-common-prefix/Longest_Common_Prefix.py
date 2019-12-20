#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Longest_Common_Prefix.py
@time: 2018/2/3 11:34
@desc: 寻找数组最大公共前缀
"""

'''
  解析:给定一个string类型数组，要求写一个方法，返回数组中这些字符串的最长公共前缀。
  
  思路:
     将str[0]当做临时最长公共前缀，然后用这个前缀依次和剩下的字符串比较，输出最大公共字符串
'''

class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
            return ""
        i = 0
        j = 0
        end = 0
        while j < len(strs) and i < len(strs[j]):
            if j == 0:
                char = strs[j][i]
            else:
                if strs[j][i] != char:
                    break

            if j == len(strs) - 1:
                i += 1
                j = 0
                end += 1
            else:
                j += 1
        return strs[j][:end]




if __name__ == '__main__':
    pass