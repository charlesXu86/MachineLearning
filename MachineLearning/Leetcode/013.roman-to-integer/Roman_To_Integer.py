#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Roman_To_Integer.py
@time: 2018/1/30 17:18
@desc: 将罗马数字转化为阿拉伯数字
"""

'''
  思路:
    罗马数字由五个字母组合而成，分别是I(1)、V(5)、X(10)、L(50)、C(100)、D(500)、M(1000)，
    当有连续的低级别和高级别的字母出现时，要减去低级别字母对应的数，比如IV代表-1+5=4，IX代表-1+10=9，XL代表-10+50=40......
    但是当高级别和低级别连续出现则不用，比如VI就代表5+1=6，LXXX代表50+10+10+10=80. 
    比如：1-12的罗马数字分别为Ⅰ、Ⅱ、Ⅲ、Ⅳ（IIII）、Ⅴ、Ⅵ、Ⅶ、Ⅷ、Ⅸ、Ⅹ、Ⅺ、Ⅻ
'''

'''
   用if 和 switch不好
   正确的做法是先构建一个字典
'''


class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str 输入的罗马字符
        :rtype: int
        """
        d = {'I':1,'V':5,'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
        res = 0
        for i in range(0, len(s) - 1):
            c = s[i]
            cafter = s[i + 1]
            if d[c] < d[cafter]:
                res += d[c]
            else:
                res -= d[c]
        res += d[s[-1]]
        return res

    if __name__ == '__main__':
        s = 'Ⅻ'
        romanToInt(s)
