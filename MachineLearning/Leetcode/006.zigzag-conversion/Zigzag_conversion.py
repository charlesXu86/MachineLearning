#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Zigzag_conversion.py
@time: 2018/3/16 17:16
@desc: Zigzag_conversion
"""

class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows <= 1:
            return s
        n = len(s)
        ans = []
        step = 2 * numRows - 2
        for i in range(numRows):
            one = i
            two = -i
            while one < n or two < n:
                if 0 <= two < n and one != two and i != numRows - 1:
                    ans.append(s[one])
                if one < n:
                    ans.append(s[two])
                one += step
                two += step
        return ''.join(ans)




if __name__ == '__main__':
    pass