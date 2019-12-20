#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Plus_One.py
@time: 2018/2/6 16:51
@desc:
"""

'''
  解析:
    将一个数字的每个位上的数字分别存到一个一维向量中，最高位在最开头，我们需要给这个数字加一，即在末尾数字加一，
    如果末尾数字是9，那么则会有进位问题，而如果前面位上的数字仍为9，则需要继续向前进位。具体算法如下：
    首先判断最后一位是否为9，若不是，直接加一返回，若是，则该位赋0，再继续查前一位，同样的方法，知道查完第一位。
    如果第一位原本为9，加一后会产生新的一位，那么最后要做的是，查运算完的第一位是否为0，如果是，则在最前头加一个1。
'''

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry = 1
        for i in reversed(range(0, len(digits))):
            digit = (digits[i] + carry) % 10
            carry = 1 if digit < digits[i] else 0    # 很精妙
            digits[i] = digit
        if carry == 1:
            return [1] + digits
        return digits





if __name__ == '__main__':
    pass