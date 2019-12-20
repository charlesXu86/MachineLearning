#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Count_and_say.py
@time: 2018/2/1 8:48
@desc:
"""

'''
  这是一个递归问题:
  递归函数的特性:
    1,必须有一个明确的结束条件
    2,每次进入更深一层递归时，问题规模比上一次都应有所减少
    3,相邻两次重复之间有紧密的联系。
    4,递归效率不高。 
    
  =======================
1.     1
2.     11
3.     21
4.     1211
5.     111221  
 思路:
   统计每个数字出现的次数。所以要读出来的内容必须是字符型(char),不能是数字类型(int,float)
   关键是要将数组内容转换成字符型
'''
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        first = '1' #将第一个数转换成字符串
        for i in range(n-1):  # 第一行不需要读
            a, c, count = first[0], '', 0 # a用来读取上一行的第一个字符，c用来存储，count用来统计
            for j in first:
                if a == j:
                    count += 1
                else:
                    c += str(count) + a
                    a = j
                    count = 1
            c += str(count) + a
            first = c
        return first




if __name__ == '__main__':
    pass