#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Climing_Stairs.py
@time: 2018/2/7 10:05
@desc:
"""
'''
    Example 1:

    Input: 2
    Output:  2
    Explanation:  There are two ways to climb to the top.
    
    1. 1 step + 1 step
    2. 2 steps
    Example 2:
    
    Input: 3
    Output:  3
    Explanation:  There are three ways to climb to the top.
    
    1. 1 step + 1 step + 1 step
    2. 1 step + 2 steps
    3. 2 steps + 1 step
'''
'''
   n <= 1   return 1
   n > 1时，对于每一个台阶i，要到达台阶，最后一步都有两种方法，从i-1迈一步，或从i-2迈两步。
   也就是说达到台阶i的方法数 = 到达台阶i-1的方法数 + 到达台阶i-2的方法数。所以该问题是个DP问题。
   
   状态转移方程其实就是Fibonacci数列。
'''

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 1:
            return 1
        res = []
        res.append(1)
        res.append(1)
        for i in range(2, n + 1):
            res.append(res[-1] + res[-2])
        return res[-1]



if __name__ == '__main__':
    pass