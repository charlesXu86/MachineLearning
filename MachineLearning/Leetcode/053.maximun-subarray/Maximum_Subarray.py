#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Maximum_Subarray.py
@time: 2018/2/6 8:42
@desc: 求最大子数组的和
"""

'''
  For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
  the contiguous subarray [4,-1,2,1] has the largest sum = 6.
'''

'''
  思路分析:
     1, O(n)的解法:
        定义两个变量res和curSum,res保存最终要返回的结果，即最大子数组的和，curSum初始值为0，每遍历
        一个num，就比较curSum + num 和 num 中较大的值存入curSum.然后再把res和curSum中的较大值存入res。
        以此类推，遍历完整个数组
        
     2,O(nlogn)的解法 (分治法 -> devide and conque)
'''

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        preSum = maxSum = nums[0]
        for i in range(1, len(nums)):
            preSum = max(preSum + nums[i], nums[i])
            maxSum = max(preSum, maxSum)
        return maxSum



if __name__ == '__main__':
    pass