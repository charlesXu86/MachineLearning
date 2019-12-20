#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Search_Insert_Position.py 
@desc:
@time: 2018/02/03 
"""

'''
Example 1:

Input: [1,3,5,6], 5
Output: 2
Example 2:

Input: [1,3,5,6], 2
Output: 1
Example 3:

Input: [1,3,5,6], 7
Output: 4
Example 1:

Input: [1,3,5,6], 0
Output: 0
'''

'''
 思路:
    考察二分查找。每次取中间，如果等于目标值就返回，否则根据大小关系切去一半

'''
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nlen = len(nums)
        pos = 0
        while pos < nlen:
            mid = pos + (nlen - pos) / 2
            if nums[mid] > target:   #大于
                nlen = mid
            elif nums[mid] < target: # 小于
                pos = mid + 1
            else:                    # 等于
                return mid
        return pos

