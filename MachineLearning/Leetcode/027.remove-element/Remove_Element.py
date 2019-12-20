#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Remove_Element.py
@time: 2018/2/5 8:55
@desc:
"""

'''
Example:

Given nums = [3,2,2,3], val = 3,
Your function should return length = 2, with the first two elements of nums being 2.
'''

class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        slow = -1
        for i in range(0, len(nums) - 1):
            if nums[i] != val:
                slow += 1
                nums[slow] = nums[i]
        return slow + 1

if __name__ == '__main__':
    pass