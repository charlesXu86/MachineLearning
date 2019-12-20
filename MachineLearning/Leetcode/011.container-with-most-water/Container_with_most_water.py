#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Container_with_most_water.py 
@desc:
@time: 2018/07/14 
"""
'''
Given n non-negative integers a1, a2, ..., an, 
where each represents a point at coordinate (i, ai). 
n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). 
Find two lines, which together with x-axis forms a container, 
such that the container contains the most water.

Note: You may not slant the container and n is at least 2.

给一个数组，其中数组在下标i处的值为A[i]，坐标(i,A[i])和坐标(i,0)构成一条垂直于坐标轴x的直线。
现任取两条垂线和x轴组成四边形容器。问其中盛水量最大为多少？

思路：双指针



'''

class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        ans = left = 0
        right = len(height) - 1
        while left < right:
            ans = max(ans, (right - left) * min(height[left], height[right]))
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
        return ans


