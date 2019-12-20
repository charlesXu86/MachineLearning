#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Two_Sum.py
@time: 2018/1/24 18:57
@desc:
"""



'''
  思路:
     题目是哟输入一个数组和一个target，要在数组中找到两个数字，和为target，然后从小到大输出数组中数字的位置。
  假设只有一个答案。
   
  首先建立一个字典d={},字典的key是数组的值num，value是相应的位置，只要满足num和target-num都在字典里找到答案

'''

class Solution(object):
    def twoSum(self, nums, target):
        '''
        :param nums:
        :param target:
        :return:
        '''
        d = {}
        for i, num in enumerate(nums):
            if target - num in d:
                return [d[target - num], i]
            d[num] = i

    #
    if __name__=='__main__':
        nums = [2, 7, 11, 15]
        twoSum(nums, 9)
        print()
