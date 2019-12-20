#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: BuppleSort.py
@time: 2018/1/25 10:04
@desc: 冒泡排序
"""

'''
  算法原理:
     1,从第一位开始比较相邻的两个元素。如果前者比后者大(由小到大排序)，那么就交换他们
     2.针对每一个两两相邻的元素都做比较操作，直到把所有元素比较完。这个时候最后一个元素是最大值
     3,此时我们再从头比较，重复第二步的操作，直到比较出倒数第二大的元素
     4,以此类推
  时间复杂度计算:分两种极端情况(1)全部有序   o(n) 只用比较一趟
                           (2)全部反序   
     
'''

import random

def BubbleSort(myList):
    length = len(myList)
    while length > 0:
        length -= 1
        cur = 0
        while cur < length:
            # 拿到当前元素
            if myList[cur] < myList[cur + 1]:
                myList[cur], myList[cur + 1] = myList[cur + 1], myList[cur]
            cur += 1
    return myList

if __name__ == '__main__':
    myList = [random.randint(1, 1000) for i in range(100)]
    BubbleSort(myList)
    print(myList)