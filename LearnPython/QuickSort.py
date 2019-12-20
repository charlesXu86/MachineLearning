#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: QuickSort.py 
@desc: 快速排序实现
@time: 2017/12/21 
"""

'''
 快排的思想： 首先任意选取一个数据（通常选用数组的第一个数）作为关键数据，然后将所有比它小的数都放到它前面，
              所有比它大的数都放到它后面，这个过程称为一趟快速排序。
            
 百度百科给的算法：

一趟快速排序的算法是：
1）设置两个变量i、j，排序开始的时候：i=0，j=N-1；
2）以第一个数组元素作为关键数据，赋值给key，即key=A[0]；
3）从j开始向前搜索，即由后开始向前搜索(j--)，找到第一个小于key的值A[j]，将A[j]和A[i]互换；
4）从i开始向后搜索，即由前开始向后搜索(i++)，找到第一个大于key的A[i]，将A[i]和A[j]互换；
5）重复第3、4步，直到i=j； (3,4步中，没找到符合条件的值，即3中A[j]不小于key,4中A[i]不大于key的时候改变j、i的值，使得j=j-1，i=i+1，直至找到为止。
   找到符合条件的值，进行交换的时候i， j指针位置不变。另外，i==j这一过程一定正好是i+或j-完成的时候，此时令循环结束）。
   
 时间复杂度： O(nlgn)
'''

def QuickSort(myList, start, end):
    if start < end:
        i, j = start, end
        # 设置基准数
        base = myList[i]

        while i < j:
            # 如果列表后边的数比基准数大或者相等，则前移一位，直到出现比他小的
            while(i < j) and (myList[j] >= base):
                j = j - 1
            # 如果找到，则互换
            myList[i] = myList[j]

            while(i < j) and (myList[i] <= base):
                i = i + 1
            myList[j] = myList[i]

        # 做完第一轮之后，列表被分成两个半区，并且i=j，需要将这个数设置回base
        myList[i] = base

        #递归前后半区
        QuickSort(myList, start, i-1)
        QuickSort(myList, j + 1, end)
    return myList

myList = [1, 6, 7, 66, 78, 43, 23, 99, 26, 1111]
print('Quick Sort:')
QuickSort(myList, 0, len(myList) - 1)
print(myList)