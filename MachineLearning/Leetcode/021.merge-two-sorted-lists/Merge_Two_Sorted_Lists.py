#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Merge_Two_Sorted_Lists.py 
@desc: 合并两个链表，从小到大排序
@time: 2018/02/03 
"""

'''
   Input: 1->2->4, 1->3->4
   Output: 1->1->2->3->4->4
'''
'''
   思路:
     新建一个链表，然后用两个链表当前位置去比较，谁的小就放谁。
     当一个链表放完之后，就把另一个链表剩下的元素再放进去。(因为他都比较大)
'''

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1 and not l2:
            return
        result = ListNode(0)
        l = result
        while l1 and l2:
               if l1.val < l2.val:
                    l.next = l1
                    l1 = l1.next
               else:
                    l.next = l2
                    l2 = l2.next
               # 融合后的链表的下一位
               l = l.next
        # 把剩余的放在后面
        l.next = l1 or l2
        # 返回融合后链表从第二个对象开始，第一个是ListNode(0)
        return result.next

if __name__=='__main__':
    arr1 = [1,2,4]
    arr2 = [4,5,6]
    l1 = ListNode(arr1[0])
    p1 = l1
    l2 = ListNode(arr2[0])
    p2 = l2
    for i in arr1[1:]:
        p1.next = ListNode(i)
        p1 = p1.next
    for i in arr2[1:]:
        p2.next = ListNode(i)
        p2 = p2.next
    s = Solution()
    q = s.mergeTwoLists(l1, l2)
    print(q)


