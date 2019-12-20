#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Remove_Duplicates_From_Sorted_List.py
@time: 2018/2/7 18:28
@desc:
"""

'''
   For example,
    Given 1->1->2, return 1->2.
    Given 1->1->2->3->3, return 1->2->3
'''

'''
   python 链表的操作
'''
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(None)
        dummy.next = head
        p = dummy
        while p and p.next:
            if p.val == p.next.val:
                p.next = p.next.next
            else:
                p = p.next
        return dummy.next


if __name__ == '__main__':
    pass