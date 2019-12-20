#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Valid_Parentheses.py 
@desc:
@time: 2018/02/04 
"""

'''
  思路:
    栈最典型的应用就是验证配对情况，作为有效的括号，有一个右括号就必定有一个左括号在前面，
    所以我们可以将左括号都push进栈中，遇到右括号的时候再pop来消掉。
    这里不用担心连续不同种类左括号的问题，因为有效的括号对最终还是会有紧邻的括号对。如栈中是({[，来一个]变成({，再来一个}，变成(。
'''
'''
  了解堆栈的性质
'''

class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        d = ['()', '[]', '{}']
        for i in range(0, len(s)):
            stack.append(s[i])
            if len(stack) >2 and stack[-2] + stack[-1] in d:
                stack.pop()
                stack.pop()
        return len(stack) == 0