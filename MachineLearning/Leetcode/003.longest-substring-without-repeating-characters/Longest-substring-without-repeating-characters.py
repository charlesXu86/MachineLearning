#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Longest-substring-without-repeating-characters.py 
@desc:  求一个字符串不重复的最大长度
@time: 2018/01/29 
"""

'''
  思路:
     遍历字符串中的每个元素。借助一个辅助键值对来存储某个元素最后一次出现的下标。
     用一个整形变量存储当前无重复字符串的子串开始的下标
  复杂度分析:
    时间复杂度O(n)   空间复杂度O(n)
'''

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str ,  待处理的字符串
        :rtype: int
        """
        res = 0
        if s is None or len(s) == 0:
            return res
        d = {}
        tmp = 0
        start = 0
        for i in range(len(s)):
            if s[i] in d and d[s[i]] > start:
                start = d[s[i]] + 1
            tmp = i - start + 1
            d[s[i]] = i
            res = max(res, tmp)
        return res


    # 经典简洁版本
    def _lengthOfLongestSubstring(self, s):
        """
        :type s: str ,  待处理的字符串
        :rtype: int
        """
        d = {}
        start = 0
        ans = 0
        for i, c in enumerate(s):
            if c in d:
                start = max(start, d[c] + 1)
            d[c] = i
            ans = max(ans, i-start+1)
        return ans



    if __name__=='__main__':
        str = ['120135435', 'abcabcadd', 'bbbbb']
        for s in str:
            res = lengthOfLongestSubstring(s)
            print('{0}最大子字符串为:{1}'.format(s, res))
