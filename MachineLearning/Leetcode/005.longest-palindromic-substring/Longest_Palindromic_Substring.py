#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Longest_Palindromic_Substring.py
@time: 2018/2/8 13:37
@desc:
"""

'''
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.
  
思路1：KMP匹配   字符串匹配
最长回文串有如下性质：
对于串S, 假设它的 Reverse是 S', 那么S的最长回文串是 S 和 S' 的最长公共字串。

例如 S = abcddca,  
     S'= acddcba， S和S'的最长公共字串是 cddc 也是S的最长回文字串。

如果S‘是 模式串，我们可以对S’的所有后缀枚举(S0, S1, S2, Sn) 然后用每个后缀和S匹配，寻找最长的匹配前缀。

例如当前枚举是 S0 = acddcba 最长匹配前缀是 a

S1  = cddcba 最长匹配前缀是 cddc

S2 = ddcba 最长匹配前缀是 ddc

'''

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        left = right = 0
        n = len(s)
        for i in range(n-1):
            if 2 * (n - i) + 1 < right - left + 1:
                break
            l = r = i
            while l >= 0 and r < n and s[l] == s[r]:
                l -= 1
                r += 1
            if r - l - 2 > right - left:
                left = l + 1
                right = r - 1
            l = i
            r = i + 1
            while l >= 0 and r < n and s[l] == s[r]:
                l -= 1
                r += 1
            if r - l - 2 > right - left:
                left = l + 1
                right = r - 1
        return s[left:right + 1]



if __name__ == '__main__':
    pass