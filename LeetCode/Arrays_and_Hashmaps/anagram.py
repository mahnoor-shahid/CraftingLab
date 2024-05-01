'''
242. Valid Anagram

Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Example 1:

Input: s = "anagram", t = "nagaram"
Output: true

Example 2:

Input: s = "rat", t = "car"
Output: false


Constraints:

1 <= s.length, t.length <= 5 * 104
s and t consist of lowercase English letters.
'''
import time
from collections import Counter


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        s = s.lower()
        t = t.lower()
        if len(s) == len(t):
            return sorted(s) == sorted(t)
        else:
            return False

    def isAnagram2(self, s: str, t: str) -> bool:
        s = s.lower()
        t = t.lower()
        if len(s) == len(t):
            return Counter(s) == Counter(t)
        else:
            return False
obj1 = Solution()

start_time= time.time()
print(obj1.isAnagram("anaGram","nagaram"))
print(time.time() - start_time)

start_time= time.time()
print(obj1.isAnagram2("anaGram","nagaram"))
print(time.time() - start_time)

