'''
49. Group Anagrams

Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Example 2:

Input: strs = [""]
Output: [[""]]

Example 3:

Input: strs = ["a"]
Output: [["a"]]

'''
import time
from collections import Counter
from typing import List

class Solution:
    def groupAnagrams(self, strs:List[str]) -> List[List[str]]:
        hashmap = dict()
        for st in strs:
            freq_count = [0]*26
            for c in st:
                freq_count[ord(c) - ord('a')] += 1
            if str(freq_count) not in hashmap.keys():
                hashmap[str(freq_count)]=[st]
            else:
                hashmap[str(freq_count)].append(st)
        return hashmap.values()

obj1 = Solution()
start_time= time.time()
strs = ["eat","tea","tan","ate","nat","bat"]
print(obj1.groupAnagrams(strs))
end_time=time.time()
print(end_time-start_time)