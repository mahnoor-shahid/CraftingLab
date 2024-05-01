'''

217. Contains Duplicate

Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

Example 1:

Input: nums = [1, 2, 3, 1]
Output: true

Example 2:

Input: nums = [1, 2, 3, 4]
Output: false

Example 3:

Input: nums = [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
Output: true

Constraints:

1 <= nums.length <= 105
-109 <= nums[i] <= 109
'''

import time
from typing import List

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums = sorted(nums) ## time complexity is nlogn
        for i in range(0, len(nums)-1):
            if nums[i]==nums[i+1]: ## space complexity is 1
                return True
        return False

    def containsDuplicate_with_set(self, nums: List[int]) -> bool:
        hashmap = set() ## space complexity increases from 1 to n
        for n in nums: ## since there is no sorting time complexity is n
            if n in hashmap:
                return True
            else:
                hashmap.add(n)
        return False


start_time = time.time()
obj1 = Solution()
start_time = time.time()
print(obj1.containsDuplicate(nums=[1,2,3,4,5,6,2]))
end_time = time.time()
print(end_time - start_time)

start_time = time.time()
print(obj1.containsDuplicate(nums=[1,2,3,4,5,6,2]))
end_time = time.time()
print(end_time - start_time)


