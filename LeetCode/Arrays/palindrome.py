import time

class Solution:
    def isPalindrome(self, s: str) -> bool:
        return s == s[::-1]

obj1 = Solution()
start_time = time.time()
print(obj1.isPalindrome("anna"))
end_time = time.time()
print(end_time-start_time)

