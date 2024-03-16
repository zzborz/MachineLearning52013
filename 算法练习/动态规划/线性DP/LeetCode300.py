from typing import List


class Solution:
    def length0fLIS(self,nums: List[int]) -> int:
        size=len(nums)
        dp=[1 for _ in range(size)]

        for i in range(size):
            for j in range(i+1):
                if nums[i]>nums[j]:
                    dp[i]=max(dp[i],dp[j]+1)
        return max(dp)
