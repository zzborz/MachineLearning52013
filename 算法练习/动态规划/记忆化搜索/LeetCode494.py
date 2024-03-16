from typing import List


class Solution:
    def findTargetSumWays(self,nums:List[int],target:int)->int:
        Size=len(nums)

        def dfs(i,cus_sum):
            if i==Size:
                if cus_sum==target:
                    return 1
                else:
                    return 0
            ans=dfs(i+1,cus_sum-nums[i])+dfs(i+1,cus_sum+nums[i])
            return ans
        return dfs(0,0)

