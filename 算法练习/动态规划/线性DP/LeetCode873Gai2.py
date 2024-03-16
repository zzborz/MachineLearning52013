from typing import List


class Solution:
    def lenLongestFibSubseq(self,arr:List[int])->int:
        size=len(arr)
        dp=[[0 for _ in range(size)] for _ in range(size)]
        ans=0

        #初始化dp
        for i in range(size):
            for j in range(i + 1, size):
                dp[i][j] = 2

        idx_map={}
        #将value:idx映射为哈希表，这样可以快速通过value 获取到idx
        for idx,value in enumerate(arr):
            idx_map[value]=idx

        for i in range(size):
            for j in range(i+1,size):
                if arr[i]+arr[j] in idx_map:
                    #获取arr[i]+arr[j]的idx，即斐波那契数列子序列的下一项元素
                    k=idx_map[arr[i]+arr[j]]
                    #更新dp[i][j]
                    dp[j][k]=max(dp[j][k],dp[i][j]+1)
                    ans=max(ans,dp[j][k])
        if ans>=3:
            return ans
        return 0

#gptest
