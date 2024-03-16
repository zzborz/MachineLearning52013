class solution:
    def fib(self,n:int)->int:
        if n==0:
            return 0
        if n==1:
            return 1

        dp=[0 for _ in range(n+1)]#用于存储每个位置的值，dp【i】表示第i个位置的值
        dp[0]=0
        dp[1]=1

        for i in range(2,n+1):
            dp[1]=dp[i-1]+dp[i-2]
        return dp[n]
