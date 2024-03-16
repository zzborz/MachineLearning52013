from typing import List


class solution:
    def tribonacci(self,n:int)->int:
        memo=[0 for _ in range(n+1)]
        return self.my_tribonacci(n,memo)

    def my_tribonacci(self,n:int,memo:List[int])->int:
        if n==0:
            return 0
        if n==1 or n==2:
            return 1

        if memo[0]!=0:
            return memo[0]

        memo[n]=self.my_tribonacci(n-3,memo)+self.my_tribonacci(n-2,memo)+self.my_tribonacci(n-1,memo)
        return memo[n]