from typing import List


class Solution:
    def fib(self,n:int)->int:
        #使用数组保存已经求解的f(k)的结果
        memo=[0 for _ in range(n+1)]
        return self.My_fib(n,memo)

    def My_fib(self,n:int,memo:List[int])->int:
        if n==0:
            return 0
        if n==1:
            return 1
        if memo[n]!=0:#已经求解过不为0的返回值
            return memo[n]
        memo[n]=self.my_fib(n-1,memo)+self.my_fib(n-2,memo)
        return memo[n]
