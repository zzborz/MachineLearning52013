#方法一 非递归
n=10
sum=1
for i in range(1,n+1):
    sum*=i
print(sum)
#方法二 递归
def jiecheng(n):
    #递归的终止条件
    if n==1:
        return 1
    #递归的调用
    else:
        return n*jiecheng(n-1)
print(jiecheng(10))