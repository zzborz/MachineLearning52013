a=int(input("请输入左区间整数:"))
b=int(input("请输入右区间整数:"))
if a>b:
    print("左区间整数不能大于右区间整数")
else:
    for i in range(a,b+1):
        if i>1:
            for j in range(2,i):
                if i%j==0:
                    print(i,"是合数")
                    break
            else:
                print(i,"是素数")
        else:
            print(i,"既不是素数也不是合数")

# 方法2
def prime(n):
    flag=True
    for i in range(2,n):
        if n%i==0:
            flag=False
            break
    return flag
a=int(input("请输入左区间整数:"))
b=int(input("请输入右区间整数:"))
list=[]
for i in range(a,b+1):
    if prime(i):
        list.append(i)
print(list)