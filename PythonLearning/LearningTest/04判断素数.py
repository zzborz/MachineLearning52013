#素数是大于1分解只有他自己和1的数
a=37
flag=False
for i in range(2,a):
    if a%i==0:
        flag=True
        break
if flag:
    print("不是素数")
else:
    print("是素数")

