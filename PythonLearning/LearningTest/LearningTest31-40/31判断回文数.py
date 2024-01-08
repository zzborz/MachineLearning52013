a=int(input("请输入数字："))

a=str(a)
b=a[::-1]#字符串反转
if a==b:
    print("是回文数")
else:
    print("不是回文数")

