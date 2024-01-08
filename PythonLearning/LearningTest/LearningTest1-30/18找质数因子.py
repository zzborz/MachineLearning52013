a=int(input("输入一个整数："))
y=2
list=[]
while y<=a:
    if a%y==0:
        list.append(y)
        a=a/y
    else:
        y+=1
print(list)