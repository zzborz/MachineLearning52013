a=int(float(input("请输入一个四位数:")))
a1=a//1000
a2=a//100%10
a3=a//10%10
a4=a%10
print(a4,a3,a2,a1)
#方法2
a=int(float(input("请输入一个四位数:")))
a=str(a)
print(a[::-1])
#方法3
a=int(float(input("请输入一个四位数:")))
list=[]
for i in range(4):
    list.append(a%10)
    a=a//10
print(list)
