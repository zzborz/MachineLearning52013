a=int(input("请输入第一条边:"))
b=int(input("请输入第二条边:"))
c=int(input("请输入第三条边:"))
p=(a+b+c)/2
#math.sqrt()是求平方根的函数
s=(p*(p-a)*(p-b)*(p-c))**0.5
print("三角形的面积是:",s)
