a = float(input("请输入a:"))
b = float(input("请输入b:"))
c = float(input("请输入c:"))
#方法1
list=[a,b,c]
list1=sorted(list)
print("最大值是:",list1[-1])
print("最小值是:",list1[0])
#方法2
print("最大值是:",max(a,b,c))
print("最小值是:",min(a,b,c))

