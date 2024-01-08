a=input("请输入一个字符:")
result=a.isalpha()
if result==True:
    print("输入的是字母")
else:
    print("输入的不是字母")

#方法2
a=input("请输入一个字符:")
if a>="a" and a<="z" or a>="A" and a<="Z":
    print("输入的是字母")
else:
    print("输入的不是字母")
