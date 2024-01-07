a=int(input("请输入第一条边:"))
b=int(input("请输入第二条边:"))
c=int(input("请输入第三条边:"))
if a<=0 or b<=0 or c<=0:
    print("输入的边长不能小于等于0")
if a+b<=c or a+c<=b or b+c<=a:
    print("输入的三条边不能构成三角形")
else:
    if a==b==c:
        print("输入的三条边构成等边三角形")
    elif a==b or a==c or b==c:
        print("输入的三条边构成等腰三角形")
    elif a*a+b*b==c*c or a*a+c*c==b*b or b*b+c*c==a*a:
        print("输入的三条边构成直角三角形")
    else:
        print("输入的三条边构成普通三角形")