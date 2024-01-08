string=input('请输入一串字符：')

# 统计字符个数
char=0
number=0
space=0
other=0
for i in string:
    if i.isalpha():
        char+=1
    elif i.isdigit():
        number+=1
    elif i.isspace():
        space+=1
    else:
        other+=1
print('char=%d,number=%d,space=%d,other=%d'%(char,number,space,other))