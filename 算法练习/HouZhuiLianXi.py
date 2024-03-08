from pythonds.basic.stack import Stack


def infixToPostfix(infixexpr):
    ##生成字典
    prec = {}
    ##定义运算符优先级
    prec['*'] = 3
    prec['/'] = 3
    prec['+'] = 2
    prec['-'] = 2
    prec['('] = 1
    ##生成空栈存储运算符
    opStack = Stack()
    ##生成空列表存储后缀表达式
    postfixList = []
    ##将后缀表达式分割成列表
    tokenList = infixexpr.split()
    print(tokenList)
    ##遍历后缀表达式
    for token in tokenList:
        ##如果是数字，直接加入后缀表达式或者字符串
        if token in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in "0123456789":
            postfixList.append(token)
        ##如果是左括号，直接入栈
        elif token == '(':
            opStack.push(token)
        ##如果是右括号，弹出栈中所有运算符，直到遇到左括号
        elif token == ')':
            TopToken = opStack.pop()
            while TopToken != '(':
                postfixList.append(TopToken)
                TopToken = opStack.pop()
        else:
            ##如果是运算符，弹出栈中所有优先级大于或等于该运算符的运算符
            while (not opStack.isEmpty()) and (prec[opStack.peek()] >= prec[token]):
                postfixList.append(opStack.pop())
            opStack.push(token)
        ##将栈中剩余的运算符加入后缀表达式
    while not opStack.isEmpty():
        postfixList.append(opStack.pop())
        ##使用空格连接后缀表达式中的元素
    return "".join(postfixList)


print(infixToPostfix("( A + B ) * C - ( D - E ) * ( F + G )"))
