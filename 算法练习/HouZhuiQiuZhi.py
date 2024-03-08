from pythonds.basic.stack import Stack
def psotfixEval(postfixExpr):
    #生成空栈
    operandStack = Stack()
    tokenList=postfixExpr.split()
    #遍历后缀表达式
    for token in tokenList:
        #如果是数字，直接入栈
        if token in "0123456789":
            operandStack.push(int(token))
        else:
            #如果是运算符，弹出栈顶两个元素，进行运算，将结果入栈
            operand2=operandStack.pop()
            operand1=operandStack.pop()
            result=doMath(token,operand1,operand2)
            operandStack.push(result)
    return operandStack.pop()
def doMath(op,op1,op2):
    if op=="*":
        return op1*op2
    elif op=="/":
        return op1/op2
    elif op=="+":
        return op1+op2
    else:
        return op1-op2
print(psotfixEval('7 8 + 3 2 + /'))