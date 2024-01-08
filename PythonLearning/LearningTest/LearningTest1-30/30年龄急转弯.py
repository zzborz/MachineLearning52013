def func(n):
    if n==1:
        return 10
    else:
        return 2+func(n-1)
print(func(5))