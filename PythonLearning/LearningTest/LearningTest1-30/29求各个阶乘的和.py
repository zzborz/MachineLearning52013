import math
math.factorial(5)
Sum=0
for i in range(1,21):
    Sum+=math.factorial(i)
print(Sum)

Sum1=0
Sum2=1
for i in range(1,21):
    for j in range(1,i+1):
        Sum2*=j
    Sum1+=Sum2
    Sum2=1
print(Sum1)