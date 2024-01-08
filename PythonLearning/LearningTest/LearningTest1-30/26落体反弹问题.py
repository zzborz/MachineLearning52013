n=100
count=0
list=[]
while count<10:
    if count==0:
        list.append(n)
        n/=2
        count+=1
    else:
        list.append(n)
        n/=2
        count+=1
print(list)