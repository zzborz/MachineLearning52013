for i in range(1,101):
    if i%2 == 1:
        print(i)
    else:
        continue
for i in range(1,101):
    if i%2 != 0:
        print(i)
    else:
        continue
list=[]
for i in range(1,101):
    if i%2 == 1:
        list.append(i)
    else:
        continue
print(list)
list2=[]
for i in range(1,101,2):
    list2.append(i)
print(list2)