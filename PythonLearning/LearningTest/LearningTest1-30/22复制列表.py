list=[1,2,3,4]
list1=list
#浅复制
list[0]=5
print(list1)
#深拷贝
import copy
list=[1,2,3,4]
list1=copy.copy(list)
list[0]=5
print(list1)