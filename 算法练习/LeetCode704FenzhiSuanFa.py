# **描述**：给定一个含有 $n$ 个元素有序的（升序）整型数组 $nums$ 和一个目标值 $target$。
#
# **要求**：返回 $target$ 在数组 $nums$ 中的位置，如果找不到，则返回 $-1$。
#
# **说明**：
#
# - 假设 $nums$ 中的所有元素是不重复的。
# - $n$ 将在 $[1, 10000]$ 之间。
# - $-9999 \le nums[i] \le 9999$
from typing import List


class solution:
    def search(self,nums:List[int],target:int)->int: #二分查找分治算法
        left,right=0,len(nums)-1
        #在区间[left,right]中查找target
        while left<=right:#取中间节点
            mid=left+(right-left)//2
            #判断target在mid的左边还是右边
            if nums[mid]<target:
                left=mid+1
            else:
                right=mid
        #判断是否找到target
        return left if nums[left]==target else -1