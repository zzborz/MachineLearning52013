# **描述**：给定一个整数数组 $nums$。
#
# **要求**：对该数组升序排列。
#
# **说明**：
#
# - $1 \le nums.length \le 5 * 10^4$。
# - $-5 * 10^4 \le nums[i] \le 5 * 10^4$。
#
# **示例**：输入    nums = [5,2,3,1]
# 输出    [1,2,3,5]
# **提示**：请尝试使用 $O(nlogn)$ 时间复杂度的算法解决此问题。
from typing import List


class Solution:
    def merge(self,left_arr,right_arr): #合并两个有序数组
        arr=[]
        while left_arr and right_arr: #两个数组都不为空时
            if left_arr[0]<right_arr[0]:
                arr.append(left_arr.pop(0))
            else:
                arr.append(right_arr.pop(0))
        while left_arr:#当left_arr不为空时,则将所有结果都加入序列中
            arr.append(left_arr.pop(0))
        while right_arr:
            arr.append(right_arr.pop(0))
        return arr #返回排好序的结果数组

    #分解数组
    def merge_sort(self,arr):
        if len(arr)<=1:   #当数组长度小于等于1时,则直接返回
            return arr
        mid=len(arr)//2   #找到数组的中间位置
        left_arr=self.merge_sort(arr[:mid]) #递归将左边序列进行分解和排序
        right_arr=self.merge_sort(arr[mid:]) #递归将右边序列进行分解和排序
        return self.merge(left_arr,right_arr)#将当前序列组中有序子序列逐层向上，进行两两合并


    def sortArray(self,nums:List[int])->List[int]:#主函数
        return self.merge_sort(nums)

