class Solution:
    def BubbleSort(selfself,nums:[int])->[int]:
        #第i趟排序
        for i in range(len(nums)-1):
            flag=False #标记是否发生交换
            #从数组中钱n-i+1个元素中，相邻两个元素进行比较
            for j in range(len(nums)-i-1):
                if nums[j]>nums[j+1]:#相邻两个元素进行比较，如果前者大于后者，则交换位置
                    nums[j],nums[j+1]=nums[j+1],nums[j]
                    flag=True
            if not flag:#如果没有发生交换，则说明数组已经有序
                break
        return nums

    def sortArray(self,nums:[int])->[int]:
        return self.BubbleSort(nums)


