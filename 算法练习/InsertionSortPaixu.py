class Solution:
    def insertionSort(self, nums: [int]) -> [int]:
        for i in range(1, len(nums)):  # 第i趟排序
            temp = nums[i]  # 待插入的元素
            j = i
            while j > 0 and nums[j - 1] > temp:  # 从后往前查找待插入的位置
                nums[j] = nums[j - 1]  # 向后移动
                j -= 1
            nums[j] = temp
        return nums

    def sortArray(self, nums: [int]) -> [int]:
        return self.insertionSort(nums)

# 思路：从第i个元素开始，前面的数组已经排好序，记录第i个的元素为temp，然后往前
# 查找j-1（j=i)的位置,temp比他大不插入。比他小，继续j--，将nums[j]被j-1的元素替换，
# 直到找到位置插入，
# 然后j位置的元素后移，j-1位置的元素赋值为temp
# 时间复杂度：O(n^2)
# 空间复杂度：O(1)
