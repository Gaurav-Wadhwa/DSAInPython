# Problem Statement 1
# You are given an array nums consisting of only 0, 1, and 2.
# Sort the array in-place so that all 0s come first, followed by all 1s, then all 2s.
# You must solve it without using a library sort function and ideally in one pass with O(1) extra space.

def swap(nums, index1, index2):
    temp = nums[index1]
    nums[index1] = nums[index2]
    nums[index2] = temp

def pivot(nums, pivot_index, end_index):
    swap_index = pivot_index
    for i in range(pivot_index + 1, end_index + 1):
        if nums[i] < nums[pivot_index]:
            swap_index += 1
            swap(nums, swap_index, i)
    swap(nums, pivot_index, swap_index)
    return swap_index

def quick_sort_helper(nums, left, right):
    if left < right:
        pivot_index = pivot(nums, left, right)
        quick_sort_helper(nums, left, pivot_index - 1)
        quick_sort_helper(nums, pivot_index + 1, right)
    return nums

def quick_sort(nums):
    return quick_sort_helper(nums, 0, len(nums) - 1)

#########################################################################################

# Problem Statement 2
# Given an array of integers nums (space separated integers) sorted in non-decreasing order, and a target value target,
# find the starting and ending position of the target value.
# If the target is not found, return [-1, -1].

def search_range(nums, target):
    def find_first(nums, target):
        left = 0
        right = len(nums) - 1
        first = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                first = mid
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return first

    def find_last(nums, target):
        left = 0
        right = len(nums) - 1
        last = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                last = mid
                left = mid + 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return last

    return [find_first(nums, target), find_last(nums, target)]

#########################################################################################

# Problem Statement 3
# You are given an integer array nums sorted in ascending order, but it has been rotated at some pivot unknown to you beforehand.
# Given the array and an integer target, return the index of target if it exists, otherwise return -1.
# You must write an algorithm with O(log n) runtime complexity.

def search_rotated_array(nums, target):
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

#########################################################################################