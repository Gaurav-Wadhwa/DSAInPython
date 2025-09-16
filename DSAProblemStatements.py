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

# Problem Statement 4
# Given a list of integers nums and an integer val, write a function remove_element
# that removes all occurrences of val from nums in-place and returns the new length
# of the modified list.
#
# The function should not allocate extra space for another list; instead, it should
# modify the input list in-place with O(1) extra memory.
#
# Constraints:
# - Only pop() can be used to remove elements.
# - It is acceptable to have unused space at the end of the list after removal.
#
# Input:
# - A list of integers nums.
# - An integer val representing the value to remove.
#
# Output:
# - An integer representing the new length of the list after removing val.

def remove_element(nums, val):
    i = 0
    while i < len(nums):
        if nums[i] == val:
            nums.pop(i)
        else:
            i += 1
    return len(nums)

#########################################################################################

# Problem Statement 5
# Write a Python function that takes a list of integers as input and returns a tuple
# containing the maximum and minimum values in the list.
#
# The function should have the following signature:
# def find_max_min(myList):
#
# The function should traverse the list and keep track of the current maximum
# and minimum values. It should then return these values as a tuple, with the
# maximum value as the first element and the minimum value as the second element.
#
# Example:
# Input:  [5, 3, 8, 1, 6, 9]
# Output: (9, 1)

def find_max_min(nums):
    min_val = nums[0]
    max_val = nums[0]
    for num in nums:
        if num < min_val:
            min_val = num
        elif num > max_val:
            max_val = num
        else:
            pass
    return (max_val, min_val)

#########################################################################################

# Problem Statement 6
# Write a Python function called find_longest_string that takes a list of strings
# as input and returns the longest string in the list.
#
# The function should iterate through each string in the list, check its length,
# and keep track of the longest string seen so far. Once it has looped through
# all the strings, the function should return the longest string found.
#
# Example:
# Input:  ['apple', 'banana', 'kiwi', 'pear']
# Output: 'banana'

def find_longest_string(string_list):
    longest = ""
    for string in string_list:
        if len(string) > len(longest):
            longest = string
    return longest

#########################################################################################

# Problem Statement 7
# Given a sorted list of integers, rearrange the list in-place such that all unique
# elements appear at the beginning of the list.
#
# The function should return the new length of the list containing only unique elements.
# The original list should be modified in-place, without using extra space.
#
# Constraints:
# - The input list is sorted in non-decreasing order.
# - The input list may contain duplicates.
# - Time complexity: O(n), where n is the length of the input list.
# - Space complexity: O(1), i.e., no extra data structures or lists may be used.
#
# Example:
# Input:  [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
# Output: new_length = 5, Modified list = [0, 1, 2, 3, 4, 2, 2, 3, 3, 4]
# Explanation: The first 5 elements of the list are unique: [0, 1, 2, 3, 4].

def remove_duplicates(nums):
    if len(nums) == 0:
        return 0
    j = 0
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            j += 1
            nums[j] = nums[i]
    return j + 1

#########################################################################################

# Problem Statement 8
# You are given a list of integers representing stock prices for a company over
# a period of time, where each element in the list corresponds to the stock price
# for a specific day.
#
# You are allowed to buy one share of the stock on one day and sell it on a later day.
# You are only allowed to complete one transaction (buy once and sell once).
#
# Write a function max_profit that returns the maximum profit you can achieve
# from this transaction. If no profit is possible, return 0.
#
# Constraints:
# - Each element of the input list is a positive integer representing the stock price.
# - You must buy before you sell.
# - Function signature: def max_profit(prices):
#
# Example:
# Input:  [7, 1, 5, 3, 6, 4]
# Output: 5
# Explanation: Buy on day 2 (price=1), sell on day 5 (price=6), profit = 6 - 1 = 5.

def max_profit(prices):
    max_profit = 0
    min_price = prices[0]
    for i in range(1, len(prices)):
        min_price = min(min_price, prices[i])
        profit = prices[i] - min_price
        max_profit = max(max_profit, profit)
    return max_profit

#########################################################################################

# Problem Statement 9
# You are given a list of n integers and a non-negative integer k.
#
# Write a function rotate that rotates the list to the right by k steps.
# The function should modify the list in-place and should not return anything.
#
# Constraints:
# - Each element of the input list is an integer.
# - The integer k is non-negative.
# - Function signature: def rotate(nums, k):
#
# Example:
# Input:  nums = [1, 2, 3, 4, 5, 6, 7], k = 3
# Output: nums = [5, 6, 7, 1, 2, 3, 4]
#
# Explanation:
# Rotating the list step by step:
# [7, 1, 2, 3, 4, 5, 6]
# [6, 7, 1, 2, 3, 4, 5]
# [5, 6, 7, 1, 2, 3, 4]

def rotate(nums, k):
    k = k % len(nums)
    nums[:] = nums[-k:] + nums[:-k]
    return nums

#########################################################################################

# Problem Statement 10
# Given an array of integers nums, write a function max_subarray(nums) that finds
# the contiguous subarray (containing at least one number) with the largest sum
# and returns its sum.
#
# Remember to also handle the case when the input array has 0 items.
#
# Function Signature:
# def max_subarray(nums):
#
# Input:
# - A list of integers nums.
#
# Output:
# - An integer representing the sum of the contiguous subarray with the largest sum.
#
# Example:
# Input:  [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# Output: 6
# Explanation: The subarray [4, -1, 2, 1] has the largest sum = 6.

def max_subarray(nums):
    if len(nums) == 0:
        return 0
    max_sum = nums[0]
    current = nums[0]
    for i in range(1, len(nums)):
        current = max(current + nums[i], nums[i])
        max_sum = max(max_sum, current)
    return max_sum

#########################################################################################
