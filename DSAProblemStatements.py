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

# Problem Statement 11
# You are given an integer array nums. Your task is to return the length of the
# longest strictly increasing subsequence in the array.
#
# A subsequence is a sequence that can be derived by deleting some or no elements
# from the array without changing the order of the remaining elements.
#
# Input:
# - nums (List[int]): A list of integers.
#
# Output:
# - An integer representing the length of the longest strictly increasing subsequence.
#
# Examples:
# Input:  [10, 9, 2, 5, 3, 7, 101, 18]
# Output: 4
# Explanation: The LIS is [2, 3, 7, 101].
#
# Input:  [7, 7, 7, 7, 7, 7, 7]
# Output: 1
#
# Input:  [0, 1, 0, 3, 2, 3]
# Output: 4

def longest_increasing_subsequence(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

#########################################################################################

# Problem Statement 12
# You are given an integer array prices where prices[i] represents the price of a 
# given stock on the i-th day.
#
# On each day, you may decide to buy and/or sell the stock. However, you can only 
# hold at most one share of the stock at any time. You can also buy and then 
# immediately sell the stock on the same day.
#
# Find and return the maximum profit you can achieve.
#
# Input:
# - prices (List[int]): An array representing the prices of the stock on each day.
#
# Output:
# - An integer representing the maximum profit you can achieve.
#
# Examples:
# Input:  [7,1,5,3,6,4]
# Output: 7
# Explanation: Buy on day 2 (price=1) and sell on day 3 (price=5), profit=4. 
#              Then buy on day 4 (price=3) and sell on day 5 (price=6), profit=3. 
#              Total profit = 7.
#
# Input:  [1,2,3,4,5]
# Output: 4
# Explanation: Buy on day 1 (price=1) and sell on day 5 (price=5). Profit = 4.
#
# Input:  [7,6,4,3,1]
# Output: 0
# Explanation: No profitable transactions are possible.

def max_profit(prices):
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    return profit

#########################################################################################

# Problem Statement 13
# The Tribonacci sequence is defined as follows:
# T0 = 0, T1 = 1, T2 = 1
# For n >= 0, Tn+3 = Tn + Tn+1 + Tn+2
#
# Given an integer n, return the value of Tn.
# Use a dynamic programming approach to optimize the solution.
#
# Constraints:
# - 0 ≤ n ≤ 37
# - Function signature: def tribonacci(n):
#
# Examples:
# Input:  4
# Output: 4
# Explanation: T3 = 0 + 1 + 1 = 2, T4 = 1 + 1 + 2 = 4
#
# Input:  25
# Output: 1389537

def tribonacci(n):
    dp = [0, 1, 1]
    for i in range(3, n + 1):
        dp.append(dp[i - 3] + dp[i - 2] + dp[i - 1])
    return dp[n]

#########################################################################################

# Problem Statement 14
# Given an integer numRows, return the first numRows of Pascal's triangle.
# In Pascal's triangle, each number is the sum of the two numbers directly
# above it. The first row is row 0, which is [1].
#
# Constraints:
# - numRows >= 1
# - Function signature: def generate(numRows):
#
# Input:
# - numRows (int): The number of rows of Pascal's triangle to generate.
#
# Output:
# - List[List[int]]: A list of lists where each list represents a row in Pascal's triangle.
#
# Examples:
# Input:  3
# Output: [
#   [1],
#   [1, 1],
#   [1, 2, 1]
# ]
#
# Input:  1
# Output: [
#   [1]
# ]
#
# Input:  5
# Output: [
#   [1],
#   [1, 1],
#   [1, 2, 1],
#   [1, 3, 3, 1],
#   [1, 4, 6, 4, 1]
# ]

def generate(numRows):
    if numRows == 0:
        return []
    triangle = [[1]]
    for i in range(1, numRows):
        new_row = [1]
        for j in range(1, i):
            new_row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])
        new_row.append(1)
        triangle.append(new_row)
    return triangle

#########################################################################################

# Problem Statement 15
# Given an integer rowIndex, return the kth (0-indexed) row of Pascal's Triangle.
#
# In Pascal's Triangle:
# - Each row starts and ends with 1.
# - Each element in between is the sum of the two elements directly above it.
#
# Constraints:
# - 0 <= rowIndex <= 33
# - Function signature: def getRow(rowIndex):
#
# Input:
# - rowIndex (int): The index of the row to return (0-indexed).
#
# Output:
# - List[int]: The kth row of Pascal's Triangle.
#
# Examples:
# Input: 3
# Output: [1, 3, 3, 1]
#
# Input: 0
# Output: [1]
#
# Input: 1
# Output: [1, 1]

def get_row(rowIndex):
    dp = [0] * (rowIndex + 1)
    dp[0] = 1
    for i in range(1, rowIndex + 1):
        for j in range(i, 0, -1):
            dp[j] = dp[j] + dp[j - 1]
    return dp

#########################################################################################

# Problem Statement 16
# You are given an integer array cost where cost[i] is the cost of the i-th step
# on a staircase. Once you pay the cost, you can either climb one or two steps.
#
# You can either start from step 0 or step 1. Return the minimum cost to reach
# the top of the floor.
#
# Constraints:
# - 2 <= len(cost) <= 1000
# - Each cost[i] is a non-negative integer
# - Function signature: def minCostClimbingStairs(cost):
#
# Input:
# - cost (List[int]): An array of integers where cost[i] represents the cost of the i-th step.
#
# Output:
# - An integer representing the minimum cost to reach the top.
#
# Examples:
# Input:  [10, 15, 20]
# Output: 15
# Explanation: Start at index 1, pay 15, and climb two steps to reach the top.
#
# Input:  [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
# Output: 6
# Explanation: Take steps 0 → 2 → 4 → 6 → 7 → 9 → top with total cost = 6.

def min_cost_climbing_stairs(cost):
    dp = [None] * (len(cost) + 1)
    dp[0], dp[1] = 0, 0
    for i in range(2, len(dp)):
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
    return dp[-1]

#########################################################################################

# Problem Statement 17
# You are climbing a staircase. It takes n steps to reach the top. Each time you
# can either climb 1 step or 2 steps. Return the number of distinct ways you can
# climb to the top.
#
# Constraints:
# - 1 <= n <= 45
# - Function signature: def climbStairs(n):
#
# Input:
# - n (int): The total number of steps required to reach the top.
#
# Output:
# - An integer representing the number of distinct ways to reach the top.
#
# Examples:
# Input:  2
# Output: 2
# Explanation: There are two ways to climb to the top:
# 1. 1 step + 1 step
# 2. 2 steps
#
# Input:  3
# Output: 3
# Explanation: There are three ways to climb to the top:
# 1. 1 step + 1 step + 1 step
# 2. 1 step + 2 steps
# 3. 2 steps + 1 step

def climb_stairs(n):
    if n == 1 or n == 2:
        return n
    dp = [None] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, len(dp)):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

#########################################################################################

# Problem Statement 18
# You are a professional robber planning to rob houses along a street. Each house
# has a certain amount of money stashed. The only constraint stopping you from
# robbing each of them is that adjacent houses have security systems connected,
# and it will automatically contact the police if two adjacent houses are broken
# into on the same night.
#
# Given an integer array nums representing the amount of money at each house,
# return the maximum amount of money you can rob tonight without alerting the police.
#
# Constraints:
# - 1 <= len(nums) <= 100
# - 0 <= nums[i] <= 400
# - Function signature: def rob(nums):
#
# Input:
# - nums (List[int]): An array representing the amount of money in each house.
#
# Output:
# - An integer representing the maximum money you can rob without triggering alarms.
#
# Examples:
# Input:  [1, 2, 3, 1]
# Output: 4
# Explanation: Rob house 1 (1) + house 3 (3) = 4.
#
# Input:  [2, 7, 9, 3, 1]
# Output: 12
# Explanation: Rob house 1 (2) + house 3 (9) + house 5 (1) = 12.

def rob(nums):
    if len(nums) == 0:
        return 0
    if len(nums) == 1:
        return nums[0]
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])
    return dp[-1]

#########################################################################################

# Problem Statement 19
# Given a triangle array, return the minimum path sum from top to bottom.
# For each step, you may move to an adjacent number of the row below.
# More formally, if you are on index i on the current row, you may move to either
# index i or index i+1 on the next row.
#
# Constraints:
# - 1 <= len(triangle) <= 200
# - -10^4 <= triangle[i][j] <= 10^4
# - Function signature: def minimumTotal(triangle):
#
# Input:
# - triangle (List[List[int]]): A list of lists where each inner list represents a row in the triangle.
#
# Output:
# - An integer representing the minimum path sum from the top to the bottom.
#
# Examples:
# Input:  [[2], [3,4], [6,5,7], [4,1,8,3]]
# Output: 11
# Explanation: The path with the minimum sum is 2 → 3 → 5 → 1 = 11.
#
# Input:  [[-10]]
# Output: -10
# Explanation: There is only one element in the triangle.

def minimumTotal(triangle):
    for row in range(len(triangle) - 2, -1, -1):
        for col in range(row + 1):
            triangle[row][col] += min(triangle[row + 1][col], triangle[row + 1][col + 1])
    return triangle[0][0]

#########################################################################################

# Problem Statement 20
# Given an n × n array of integers matrix, return the minimum sum of any falling
# path through the matrix.
#
# A falling path starts at any element in the first row and chooses the element
# in the next row that is either directly below or diagonally left/right.
# Specifically, the next element from position (row, col) will be one of:
# (row + 1, col - 1), (row + 1, col), or (row + 1, col + 1).
#
# Constraints:
# - 1 <= n <= 100
# - -100 <= matrix[i][j] <= 100
# - Function signature: def minFallingPathSum(matrix):
#
# Input:
# - matrix (List[List[int]]): A 2D list where each inner list represents a row of the matrix.
#
# Output:
# - An integer representing the minimum sum of any falling path through the matrix.
#
# Examples:
# Input:  [[2, 1, 3], [6, 5, 4], [7, 8, 9]]
# Output: 13
# Explanation: The minimum path sum is 2 → 1 → 4 = 13.
#
# Input:  [[-19, 57], [-40, -5]]
# Output: -59
# Explanation: The minimum path sum is -19 → -40 = -59.

def min_falling_path_sum(matrix):
    n = len(matrix)
    for row in range(n - 2, -1, -1):
        for col in range(n):
            if col == 0:
                matrix[row][col] += min(matrix[row + 1][col], matrix[row + 1][col + 1])
            elif col == len(matrix[row]) - 1:
                matrix[row][col] += min(matrix[row + 1][col - 1], matrix[row + 1][col])
            else:
                matrix[row][col] += min(matrix[row + 1][col - 1], matrix[row + 1][col], matrix[row + 1][col + 1])
    return min(matrix[0])

#########################################################################################

# Problem Statement 21
# Given an n × n integer matrix grid, return the minimum sum of a falling path
# with non-zero shifts. A falling path with non-zero shifts is a choice of
# exactly one element from each row of grid such that no two elements chosen in
# adjacent rows are in the same column.
#
# Constraints:
# - 1 <= n <= 200
# - -99 <= grid[i][j] <= 99
# - Function signature: def minFallingPathSum(grid):
#
# Input:
# - grid (List[List[int]]): A 2D list where each inner list represents a row of the matrix.
#
# Output:
# - An integer representing the minimum sum of a falling path with non-zero shifts.
#
# Examples:
# Input:  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# Output: 13
# Explanation: The minimum falling path sum is 1 → 5 → 7 = 13.
#
# Input:  [[5, 1, 3], [2, 4, 6], [7, 8, 9]]
# Output: 11
# Explanation: The minimum falling path sum is 1 → 2 → 8 = 11.

def min_falling_path_sum(grid):
    n = len(grid)
    dp = grid[-1]
    for row in range(n - 2, -1, -1):
        new_dp = [0] * n
        min1 = float('inf')
        min1_idx = -1
        min2 = float('inf')
        for col in range(n):
            if dp[col] < min1:
                min2 = min1
                min1 = dp[col]
                min1_idx = col
            elif dp[col] < min2:
                min2 = dp[col]
        for col in range(n):
            if col == min1_idx:
                new_dp[col] = grid[row][col] + min2
            else:
                new_dp[col] = grid[row][col] + min1
        dp = new_dp
    return min(dp)

#########################################################################################

# Problem Statement 22
# There is a robot on an m × n grid. The robot starts at the top-left corner
# (i.e., grid[0][0]) and aims to reach the bottom-right corner
# (i.e., grid[m - 1][n - 1]). The robot can only move either down or right
# at any point in time.
#
# Constraints:
# - 1 <= m, n <= 100
# - The answer is guaranteed to be less than or equal to 2 * 10^9
# - Function signature: def uniquePaths(m, n):
#
# Input:
# - m (int): Number of rows in the grid.
# - n (int): Number of columns in the grid.
#
# Output:
# - An integer representing the number of unique paths from top-left
#   to bottom-right corner.
#
# Examples:
# Input:  m = 3, n = 7
# Output: 28
# Explanation: There are 28 unique paths from the top-left to the bottom-right corner.
#
# Input:  m = 3, n = 2
# Output: 3
# Explanation: There are 3 unique paths from the top-left to the bottom-right corner.

def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    for row in range(1, m):
        for col in range(1, n):
            dp[row][col] = dp[row - 1][col] + dp[row][col - 1]
    return dp[-1][-1]

#########################################################################################

# Problem Statement 23
# You are given an m × n integer grid. There is a robot initially located at the
# top-left corner (i.e., grid[0][0]) and it tries to move to the bottom-right
# corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or
# right at any point in time.
#
# An obstacle and space are marked as 1 and 0 respectively in grid. A path that
# the robot takes cannot include any square that is an obstacle.
#
# Constraints:
# - 1 <= m, n <= 100
# - grid[i][j] is either 0 (open space) or 1 (obstacle)
# - The starting cell and ending cell may also be obstacles
# - Function signature: def uniquePathsWithObstacles(grid):
#
# Input:
# - grid (List[List[int]]): An m × n grid where grid[i][j] is either 0 (open space)
#   or 1 (obstacle).
#
# Output:
# - An integer representing the number of possible unique paths from the top-left
#   corner to the bottom-right corner avoiding obstacles.
#
# Examples:
# Input:  [[0,0,0],[0,1,0],[0,0,0]]
# Output: 2
# Explanation: There are 2 unique paths from the top-left to the bottom-right corner.
#
# Input:  [[0,1,0],[0,0,0],[0,0,0]]
# Output: 3
# Explanation: There are 3 unique paths from the top-left to the bottom-right corner.

def unique_paths_with_obstacles(grid):
    m = len(grid)
    n = len(grid[0])
    dp = [[0] * n for _ in range(m)]
    if grid[0][0] == 0:
        dp[0][0] = 1
    for row in range(1, m):
        if grid[row][0] == 0:
            dp[row][0] = dp[row - 1][0]
    for col in range(1, n):
        if grid[0][col] == 0:
            dp[0][col] = dp[0][col - 1]
    for row in range(1, m):
        for col in range(1, n):
            if grid[row][col] == 0:
                dp[row][col] = dp[row - 1][col] + dp[row][col - 1]
    return dp[-1][-1]

#########################################################################################

# Problem Statement 24
# You are given an array ARR of integers of size N. Your task is to determine:
# 1. The number of unique elements that occur an odd number of times.
# 2. The number of unique elements that occur an even number of times.
#
# Use a hashmap (Python dictionary) to track the frequency of each element.
#
# Input:
# - ARR (List[int]): A list of integers.
# - N (int): The size of the array.
#
# Output:
# - A tuple of two integers:
#   - The first integer is the count of elements occurring an odd number of times.
#   - The second integer is the count of elements occurring an even number of times.
#
# Examples:
# Input:  [1, 2, 3, 2, 3, 3]
# Output: (2, 1)
# Explanation: 1 occurs once (odd), 2 occurs twice (even), 3 occurs three times (odd).
#              Odd = 2 (1,3), Even = 1 (2).
#
# Input:  [5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7]
# Output: (1, 2)
# Explanation: 5 occurs 6 times (even), 6 occurs 4 times (even), 7 occurs once (odd).
#              Odd = 1 (7), Even = 2 (5,6).

def count_odd_even_occurrences(ARR):
    my_dict = {}
    for i in ARR:
        my_dict[i] = my_dict.get(i, 0) + 1
    odd = 0
    even = 0
    for key, value in my_dict.items():
        if value % 2 != 0:
            odd += 1
        else:
            even += 1
    return (odd, even)

#########################################################################################

# Problem Statement 25
# You are given an array of strings strs. Your task is to group the anagrams together 
# and return the result. An anagram is a word formed by rearranging the letters of 
# another, using all the original letters exactly once.
#
# Input:
# - strs (List[str]): A list of strings.
#
# Output:
# - List[List[str]]: A list of lists, where each inner list contains strings that are 
#   anagrams of each other. The order of the output and the order of strings within 
#   each group does not matter.
#
# Examples:
# Input:  ["eat", "tea", "tan", "ate", "nat", "bat"]
# Output: [["eat","tea","ate"], ["tan","nat"], ["bat"]]
#
# Input:  [""]
# Output: [[""]]
#
# Input:  ["a"]
# Output: [["a"]]

def group_anagrams(strs):
    anagrams = {}
    for s in strs:
        canonical = "".join(sorted(s))
        if canonical in anagrams:
            anagrams[canonical].append(s)
        else:
            anagrams[canonical] = [s]
    return list(anagrams.values())

#########################################################################################

# Problem Statement 26
# You are given an array of integers where the numbers are in random order. 
# Your task is to find and return the number that occurs the most times in the array. 
# If there are two or more numbers with the maximum frequency, return the number 
# that appears first in the array (i.e., the number with the lowest index).
#
# Input:
# - arr (List[int]): A list of integers.
#
# Output:
# - An integer which is the most frequent number in the array. If there are ties, 
#   the number that appears first should be returned.
#
# Examples:
# Input:  [1, 3, 2, 2, 1, 1, 4, 5, 1]
# Output: 1
# Explanation: The number 1 appears 4 times (most frequent).
#
# Input:  [4, 4, 5, 5, 6, 6, 6]
# Output: 6
# Explanation: The number 6 appears 3 times (most frequent). 
#              Even though 4 and 5 each appear twice, 6 occurs more often.

def most_frequent_number(arr):
    freq = {}
    for n in arr:
        freq[n] = freq.get(n, 0) + 1
    max_freq = 0
    mode = None
    for n in arr:
        if freq[n] > max_freq:
            max_freq = freq[n]
            mode = n
    return mode

#########################################################################################

# Problem Statement 27
# You are given an array ARR of integers. Your task is to find the greatest number 
# in the array such that this number is also equal to the product of two different 
# elements from the same array.
#
# If no such number exists, return -1.
#
# Input:
# - ARR (List[int]): A list of integers.
#
# Output:
# - An integer which is the greatest number that is also the product of two different 
#   elements in the array. If no such number exists, return -1.
#
# Examples:
# Input:  [1, 2, 3, 6, 12]
# Output: 12
# Explanation: The number 12 is present in the array and is equal to 2 * 6.
#
# Input:  [4, 2, 3, 8]
# Output: 8
# Explanation: The number 8 is present in the array and is equal to 2 * 4.
#
# Input:  [5, 7, 11]
# Output: -1
# Explanation: None of the numbers can be expressed as a product of two different 
# elements from the array.

def greatest_product_equal_to_element(arr):
    s = set(arr)
    sorted_arr = sorted(arr, reverse = True)
    for x in sorted_arr:
        for a in s:
            if x % a == 0 and a != x:
                b = x // a
                if b in s:
                    return x
    return -1

#########################################################################################

# Problem Statement 28
# You are given an array of integers nums and an integer target. Your task is to 
# find two distinct indices in the array such that the sum of the elements at these 
# indices equals the given target.
#
# You must return the indices in a list. You may not use the same element twice. 
# You can assume that each input has exactly one solution.
#
# Input:
# - nums (List[int]): A list of integers.
# - target (int): The target sum.
#
# Output:
# - List[int]: A list containing two indices [i, j] such that nums[i] + nums[j] = target.
#
# Examples:
# Input:  nums = [2, 7, 11, 15], target = 9
# Output: [0, 1]
# Explanation: nums[0] + nums[1] = 2 + 7 = 9
#
# Input:  nums = [3, 2, 4], target = 6
# Output: [1, 2]
# Explanation: nums[1] + nums[2] = 2 + 4 = 6
#
# Input:  nums = [3, 3], target = 6
# Output: [0, 1]
# Explanation: nums[0] + nums[1] = 3 + 3 = 6

def two_sum(nums, target):
    my_dict = {}
    for i in range(len(nums)):
        complement = target - nums[i]
        if complement in my_dict:
            return [my_dict[complement], i]
        my_dict[nums[i]] = i

#########################################################################################

# Problem Statement 29
# You are given an integer array nums. Write a function that returns True 
# if any value appears at least twice in the array and False if every element is distinct.
#
# Your task is to implement the solution using a hashmap (Python dictionary).
#
# Input:
# - nums (List[int]): A list of integers.
#
# Output:
# - bool: True if any value appears at least twice in the array, otherwise False.
#
# Examples:
# Input:  [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
# Output: True
#
# Input:  [1, 2, 3, 4]
# Output: False
#
# Input:  [1, 2, 3, 1]
# Output: True

def contains_duplicate(nums):
    num_dict = {}
    for num in nums:
        if num in num_dict:
            return True
        num_dict[num] = True
    return False

#########################################################################################

# Problem Statement 30
# Given a list of integers, write a function to find the maximum element in the list.
#
# Constraints:
# - The list will contain at least one element.
# - Function signature: def find_max_element(lst):
#
# Input:
# - lst (List[int]): A list of integers.
#
# Output:
# - int: The maximum element in the list.
#
# Examples:
# Input:  [3, 5, 2, 9, 6]
# Output: 9
#
# Input:  [-1, -2, -3, -4]
# Output: -1
#
# Input:  [7]
# Output: 7

def find_max_element(lst):
    highest = lst[0]
    for num in lst[1:]:
        if num > highest:
            highest = num
    return highest

#########################################################################################

# Problem Statement 31
# Given a list of integers, write a function to find the sum of all the elements in the list.
#
# Constraints:
# - The list will contain at least one element.
# - Function signature: def sum_of_elements(lst):
#
# Input:
# - lst (List[int]): A list of integers.
#
# Output:
# - int: The sum of all the elements in the list.
#
# Examples:
# Input:  [7]
# Output: 7
#
# Input:  [-1, -2, -3, -4]
# Output: -10
#
# Input:  [1, 2, 3, 4, 5]
# Output: 15

def sum_of_elements(lst):
    s = 0
    for num in lst:
        s += num
    return s

#########################################################################################

# Problem Statement 32
# Given a list of integers, determine if it is a palindrome.
# A list is considered a palindrome if it reads the same forward and backward.
#
# Constraints:
# - The list will contain at least one element.
# - Function signature: def is_palindrome(lst):
#
# Input:
# - lst (List[int]): A list of integers.
#
# Output:
# - bool: Return True if the list is a palindrome, otherwise False.
#
# Examples:
# Input:  [7, 8, 9, 8, 7]
# Output: True
#
# Input:  [1, 2, 3, 4, 5]
# Output: False
#
# Input:  [1, 2, 3, 2, 1]
# Output: True

def is_palindrome(lst):
    m = len(lst) // 2
    i = 0
    j = len(lst) - 1
    while i < m:
        if lst[i] != lst[j]:
            return False
        else:
            i += 1
            j -= 1
    return True

#########################################################################################

# Problem Statement 33
# Given a list of integers, write a function to reverse the order of elements in the list.
#
# Constraints:
# - The list will contain at least one element.
# - Function signature: def reverse_list(lst):
#
# Input:
# - lst (List[int]): A list of integers.
#
# Output:
# - List[int]: The list with elements in reversed order.
#
# Examples:
# Input:  [1, 2, 3, 4, 5]
# Output: [5, 4, 3, 2, 1]
#
# Input:  [10, 20, 30]
# Output: [30, 20, 10]
#
# Input:  [7, 8, 9]
# Output: [9, 8, 7]

def reverse_list(lst):
    rev_lst = []
    for i in range(len(lst) - 1, -1, -1):
        rev_lst.append(lst[i])
    return rev_lst

#########################################################################################

# Problem Statement 34
# Given a list of integers and an integer D, write a function to rotate the list
# to the left by D positions.
#
# Constraints:
# - The list will contain at least one element.
# - 0 <= D <= 10^6 (use modulo operation to handle D > len(ARR))
# - Function signature: def rotate_left(ARR, D):
#
# Input:
# - ARR (List[int]): A list of integers.
# - D (int): The number of positions to rotate the list to the left.
#
# Output:
# - List[int]: The list after rotating it to the left by D positions.
#
# Examples:
# Input:  ARR = [1, 2, 3, 4, 5], D = 2
# Output: [3, 4, 5, 1, 2]
#
# Input:  ARR = [10, 20, 30, 40, 50], D = 3
# Output: [40, 50, 10, 20, 30]
#
# Input:  ARR = [7, 8, 9, 10], D = 1
# Output: [8, 9, 10, 7]

def rotate_left(ARR, D):
    D = D % len(ARR)
    ARR[:] = ARR[D:] + ARR[:D]
    return ARR

#########################################################################################

# Problem Statement 35
# You are given a large integer represented as an integer array digits,
# where each digits[i] is the i-th digit of the integer. The digits are ordered
# from most significant to least significant in left-to-right order.
# The large integer does not contain any leading zeroes.
#
# Write a function to increment the large integer by one and return the resulting array of digits.
#
# Constraints:
# - The list will contain at least one element.
# - Each element in the list is between 0 and 9 (inclusive).
# - Function signature: def plus_one(digits):
#
# Input:
# - digits (List[int]): A list of integers where each integer represents a digit of a large number.
#
# Output:
# - List[int]: The list representing the number after incrementing it by one.
#
# Examples:
# Input:  [1, 2, 3]
# Output: [1, 2, 4]
#
# Input:  [4, 3, 2, 1]
# Output: [4, 3, 2, 2]
#
# Input:  [9, 9, 9]
# Output: [1, 0, 0, 0]

def plus_one(digits):
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] == 9:
            digits[i] = 0
        else:
            digits[i] += 1
            return digits
    return [1] + digits

#########################################################################################

# Problem Statement 36
# Given an array nums containing n distinct numbers in the range [0, n],
# return the only number in the range that is missing from the array.
#
# Constraints:
# - The array contains exactly n distinct numbers from the range [0, n].
# - The size of the array is at least 1.
# - Function signature: def find_missing_number(nums):
#
# Input:
# - nums (List[int]): A list of integers where each integer is unique and in the range [0, n].
#
# Output:
# - int: The missing number in the range [0, n].
#
# Examples:
# Input:  [3, 0, 1]
# Output: 2
#
# Input:  [0, 1]
# Output: 2
#
# Input:  [8, 7, 6, 4, 3, 2, 0, 1]
# Output: 5

def find_missing_number(nums):
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum

#########################################################################################

# Problem Statement 37
# Write a function that checks whether the given array is sorted in non-decreasing order.
# The array is considered sorted if every element is less than or equal to the next element.
#
# Constraints:
# - The list will contain at least one element.
# - Function signature: def is_sorted(arr):
#
# Input:
# - arr (List[int]): A list of integers.
#
# Output:
# - bool: True if the array is sorted in non-decreasing order, False otherwise.
#
# Examples:
# Input:  [5, 4, 3, 2, 1]
# Output: False
#
# Input:  [1, 3, 2, 4, 5]
# Output: False
#
# Input:  [1, 2, 3, 4, 5]
# Output: True

def is_sorted(arr):
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            return False
    return True

#########################################################################################

# Problem Statement 38
# Given an integer array nums, write a function to move all 0s to the end of the array
# while maintaining the relative order of the non-zero elements.
#
# Constraints:
# - The list will contain at least one element.
# - Function signature: def move_zeroes(nums):
#
# Input:
# - nums (List[int]): A list of integers.
#
# Output:
# - List[int]: The list nums with all 0s moved to the end, preserving the order of non-zero elements.
#
# Examples:
# Input:  [0, 1, 0, 3, 12]
# Output: [1, 3, 12, 0, 0]
#
# Input:  [0, 0, 1]
# Output: [1, 0, 0]
#
# Input:  [4, 2, 4, 0, 0, 3, 0, 5, 1, 0]
# Output: [4, 2, 4, 3, 5, 1, 0, 0, 0, 0]

def move_zeroes(nums):
    insert_pos = 0
    for num in nums:
        if num != 0:
            nums[insert_pos] = num
            insert_pos += 1
    while insert_pos < len(nums):
        nums[insert_pos] = 0
        insert_pos += 1
    return nums

#########################################################################################

# Problem Statement 39
# Given two integer arrays nums1 and nums2, return an array of their intersection.
# Each element in the result must be unique, and you may return the result in any order.
#
# Constraints:
# - 1 <= len(nums1), len(nums2) <= 1000
# - Each integer in nums1 and nums2 is in the range [-10^9, 10^9].
# - Function signature: def intersection(nums1, nums2):
#
# Input:
# - nums1 (List[int]): An array of integers.
# - nums2 (List[int]): An array of integers.
#
# Output:
# - List[int]: An array of unique integers that are present in both nums1 and nums2.
#
# Examples:
# Input:  nums1 = [1, 2, 3], nums2 = [4, 5, 6]
# Output: []
#
# Input:  nums1 = [1, 2, 2, 1], nums2 = [2, 2]
# Output: [2]
#
# Input:  nums1 = [4, 9, 5], nums2 = [9, 4, 9, 8, 4]
# Output: [9, 4]

def intersection(nums1, nums2):
    set1 = set(nums1)
    set2 = set(nums2)
    intersection = set()
    for num in set2:
        if num in set1:
            intersection.add(num)
    return list(intersection)

#########################################################################################

# Problem Statement 40
# Given a binary array nums, return the maximum number of consecutive 1s in the array.
#
# Input:
# - nums (List[int]): A binary array where each element is either 0 or 1.
#
# Output:
# - int: The maximum number of consecutive 1s in the array.
#
# Examples:
# Input:  [0, 0, 0, 0]
# Output: 0
#
# Input:  [1, 0, 1, 1, 0, 1, 1, 1, 1]
# Output: 4
#
# Input:  [1, 1, 0, 1, 1, 1]
# Output: 3

def find_max_consecutive_ones(nums):
    max_ones = 0
    ones = 0
    for num in nums:
        if num == 1:
            ones += 1
        else:
            ones = 0
        max_ones = max(max_ones, ones)
    return max_ones

#########################################################################################
