def bubble_sort(array1):
    for i in range(len(array1) - 1, 0, -1):
        for j in range(i):
            if array1[j] > array1[j + 1]:
                temp = array1[j]
                array1[j] = array1[j + 1]
                array1[j + 1] = temp
    return array1

def selection_sort(array1):
    for i in range(len(array1) - 1):
        min_index = i
        for j in range(i + 1, len(array1)):
            if array1[j] < array1[min_index]:
                min_index = j
        if i != min_index:
            temp = array1[i]
            array1[i] = array1[min_index]
            array1[min_index] = temp
    return array1

def insertion_sort(array1):
    for i in range(1, len(array1)):
        temp = array1[i]
        j = i - 1
        while temp < array1[j] and j >= 0:
            array1[j + 1] = array1[j]
            array1[j] = temp
            j -= 1
    return array1