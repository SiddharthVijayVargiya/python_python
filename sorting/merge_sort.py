'''
Here's how you can implement Merge Sort in Python:

Merge Sort Algorithm
Divide:
Recursively split the array into two halves until each subarray contains a single element or is empty.


Conquer:
Merge the sorted subarrays into a single sorted array.


Combine:
During merging, ensure that elements from both subarrays are combined in sorted order.

'''
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    # Step 1: Split the array into two halves
    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    # Step 2: Merge the two sorted halves
    return merge(left_half, right_half)

def merge(left, right):
    sorted_array = []
    i = j = 0

    # Step 3: Compare elements from both halves and merge
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            sorted_array.append(left[i])
            i += 1
        else:
            sorted_array.append(right[j])
            j += 1

    # Step 4: Add remaining elements from left and right
    sorted_array.extend(left[i:])
    sorted_array.extend(right[j:])

    return sorted_array

# Example Usage
if __name__ == "__main__":
    nums = [38, 27, 43, 3, 9, 82, 10]
    print("Original Array:", nums)
    sorted_nums = merge_sort(nums)
    print("Sorted Array:", sorted_nums)
'''

Explanation
Base Case:
If the array has one or zero elements, it’s already sorted.
Recursive Division:
The array is recursively divided into two halves using slicing (arr[:mid] and arr[mid:]).
Merge Function:
Uses two pointers (i and j) to compare elements from both halves and creates a sorted array.
Appends remaining elements from left or right if one of them is exhausted.

'''


'''
Example Walkthrough
Input:
plaintext
Copy code
[38, 27, 43, 3, 9, 82, 10]
Recursive Splits:
plaintext
Copy code
[38, 27, 43, 3, 9, 82, 10]
[38, 27, 43]  [3, 9, 82, 10]
[38] [27, 43]  [3, 9] [82, 10]
[27] [43]     [82] [10]
Merging:
plaintext
Copy code
[27, 43]    [10, 82]
[27, 38, 43]    [3, 9, 10, 82]
[3, 9, 10, 27, 38, 43, 82]
Output:
plaintext
Copy code
[3, 9, 10, 27, 38, 43, 82]


'''