def selection_sort(arr):
    n = len(arr)

    for i in range(n):
        # Assume the minimum is the first unsorted element
        min_idx = i

        # Find the minimum element in remaining unsorted part
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        # Swap the found minimum element with the first unsorted element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

# Example usage:
arr = [29, 10, 14, 37, 14]
selection_sort(arr)
print("Sorted array:", arr)
'''
Let’s sort the list [29, 10, 14, 37, 14] using selection sort (in ascending order):

Step 1: Find the smallest element in the list [29, 10, 14, 37, 14]:

The smallest element is 10.
Swap 10 with the first element 29.
List after first pass: [10, 29, 14, 37, 14].
Step 2: Find the smallest element in the remaining unsorted part [29, 14, 37, 14]:

The smallest element is 14.
Swap 14 with 29.
List after second pass: [10, 14, 29, 37, 14].
Step 3: Find the smallest element in the remaining unsorted part [29, 37, 14]:

The smallest element is 14.
Swap 14 with 29.
List after third pass: [10, 14, 14, 37, 29].
Step 4: Find the smallest element in the remaining unsorted part [37, 29]:

The smallest element is 29.
Swap 29 with 37.
List after fourth pass: [10, 14, 14, 29, 37].
Now, the list is sorted.

'''