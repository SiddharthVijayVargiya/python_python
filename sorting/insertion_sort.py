def insertion_sort(arr):
    # Traverse through the array starting from the second element (index 1)
    for i in range(1, len(arr)):
        key = arr[i]  # The element to be inserted into the sorted portion of the array
        j = i - 1  # j will be used to compare the key with elements to its left

        # Move elements of arr[0..i-1] that are greater than the key
        # to one position ahead of their current position to make space for the key
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]  # Shift the element arr[j] one position to the right
            j -= 1  # Move to the previous element in the sorted portion

        # Place the key at the correct position in the sorted portion
        # This is the position where arr[j] <= key, so arr[j + 1] will be the right spot for the key
        arr[j + 1] = key


# Example usage:
arr = [8, 4, 3, 7, 6]
insertion_sort(arr)
print("Sorted array:", arr)
'''
How Insertion Sort Works:
Let's break down the code for a sample array [8, 4, 3, 7, 6]:

Start from the second element 4:

Compare it with the first element 8.
Since 4 < 8, shift 8 to the right and insert 4 before it.
Array becomes [4, 8, 3, 7, 6].
Move to the third element 3:

Compare it with 8 (shift 8), then compare it with 4 (shift 4).
Insert 3 at the beginning.
Array becomes [3, 4, 8, 7, 6].
Move to the fourth element 7:

Compare it with 8 (shift 8).
Insert 7 between 4 and 8.
Array becomes [3, 4, 7, 8, 6].
Move to the fifth element 6:

Compare it with 8 (shift 8), then compare it with 7 (shift 7).
Insert 6 between 4 and 7.
Array becomes [3, 4, 6, 7, 8].



'''