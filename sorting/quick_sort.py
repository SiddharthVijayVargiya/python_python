
def quick_sort(arr):
    # Base case: If the array has 1 or no elements, it's already sorted
    if len(arr) <= 1:
        return arr
    
    # Choose the pivot (we can use the last element as the pivot)
    pivot = arr[-1]
    
    # Partitioning step: create two sub-arrays
    smaller = [x for x in arr[:-1] if x <= pivot]
    larger = [x for x in arr[:-1] if x > pivot]
    
    # Recursively sort the sub-arrays and combine the result with the pivot in the middle
    return quick_sort(smaller) + [pivot] + quick_sort(larger)

# Example usage:
arr = [10, 7, 8, 9, 1, 5]
sorted_arr = quick_sort(arr)
print("Sorted Array:", sorted_arr)

'''
Quick Sort Visualization Notes

Quick Sort is a divide-and-conquer sorting algorithm. It works by selecting a pivot element from the array 
and partitioning the other elements into two sub-arrays according to whether they are 
smaller or greater than the pivot. 
The sub-arrays are then sorted recursively.

### Step 1: Initial Array
--------------------------
We start with the input array: [10, 7, 8, 9, 1, 5].

### Step 2: Select a Pivot
--------------------------
In Quick Sort, we select a pivot element. For simplicity, we often choose the last element as the pivot.

In our example, we choose 5 as the pivot.

### Step 3: Partitioning
--------------------------
The partitioning step involves rearranging the array such that all elements smaller than the pivot come before it, 
and all elements greater than the pivot come after it. We move the pivot element to its correct position.

Array before partitioning:

            [10, 7, 8, 9, 1, 5]

We compare each element with the pivot (5) and rearrange them:

1. Compare 10 with 5. Since 10 > 5, leave it.
2. Compare 7 with 5. Since 7 > 5, leave it.
3. Compare 8 with 5. Since 8 > 5, leave it.
4. Compare 9 with 5. Since 9 > 5, leave it.
5. Compare 1 with 5. Since 1 < 5, swap 1 and 10. The array becomes: [1, 7, 8, 9, 10, 5]
6. Now the pivot (5) is swapped with 7, so the array becomes: [1, 5, 8, 9, 10, 7]

After partitioning, the pivot 5 is at its correct position in the sorted array.

Array after partitioning:

            [1, 5, 8, 9, 10, 7]

### Step 4: Recursion
--------------------------
Now we recursively apply the same process to the two sub-arrays:

1. Left sub-array: [1] (No sorting needed since it has only one element).
2. Right sub-array: [8, 9, 10, 7].

### Step 5: Recursively Sorting the Right Sub-array
-------------------------------------------------
For the right sub-array [8, 9, 10, 7], we again select the pivot. In this case, 7 is the pivot.

Array before partitioning:

            [8, 9, 10, 7]

Partitioning steps:
1. Compare 8 with 7. Since 8 > 7, leave it.
2. Compare 9 with 7. Since 9 > 7, leave it.
3. Compare 10 with 7. Since 10 > 7, leave it.
4. Swap 7 with 8, so the array becomes: [7, 9, 10, 8]

Array after partitioning:

            [7, 9, 10, 8]

Now, pivot 7 is at its correct position.

### Step 6: Recursion on Sub-arrays Again
-------------------------------------------------
We apply the same process recursively to the two sub-arrays:

1. Left sub-array: [] (No sorting needed).
2. Right sub-array: [9, 10, 8].

For the right sub-array [9, 10, 8], we choose 8 as the pivot.

Array before partitioning:

            [9, 10, 8]

Partitioning steps:
1. Compare 9 with 8. Since 9 > 8, leave it.
2. Compare 10 with 8. Since 10 > 8, leave it.
3. Swap 8 with 9, so the array becomes: [8, 10, 9]

Array after partitioning:

            [8, 10, 9]

Now, pivot 8 is at its correct position.

### Step 7: Final Recursion
--------------------------------
Now, we recursively apply the same process to the two sub-arrays:

1. Left sub-array: [] (No sorting needed).
2. Right sub-array: [10, 9].

For the sub-array [10, 9], we choose 9 as the pivot.

Array before partitioning:

            [10, 9]

Partitioning steps:
1. Compare 10 with 9. Since 10 > 9, leave it.
2. Swap 9 with 10.

Array after partitioning:

            [9, 10]

Pivot 9 is now at its correct position.

### Final Sorted Array
----------------------
At this point, all the sub-arrays have been sorted and merged into a final sorted array:

            [1, 5, 7, 8, 9, 10]

Thus, the array is sorted in ascending order.

### Time Complexity:
-------------------
- Best case: O(n log n) - when the pivot divides the array into two nearly equal parts.
- Average case: O(n log n) - in most cases.
- Worst case: O(n^2) - when the pivot is always the smallest or largest element (this can be mitigated with a good pivot selection strategy like random pivoting or using the median of three).

Quick Sort is an efficient, divide-and-conquer sorting algorithm, but its performance can degrade in the worst case if poor pivot choices are made.

'''
