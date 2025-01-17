def bubble_sort(arr):
    n = len(arr)
    # Traverse through all elements in the array
    for i in range(n):
        # Track if any swap was made
        swapped = False
        
        # Last i elements are already sorted, no need to compare them again
        for j in range(0, n - i - 1):
            # Compare the current element with the next
            if arr[j] > arr[j + 1]:
                # Swap if the element is greater than the next element
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # If no elements were swapped, the list is already sorted
        if not swapped:
            break

# Example usage:
arr = [64, 34, 25, 12, 22, 11, 90]
print("Original array:", arr)
bubble_sort(arr)
print("Sorted array:", arr)
'''
Why Use n - i - 1 Instead of n?
In bubble sort, each pass ensures that the largest remaining element moves to the end.
So, after the first pass, the last element is in its correct place. After the second pass,
the second-last element is in its correct place, and so on.

This means that after each pass, we don't need to compare as many elements,
because the end of the list is already sorted.

Example:
Consider an array [4, 2, 3, 1].

First pass (i = 0):

Compare and swap elements until the end (j = 0 to 2):
Compare 4 and 2, swap them → [2, 4, 3, 1]
Compare 4 and 3, swap them → [2, 3, 4, 1]
Compare 4 and 1, swap them → [2, 3, 1, 4]
Now, the largest element (4) is at the end, so it's in the correct place.
Second pass (i = 1):

No need to compare the last element (4), so we only compare up to j = 1:
Compare 2 and 3, no swap → [2, 3, 1, 4]
Compare 3 and 1, swap them → [2, 1, 3, 4]
Now, the second-largest element (3) is in its correct place,
so the last two elements are sorted.
Third pass (i = 2):

Now, only compare up to j = 0:
Compare 2 and 1, swap them → [1, 2, 3, 4]
Now, the third-largest element (2) is in place, so the list is fully sorted.
Notice that each time, 
we don't need to check the already sorted elements at the end.
This is why we reduce the range of j in each pass by using n - i - 1.

What Happens if You Use for j in range(n)?
If you use for j in range(n) for each pass,
you will continue comparing elements even when they are already sorted at the end.
This means extra, unnecessary comparisons.

In the first pass (i = 0), you will compare all elements, which is fine.
In the second pass (i = 1), you will still compare the last element,
even though it's already in the correct place.
In the third pass (i = 2), you'll keep comparing the last two elements,
even though they are already sorted.
This makes the algorithm slower because you're doing more work than needed.

Visualizing the Difference:
Using for j in range(0, n - i - 1),
you're reducing the number of comparisons as more elements get sorted.
Using for j in range(n) would involve unnecessary comparisons and slow things down.
Summary:
n - i - 1 ensures you don’t compare elements that are already sorted at the end of the array.
n would compare every element, including the sorted ones, wasting time.
Does this explanation help clarify things better?

'''