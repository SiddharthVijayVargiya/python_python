def heapify(arr, n, i):
    """
    Function to maintain the max heap property.

    :param arr: List of elements
    :param n: Size of the heap
    :param i: Index of the current node
    """
    largest = i  # Assume the root is the largest
    left = 2 * i + 1  # Left child
    right = 2 * i + 2  # Right child

    # If left child is larger than the root
    if left < n and arr[left] > arr[largest]:
        largest = left

    # If right child is larger than the largest so far
    if right < n and arr[right] > arr[largest]:
        largest = right

    # If the largest is not the root
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Swap
        heapify(arr, n, largest)  # Recursively heapify the affected subtree


def heap_sort(arr):
    """
    Function to perform heap sort.

    :param arr: List of elements to be sorted
    """
    n = len(arr)

    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements from the heap one by one
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # Swap the current root with the end
        heapify(arr, i, 0)  # Heapify the reduced heap


# Example usage
if __name__ == "__main__":
    data = [12, 11, 13, 5, 6, 7]
    print("Unsorted array:", data)
    heap_sort(data)
    print("Sorted array:", data)
'''
Explanation:
Heapify:

Ensures the binary tree satisfies the max heap property (every parent node is greater than or equal to its child nodes).
Building the Max Heap:

Start from the last non-leaf node and heapify all nodes up to the root.
Sorting:

Repeatedly swap the root (maximum value in the heap) with the last element of the heap.
Reduce the size of the heap and heapify the root again.

'''


#Unsorted array: [12, 11, 13, 5, 6, 7]
#Sorted array: [5, 6, 7, 11, 12, 13]



'''
Heap Sort Visualization Notes

Heap Sort is a comparison-based sorting algorithm that utilizes a binary heap data structure. 
It works by first building a max heap from the input data and then repeatedly extracting the maximum element 
from the heap and placing it at the end of the sorted array.

### Step 1: Initial Array
--------------------------
We begin with the input array: [12, 11, 13, 5, 6, 7].

The goal is to build a max heap. In a max heap, the value of each parent node is greater than or equal to its children.

### Step 2: Build Max Heap
--------------------------
We start by treating the array as a complete binary tree:

            12
          /    \
        11      13
       /  \    /
      5    6  7

- The heapification process begins from the last non-leaf node and moves upwards. The last non-leaf node is the node at index `n//2 - 1` (index 2).

1. **Heapify node at index 2** (value = 13):  
   - 13 is already greater than its children (no swap needed).
   
2. **Heapify node at index 1** (value = 11):  
   - The left child (5) and right child (6) are smaller than 11, so no swap is required.

3. **Heapify node at index 0** (value = 12):  
   - The left child (11) and right child (13) are compared.
   - Since 13 > 12, we swap 12 and 13.

After the swap:

            13
          /    \
        11      12
       /  \    /
      5    6  7

Now we have a valid max heap.

### Step 3: Extract Elements from Heap
--------------------------
After building the max heap, we repeatedly extract the root element (the maximum element), swap it with the last element in the array, and then heapify the reduced heap.

#### **First Extraction:**
1. Swap the root (13) with the last element (7):

            7
          /    \
        11      12
       /  \    
      5    6  

   Array: [7, 11, 12, 5, 6, 13]

2. Heapify the root (node 0) to restore the max heap:
   - The left child (11) and right child (12) are compared.
   - Since 12 > 7, swap 7 and 12.
   
   After the swap:

            12
          /    \
        11      7
       /  \    
      5    6  

   Array: [12, 11, 7, 5, 6, 13]

3. Now, heapify the subtree at index 2 (node 7). No swaps are needed,
as 7 is already greater than its children.

At the end of the first extraction, the array is: [12, 11, 7, 5, 6, 13],
and the largest element (13) is correctly placed at the end of the array.

#### **Second Extraction:**
1. Swap the root (12) with the last unsorted element (6):

            6
          /    \
        11      7
       /  \    
      5    

   Array: [6, 11, 7, 5, 12, 13]

2. Heapify the root (node 0) to restore the max heap:
   - The left child (11) and right child (7) are compared.
   - Since 11 > 6, swap 6 and 11.
   
   After the swap:

            11
          /    \
        6       7
       /    
      5    

   Array: [11, 6, 7, 5, 12, 13]

3. Now, heapify the subtree at index 1 (node 6). No swap needed since 6 is greater than its child (5).

At the end of the second extraction, the array is: [11, 6, 7, 5, 12, 13], and the second largest element (12) is correctly placed at the second-to-last position.

#### **Third Extraction:**
1. Swap the root (11) with the last unsorted element (5):

            5
          /    \
        6       7
    
   Array: [5, 6, 7, 11, 12, 13]

2. Heapify the root (node 0) to restore the max heap:
   - The left child (6) and right child (7) are compared.
   - Since 7 > 5, swap 5 and 7.

   After the swap:

            7
          /    \
        6       5

   Array: [7, 6, 5, 11, 12, 13]

3. Now, heapify the subtree at index 2 (node 5). No swaps are needed.

At the end of the third extraction, the array is: [7, 6, 5, 11, 12, 13], and the third largest element (11) is placed correctly.

#### **Continue the Process:**
The process of extraction continues, with each root element being swapped with the last unsorted element and the heap being re-heapified until the array is fully sorted.

After all the extractions, the array becomes:

Final Sorted Array: [5, 6, 7, 11, 12, 13]

### Summary of Heap Sort Steps:
1. **Build the Max Heap**: Convert the array into a binary heap where the parent nodes are larger than their children.
2. **Sort**: Repeatedly swap the root with the last element and heapify the reduced heap to ensure it remains a max heap.
3. **Final Sorted Array**: After all extractions, the array is sorted in ascending order.

Heap Sort has a time complexity of O(n log n), making it efficient for large datasets. However, it is not a stable sort, meaning that equal elements may not retain their original relative order.

'''
