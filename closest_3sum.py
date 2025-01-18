nums = [-1, 2, 1, -4]
target = 1

# Step 1: Sort the array
nums.sort()  # Sorted nums = [-4, -1, 1, 2]
closest_sum = float('inf')  # Initialize with a very large value

# Step 2: Iterate through the array
for i in range(len(nums) - 2):  # We need at least 3 numbers
    left, right = i + 1, len(nums) - 1  # Two pointers

    while left < right:
        # Calculate the sum of the current triplet
        current_sum = nums[i] + nums[left] + nums[right]

        # Update the closest sum if it's closer to the target
        if abs(target - current_sum) < abs(target - closest_sum):
            closest_sum = current_sum

        # Adjust pointers based on the sum
        if current_sum < target:
            left += 1  # Increase the sum by moving left pointer
        elif current_sum > target:
            right -= 1  # Decrease the sum by moving right pointer
        else:
            # If the exact target is found, return immediately
            print("Exact target found:", current_sum)
            break

print("Closest sum:", closest_sum)
'''
Explanation:
Sort the Array:

Sorting helps use the two-pointer technique efficiently.
Iterate Through Elements:

The outer loop picks the first element of the triplet.
The two pointers (left and right) find the other two elements.
Calculate the Sum:

Compute the sum of the triplet (nums[i] + nums[left] + nums[right]).
Update Closest Sum:

Check if the current sum is closer to the target than the previously stored closest_sum.
Adjust Pointers:

Move left forward if the sum is too small.
Move right backward if the sum is too large.
Break Early:

If the exact target is found, return immediately.
Example Walkthrough:
Input:
python
Copy code
nums = [-1, 2, 1, -4]
target = 1
Steps:
Sorted nums = [-4, -1, 1, 2].

Outer loop starts with i = 0 (nums[i] = -4).

left = 1 (nums[left] = -1), right = 3 (nums[right] = 2).
current_sum = -4 + -1 + 2 = -3. Update closest_sum = -3.
Move left to 2 (nums[left] = 1).
current_sum = -4 + 1 + 2 = -1. Update closest_sum = -1.
Move left to 3, exit inner loop.
Outer loop continues with i = 1 (nums[i] = -1).

left = 2 (nums[left] = 1), right = 3 (nums[right] = 2).
current_sum = -1 + 1 + 2 = 2. Update closest_sum = 2.
Move right to 2, exit inner loop.
Output:
bash
Copy code
Closest sum: 2
Key Takeaways:
1 Always sort the array for two-pointer techniques.
2 Use the absolute difference abs(target - current_sum) to find the closest sum.
3 Consider edge cases like small arrays or large target values.


'''