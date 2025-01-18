def threeSum(nums):
    nums.sort()  # Step 1: Sort the array
    result = []
    n = len(nums)

    for i in range(n - 2):  # Step 2: Iterate through the array
        # Skip duplicates for the first element of the triplet
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        target = -nums[i]
        left, right = i + 1, n - 1

        while left < right:  # Step 3: Two-pointer approach
            current_sum = nums[left] + nums[right]
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])

                # Skip duplicates for the second and third elements
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1
            elif current_sum < target:
                left += 1  # Need a larger sum
            else:
                right -= 1  # Need a smaller sum

    return result
'''
Example:
Input:
python
Copy code
nums = [-1, 0, 1, 2, -1, -4]
Output:
python
Copy code
[[-1, -1, 2], [-1, 0, 1]]
Explanation:
After sorting: nums = [-4, -1, -1, 0, 1, 2].
Triplets that sum to 0 are:
[-1, -1, 2]
[-1, 0, 1]
Edge Cases:
Empty Array: Input is nums = []. Output is [].
No Triplets: Input is nums = [1, 2, 3]. Output is [].
Duplicates in Input: Ensure no duplicate triplets appear in the result.




'''
