'''
Problem Statement: Longest Common Prefix
Given an array of strings arr, write a function to find the longest common prefix among all the
strings. If there is no common prefix, return an empty string "".

Input:
arr: A list of strings, with 1 <= len(arr) <= 200 and each string having a length of 1 <= len(string) <= 200.
Output:
Return the longest common prefix among all the strings in the array. If no common prefix exists, return "".

'''


#Input: arr = ["flower", "flow", "flight"]
#Output: "fl"
#Explanation: The longest common prefix among all strings is "fl".




'''
Input: arr = ["dog", "racecar", "car"]
Output: ""
Explanation: There is no common prefix among the input strings.

'''


'''
Constraints:
All strings in the input array consist of lowercase English letters.
Notes:
The common prefix is the initial part of each string that is shared by all strings in the array.
You must compare each string and iteratively reduce the prefix until the longest possible common prefix is found.

'''
def longest_common_prefix(arr):
    # Step 1: Handle the edge case where the input array is empty.
    # If the input list of strings is empty, return an empty string immediately
    if not arr:
        return ""  # No strings to compare, so no common prefix is possible.
    
    # Step 2: Initialize the prefix as the first string in the array.
    # The assumption is that the first string could potentially be the common prefix,
    # and we will try to reduce it by comparing with other strings.
    prefix = arr[0]
    
    # Step 3: Iterate through each of the remaining strings in the array.
    # We start from the second string, because we already initialized the prefix
    # with the first string.
    for string in arr[1:]:
        # Step 4: Compare the current string with the prefix.
        # The loop checks if the current string starts with the current prefix.
        # If it doesn't, we reduce the prefix by removing its last character.
        # This process continues until the prefix becomes a valid common prefix
        # for the current string, or until it becomes empty.
        while string[:len(prefix)] != prefix and prefix:
            # Step 5: Gradually reduce the prefix by slicing off the last character.
            # Each time the prefix doesn't match the start of the string, 
            # we shorten the prefix from the right side by one character.
            prefix = prefix[:-1]
        
        # Step 6: If at any point the prefix becomes empty, we can break the loop early.
        # This means there is no common prefix at all between the strings.
        if not prefix:
            break  # No need to check further; no common prefix exists.

    # Step 7: After comparing with all strings, return the resulting prefix.
    # If a common prefix was found, it will be returned; otherwise, an empty
    # string will be returned if no common prefix exists across all strings.
    return prefix

# Example test case:
arr = ['a','abccd', 'abccd','ab']  # All strings are identical, so the common prefix is the entire string.
print(longest_common_prefix(arr))  # Output: "abccd"











'''Approach 1: Vertical Scanning
Start by picking the first string as the prefix candidate.
Iterate through each character of the first string.
Compare that character with the corresponding character in the other strings.
Stop when:
Characters don't match.
Reaching the end of any string.
Return the accumulated prefix.
Python Implementation:
python
Copy code'''
def longest_common_prefix(strs):
    if not strs:
        return ""
    
    # Take the first string as the reference
    prefix = strs[0]
    
    for i in range(len(prefix)):
        # Check each character with all other strings
        for string in strs[1:]:
            if i == len(string) or string[i] != prefix[i]:
                return prefix[:i]
    
    return prefix





'''Approach 2: Horizontal Scanning
Start with the first string as the initial prefix.
Compare the prefix with each subsequent string.
Reduce the prefix length until it matches the start of the current string.
Stop when the prefix becomes empty.
Python Implementation:
python
Copy code'''
def longest_common_prefix(strs):
    if not strs:
        return ""
    
    prefix = strs[0]
    
    for string in strs[1:]:
        while not string.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix





'''Approach 3: Divide and Conquer
The idea is to split the array into smaller subarrays, find the longest common prefix for each subarray, and then merge the results.

Steps:
Divide: Split the array into two halves recursively until you have individual strings or small arrays.
Conquer: Compute the longest common prefix for the left and right halves.
Combine: Merge the results by comparing the prefixes from the two halves character by character.
Implementation:
python
Copy code'''
def common_prefix(str1, str2):
    # Helper function to find the common prefix between two strings
    min_len = min(len(str1), len(str2))
    for i in range(min_len):
        if str1[i] != str2[i]:
            return str1[:i]
    return str1[:min_len]

def longest_common_prefix_divide_and_conquer(strs, left, right):
    # Base case: If there's only one string, return it
    if left == right:
        return strs[left]
    
    # Divide: Split into two halves
    mid = (left + right) // 2
    lcp_left = longest_common_prefix_divide_and_conquer(strs, left, mid)
    lcp_right = longest_common_prefix_divide_and_conquer(strs, mid + 1, right)
    
    # Conquer: Merge the results
    return common_prefix(lcp_left, lcp_right)

def longest_common_prefix(strs):
    if not strs:
        return ""
    return longest_common_prefix_divide_and_conquer(strs, 0, len(strs) - 1)





'''Approach 4: Binary Search
The idea is to use the minimum string length as the search space and perform a binary search to check the validity of prefixes of varying lengths.

Steps:
Find the shortest string's length in the array (this limits the maximum possible prefix length).
Perform binary search on the range [0, shortest length].
For each midpoint length, check if all strings share a prefix of that length.
If yes, try a longer prefix (move right).
If no, try a shorter prefix (move left).
Implementation:
python
Copy code'''
def is_common_prefix(strs, length):
    # Helper function to check if all strings have the same prefix of given length
    prefix = strs[0][:length]
    for string in strs[1:]:
        if not string.startswith(prefix):
            return False
    return True

def longest_common_prefix(strs):
    if not strs:
        return ""
    
    # Find the shortest string's length
    min_length = min(len(s) for s in strs)
    low, high = 0, min_length
    
    while low <= high:
        mid = (low + high) // 2
        if is_common_prefix(strs, mid):
            low = mid + 1  # Try for a longer prefix
        else:
            high = mid - 1  # Try for a shorter prefix
    
    return strs[0][: (low + high) // 2]