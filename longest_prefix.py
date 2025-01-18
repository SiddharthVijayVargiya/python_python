'''Approach 1: Vertical Scanning
Start by picking the first string as the prefix candidate.
Iterate through each character of the first string.
Compare that character with the corresponding character in the other strings.
Stop when:
Characters don't match.
Reaching the end of any string.
Return the accumulated prefix.
Python Implementation:'''

def longest_prefix(lst):
    if not lst:
        return ""
    
    prefix = lst[0]  # Start with the first word as the prefix
    
    for i in range(len(prefix)):
        for word in lst:
            if i >= len(word) or prefix[i] != word[i]:
                return prefix[:i]  # Return the prefix up to the current index
    return prefix  # If we don't find any mismatches, return the whole prefix

# Example List
lst = ['apple', 'app', 'apple', 'apricot', 'applle']
print(longest_prefix(lst))





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


'''
Approach 3: Divide and Conquer
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