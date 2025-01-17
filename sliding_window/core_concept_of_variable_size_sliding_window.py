#Core concept of variable sliding window in Python


# 1 You have a left pointer and a right pointer.

# 2 The right pointer expands the window by moving forward to include more elements.

# 3 The left pointer shrinks the window when a certain condition is violated.

# 4 Throughout, you keep track of the desired property (like max/min sum, substring with certain constraints, etc.).



'''

Here's a basic example where we find the length of the longest 
substring with at most two 
distinct characters using the variable sliding window approach:


'''

def longest_substring_with_k_distinct(s, k):
    # Dictionary to store the frequency of characters in the window
    char_count = {}
    
    left = 0
    max_len = 0

    # Right pointer expands the window
    for right in range(len(s)):
        char = s[right]
        char_count[char] = char_count.get(char, 0) + 1

        # Shrink the window if we have more than k distinct characters
        while len(char_count) > k:
            left_char = s[left]
            char_count[left_char] -= 1
            if char_count[left_char] == 0:
                del char_count[left_char]
            left += 1

        # Update the maximum length of the valid window
        max_len = max(max_len, right - left + 1)

    return max_len

# Example usage:
s = "eceba"
k = 2
print(longest_substring_with_k_distinct(s, k))  # Output: 3 (substring "ece")

'''

Explanation:
The window expands by moving right to include more characters.
Whenever the window has more than k distinct characters, 
the window is shrunk by moving left until it satisfies the condition again.
The goal is to find the maximum valid window size.
This approach ensures an efficient solution with a time complexity of O(n).



'''


'''

Step	left	right	                      Current Window	                char_count	                      Action	                          max_len
1	     0	      0	                             e	                             {'e': 1}	                   Expand window, valid condition	            1
2	     0	      1	                             ec	                             {'e': 1, 'c': 1}              Expand window, valid condition	            2
3	     0	      2	                             ece	                         {'e': 2, 'c': 1}              Expand window, valid condition	            3
4	     0	      3	                             eceb	                         {'e': 2, 'c': 1, 'b': 1}      Shrink window: Too many distinct characters	3
5	     1	      3	                             ceb	                         {'e': 1, 'c': 1, 'b': 1}	   Shrink window: Too many distinct characters	3
6	     2	      3	                             eb	                             {'e': 1, 'b': 1}	           Valid window achieved again	                3
7	     2	      4	                             eba	                         {'e': 1, 'b': 1, 'a': 1}	   Shrink window: Too many distinct characters	3
8	     3	      4                              ba	                             {'b': 1, 'a': 1}	           Valid window achieved again	                3


'''



import matplotlib.pyplot as plt

def visualize_sliding_window(s, k):
    char_count = {}
    left = 0
    max_len = 0
    steps = []

    for right in range(len(s)):
        char = s[right]
        char_count[char] = char_count.get(char, 0) + 1

        # Record current window state before shrinking
        steps.append((left, right, s[left:right + 1]))

        # Shrink the window if it violates the distinct character constraint
        while len(char_count) > k:
            left_char = s[left]
            char_count[left_char] -= 1
            if char_count[left_char] == 0:
                del char_count[left_char]
            left += 1

        # Record after adjustments
        max_len = max(max_len, right - left + 1)
        steps.append((left, right, s[left:right + 1]))

    # Visualizing the sliding window
    for i, (l, r, window) in enumerate(steps):
        print(f"Step {i+1}: Window = '{window}' | Left = {l}, Right = {r}, Max Length = {max_len}")
        plt.plot([l, r], [i, i], marker='o', label=f"Step {i+1}: {window}")

    plt.title("Sliding Window Visualization")
    plt.xlabel("String Index")
    plt.ylabel("Steps")
    plt.legend()
    plt.show()

# Example usage
visualize_sliding_window("eceba", 2)
'''
Notes: Variable Sliding Window with Two Pointers
Right Pointer
Movement: Always moves to the right to expand the window.
Purpose: Includes more elements in the current window to explore potential solutions.
Left Pointer
Movement: Moves towards the right only when the condition is violated (e.g., exceeding k distinct characters).
Purpose: Shrinks the window by removing elements from the left to restore validity.
Behavior: Adjusts dynamically based on the window's validity.
Key Interaction Between the Pointers
The right pointer progresses linearly through the array/string.
The left pointer adjusts its position forward as needed to maintain or restore the window's validity.
Why the Left Pointer Doesn't Move Backward?
In the variable sliding window, the left pointer only moves to the right.
Efficiency: This guarantees a time complexity of O(n),
as each pointer moves through the array at most once.
Backward Movement: Allowing the left pointer to move backward would make the algorithm inefficient, invalidating the linear time complexity.
Clarifying "Two Pointers"
If one pointer moves exclusively to the right (expanding the window) and the other moves both left and right, this deviates from the typical sliding window approach.
In the sliding window approach:
The right pointer expands the window by moving forward.
The left pointer shrinks the window by moving forward, but only when necessary.
This structure ensures efficiency and adheres to the sliding window paradigm.

'''