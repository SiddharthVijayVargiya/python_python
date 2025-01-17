'''
https://chatgpt.com/share/67570912-f560-8002-961c-c5ed6e1db84e


'''



def longestValidParentheses(s: str) -> int:
    left = right = max_length = 0

    # Left to right
    for char in s:
        if char == '(':
            left += 1
        else:
            right += 1
        if left == right:
            max_length = max(max_length, 2 * right)
        elif right > left:
            left = right = 0

    # Right to left
    left = right = 0
    for char in reversed(s):
        if char == ')':
            right += 1
        else:
            left += 1
        if left == right:
            max_length = max(max_length, 2 * left)
        elif left > right:
            left = right = 0

    return max_length

# Example usage:
s = "(()))())("
print(longestValidParentheses(s))  # Output: 4
