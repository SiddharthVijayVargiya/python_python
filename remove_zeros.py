def removezeros(s: list) -> list:
    non_zero = 0
    for i in range(len(s)):
        if s[i] != 0:
            s[non_zero] = s[i]
            non_zero += 1
    for j in range(non_zero, len(s)):  # Corrected loop to iterate over remaining indices
        s[j] = 0
    return s

s = [0, 1, 0, 1, 2, 3, 4]
print(removezeros(s))
