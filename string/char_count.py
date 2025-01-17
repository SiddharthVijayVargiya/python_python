string = 'hellow'
char_count = {}
for char in string :
    char_count[char] = char_count.get(char,0)+1
print(char_count)
print(len(char_count))

output = {'h': 1, 'e': 1, 'l': 2, 'o': 1, 'w': 1}
'''
Example Walkthrough
For my_string = "hello":

First Iteration (char = 'h'):

char_count.get('h', 0) returns 0 (default value).
0 + 1 gives 1.
Add 'h': 1 to char_count.
Second Iteration (char = 'e'):

char_count.get('e', 0) returns 0 (default value).
0 + 1 gives 1.
Add 'e': 1 to char_count.
Third Iteration (char = 'l'):

char_count.get('l', 0) returns 0 (default value).
0 + 1 gives 1.
Add 'l': 1 to char_count.
Fourth Iteration (char = 'l'):

char_count.get('l', 0) now returns 1 (current value).
1 + 1 gives 2.
Update 'l' to 2 in char_count.
Fifth Iteration (char = 'o'):

char_count.get('o', 0) returns 0 (default value).
0 + 1 gives 1.
Add 'o': 1 to char_count.


'''