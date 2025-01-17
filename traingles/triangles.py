n = 5
for i in range(1, n + 1):
    print('*' * i)
'''
*
**
***
****
*****
'''
n = 5
for i in range(n, 0, -1):
    print('*' * i)
'''
*****
****
***
**
*

'''
n = 5
for i in range(1, n + 1):
    print(' ' * (n - i) + '*' * (2 * i - 1))
'''
    *
   ***
  *****
 *******
*********
'''
n = 5
for i in range(n, 0, -1):
    print(' ' * (n - i) + '*' * (2 * i - 1))
'''
*********
 *******
  *****
   ***
    *

'''
n = 5
[print('*' * i) for i in range(1, n + 1)]
'''
*
**
***
****
*****

'''
def pascals_triangle(n):
    row = [1]
    for _ in range(n):
        print(' '.join(map(str, row)).center(n * 2))
        row = [x + y for x, y in zip([0] + row, row + [0])]

pascals_triangle(5)
'''
    1    
   1 1   
  1 2 1  
 1 3 3 1 
1 4 6 4 1

'''
n = 5
for i in range(1, n + 1):
    if i == 1 or i == n:
        print('*' * i)
    else:
        print('*' + ' ' * (i - 2) + '*')
'''
*
**
* *
*  *
*****

'''
n = 5
for i in range(1, n + 1):
    if i == 1:
        print(' ' * (n - i) + '*')
    elif i == n:
        print('*' * (2 * i - 1))
    else:
        print(' ' * (n - i) + '*' + ' ' * (2 * i - 3) + '*')
'''
    *
   * *
  *   *
 *     *
*********

'''
n = 5
for i in range(1, n + 1):
    print(''.join(str(j) for j in range(1, i + 1)))
'''
1
12
123
1234
12345
'''
n = 5
num = 1
for i in range(1, n + 1):
    for j in range(1, i + 1):
        print(num, end=" ")
        num += 1
    print()
'''
1 
2 3 
4 5 6 
7 8 9 10 
11 12 13 14 15
'''
n = 5
for i in range(1, n + 1):
    for j in range(1, i + 1):
        print((i + j) % 2, end=" ")
    print()
'''
1 
0 1 
1 0 1 
0 1 0 1 
1 0 1 0 1

'''
n = 5
for i in range(1, n + 1):
    print(' ' * (n - i) + '*' * i)
'''
    *
   **
  ***
 ****
*****
'''
n = 5
for i in range(1, n + 1):
    print(' ' * (n - i) + '*' * (2 * i - 1))
for i in range(n - 1, 0, -1):
    print(' ' * (n - i) + '*' * (2 * i - 1))
'''
    *
   ***
  *****
 *******
*********
 *******
  *****
   ***
    *
'''
n = 5
for i in range(1, n + 1):
    print(' ' * (n - i) + ' '.join(str(j) for j in range(1, i + 1)))
'''
    1
   1 2
  1 2 3
 1 2 3 4
1 2 3 4 5
'''

n = 5
for i in range(1, n + 1):
    print(' ' * (n - i) + ''.join(str(j % 10) for j in range(1, i + 1)))
'''
    1
   12
  123
 1234
12345
'''
n = 5
for i in range(n, 0, -1):
    print(' ' * (n - i) + '*' * (2 * i - 1))
'''
*********
 *******
  *****
   ***
    *
'''
n = 5
for i in range(1, n + 1):
    print('*' * i)
for i in range(n - 1, 0, -1):
    print('*' * i)
'''
*
**
***
****
*****
****
***
**
*
'''
n = 5
for i in range(1, n + 1):
    print(' '.join(str(i) for _ in range(i)))
'''
1
2 2
3 3 3
4 4 4 4
5 5 5 5 5

'''
n = 5
alpha = 65  # ASCII value of 'A'
for i in range(1, n + 1):
    print(' ' * (n - i) + ' '.join(chr(alpha + j) for j in range(i)))
'''
    A
   A B
  A B C
 A B C D
A B C D E
'''
n = 5
for i in range(1, n + 1):
    print(''.join('*' if (i + j) % 2 == 0 else ' ' for j in range(n)))
'''
* * *
 * * 
* * *
 * * 
* * *
'''
n = 5
for i in range(1, n + 1):
    print(' '.join(str(j) for j in range(1, i + 1)))
'''
1
1 2
1 2 3
1 2 3 4
1 2 3 4 5
'''
n = 5
for i in range(1, n + 1):
    if i == 1:
        print(' ' * (n - i) + '*')
    elif i == n:
        print('*' * (2 * i - 1))
    else:
        print(' ' * (n - i) + '*' + ' ' * (2 * i - 3) + '*')
'''
    *
   * *
  *   *
 *     *
*********
'''
n = 5
for i in range(1, n + 1):
    print(' '.join(str((i + j) % 2) for j in range(i)))
'''
1
0 1
1 0 1
0 1 0 1
1 0 1 0 1
'''
n = 5
for i in range(n, 0, -1):
    print(' ' * (n - i) + '*' * (2 * i - 1))
for i in range(2, n + 1):
    print(' ' * (n - i) + '*' * (2 * i - 1))
'''
*********
 *******
  *****
   ***
    *
   ***
  *****
 *******
*********
'''
n = 5
num = 1
for i in range(1, n + 1):
    print(' '.join(str(num) for _ in range(i)))
    num += 1
'''
1
2 2
3 3 3
4 4 4 4
5 5 5 5 5
'''
n = 5
for i in range(n, 0, -1):
    print(' ' * (n - i) + ' '.join(str(j) for j in range(1, i + 1)))
'''
1 2 3 4 5
 1 2 3 4
  1 2 3
   1 2
    1
'''
n = 5
for i in range(1, n + 1):
    print(' ' * (n - i) + ' '.join(str(i) for _ in range(2 * i - 1)))
for i in range(n - 1, 0, -1):
    print(' ' * (n - i) + ' '.join(str(i) for _ in range(2 * i - 1)))
'''
    1
   2 2 2
  3 3 3 3 3
 4 4 4 4 4 4 4
5 5 5 5 5 5 5 5 5
 4 4 4 4 4 4 4
  3 3 3 3 3
   2 2 2
    1

'''
n = 5
for i in range(n, 0, -1):
    print(' '.join(str((i + j) % 2) for j in range(2 * i - 1)))
for i in range(2, n + 1):
    print(' '.join(str((i + j) % 2) for j in range(2 * i - 1)))
'''
1 0 1 0 1 0 1 0 1
0 1 0 1 0 1 0
1 0 1 0 1
0 1 0
1
0 1 0
1 0 1 0 1
0 1 0 1 0 1 0
1 0 1 0 1 0 1 0 1
'''
n = 5
alpha = 65  # ASCII value of 'A'
for i in range(1, n + 1):
    print(' '.join(chr(alpha + j) for j in range(i)))
'''
A
A B
A B C
A B C D
A B C D E

'''
n = 5
alpha = 65
for i in range(n, 0, -1):
    print(' '.join(chr(alpha + j) for j in range(i)))
'''
A B C D E
A B C D
A B C
A B
A

'''
n = 5
alpha = 65
for i in range(1, n + 1):
    print(' ' * (n - i) + ' '.join(chr(alpha + j) for j in range(i)))
'''
    A
   A B
  A B C
 A B C D
A B C D E


'''
n = 5
alpha = 65
for i in range(1, n + 1):
    if i == 1 or i == n:
        print(' '.join(chr(alpha + j) for j in range(i)))
    else:
        print(chr(alpha) + ' ' * (2 * (i - 1) - 1) + chr(alpha + i - 1))
'''
A
A B
A   C
A     D
A B C D E

'''
n = 5
alpha = 65
for i in range(n, 0, -1):
    print(' '.join(chr(alpha + j) for j in range(i)))
'''
A B C D E
A B C D
A B C
A B
A
'''
n = 5
alpha = 65
for i in range(1, n + 1):
    print(' ' * (n - i) + ' '.join(chr(alpha + j) for j in range(i)))
'''
    A
   A B
  A B C
 A B C D
A B C D E

'''
n = 5
alpha = 65
for i in range(1, n + 1):
    print(' ' * (n - i) + ' '.join(chr(alpha + j) for j in range(i)))
for i in range(n - 1, 0, -1):
    print(' ' * (n - i) + ' '.join(chr(alpha + j) for j in range(i)))
'''
    A
   A B
  A B C
 A B C D
A B C D E
 A B C D
  A B C
   A B
    A

'''
n = 5
alpha = 65
for i in range(1, n + 1):
    print(' ' * (n - i) + ' '.join(chr(alpha + j) for j in range(i)) + ' '.join(chr(alpha + j) for j in range(i - 2, -1, -1)))
'''
    A
   A B A
  A B C B A
 A B C D C B A
A B C D E D C B A

'''
n = 5
alpha = 65
for i in range(0, n):
    print(' ' * i + ' '.join(chr(alpha + j) for j in range(n - i)))
'''
A B C D E
 A B C D
  A B C
   A B
    A

'''
n = 5
alpha = 65
for i in range(1, n + 1):
    print(''.join(chr(alpha + j) if (i + j) % 2 == 0 else ' ' for j in range(i)))
'''
A
 B
C D
 E 
F G H

'''
n = 5
alpha = 65
for i in range(1, n + 1):
    print(' '.join(chr(alpha + j) for j in range(i)))
for i in range(n - 1, 0, -1):
    print(' '.join(chr(alpha + j) for j in range(i)))
'''
A
A B
A B C
A B C D
A B C D E
A B C D
A B C
A B
A

'''
n = 5
alpha = 65
for i in range(1, n + 1):
    print(' '.join(chr(alpha + i - 1) for _ in range(i)))
'''
A
B B
C C C
D D D D
E E E E E

'''
