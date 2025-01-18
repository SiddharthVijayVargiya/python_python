def febonacci(n):
    if n <=1:
        return n 
    else :
        return febonacci(n-2)+ febonacci(n-1)
def generate_fibonacci_series(n):
    return [febonacci(i) for i in range(n)]
print(generate_fibonacci_series(4))


'''
Step-by-Step Visualization:
Recursive Breakdown for Each i in range(4):
febonacci(0):

Base case: n <= 1
Returns 0.
febonacci(1):

Base case: n <= 1
Returns 1.
febonacci(2):

Recursive case: febonacci(2) = febonacci(0) + febonacci(1)
febonacci(0) = 0
febonacci(1) = 1
Returns 0 + 1 = 1.
febonacci(3):

Recursive case: febonacci(3) = febonacci(1) + febonacci(2)
febonacci(1) = 1
febonacci(2) = febonacci(0) + febonacci(1)
febonacci(0) = 0
febonacci(1) = 1
febonacci(2) = 0 + 1 = 1
Returns 1 + 1 = 2.
Generated Fibonacci Series:
The results for each i are combined:



'''