# What is a Function?
# A function is a block of reusable code that performs a specific task. 
# It helps to organize your code and avoid repetition.

# How to Define and Call Functions

# Defining a Function
# Use the def keyword.
# Give the function a name (e.g., greet).
# Optionally, specify parameters (inputs for the function).
# Add the function body (the code it executes).

def greet():
    print("Hello, World!")  # Function body

# Calling a Function
# Simply write the function name followed by parentheses.
greet()  # Output: Hello, World!

# Function Parameters and Arguments

# Parameters:
# These are placeholders defined in the function definition.
# They specify what inputs the function expects.
def greet(name):  # 'name' is a parameter
    print(f"Hello, {name}!")

# Arguments:
# These are the actual values passed to the function when it is called.
greet("Alice")  # 'Alice' is an argument
# Output: Hello, Alice!

# Multiple Parameters
# You can define functions with multiple parameters.
def add(a, b):
    return a + b

result = add(5, 3)  # Pass two arguments
print(result)  # Output: 8

# Return Values
# A function can return a value using the return statement.
def square(number):
    return number * number

result = square(4)  # The function returns 16
print(result)  # Output: 16

# No Return Value:
# If a function doesn’t explicitly return a value, it returns None.
def say_hello():
    print("Hello!")

result = say_hello()  # Output: Hello!
print(result)  # Output: None

# Different Ways to Call Functions

# 1. Positional Arguments
# Pass arguments in the same order as parameters.
def describe_pet(animal, name):
    print(f"I have a {animal} named {name}.")

describe_pet("dog", "Buddy")
# Output: I have a dog named Buddy.

# 2. Keyword Arguments
# Specify the parameter names when calling the function. The order doesn’t matter.
describe_pet(name="Buddy", animal="dog")
# Output: I have a dog named Buddy.

# 3. Default Parameters
# Provide default values for parameters. These are used if no arguments are passed.
def describe_pet(name, animal="dog"):
    print(f"I have a {animal} named {name}.")

describe_pet("Buddy")  # Default value used for 'animal'
# Output: I have a dog named Buddy.

# 4. Variable-Length Arguments
# Handle an unknown number of arguments using *args (for positional) or **kwargs (for keyword).

# Example with *args:
def add_numbers(*numbers):
    return sum(numbers)

print(add_numbers(1, 2, 3, 4))  # Output: 10

# Example with **kwargs:
def describe_person(**info):
    for key, value in info.items():
        print(f"{key}: {value}")

describe_person(name="Alice", age=30, city="New York")
# Output:
# name: Alice
# age: 30
# city: New York

# Nested Function Calls
# You can call one function inside another.
def multiply(a, b):
    return a * b

def square(n):
    return multiply(n, n)  # Calls multiply function

print(square(4))  # Output: 16

# Key Takeaways
# Function Definition:
# Use def to define a function.
# Specify parameters in parentheses (optional).
# Function Calling:
# Use the function name followed by parentheses.
# Pass arguments inside parentheses if required.
# Return Values:
# Use return to send a value back to the caller.
# Types of Arguments:
# Positional, keyword, default, and variable-length (*args and **kwargs).

# Practice Example
# Try writing this function:
# Define a function calculator that:
# Takes three arguments: num1, num2, and operation.
# Performs addition, subtraction, multiplication, or division based on the operation parameter.
# Call the function with different operations.

def calculator(num1, num2, operation):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2
    else:
        return "Invalid operation"

print(calculator(10, 5, "add"))  # Output: 15
print(calculator(10, 5, "multiply"))  # Output: 50
