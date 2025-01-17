# Dictionary For Loops in Python

# Basic Dictionary Structure
my_dict = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}

# 1. Iterating Over Keys
# By default, iterating over a dictionary in a for loop gives its keys.
for key in my_dict:
    print(key)  # Output: name, age, city

# Alternatively:
for key in my_dict.keys():
    print(key)  # Output: name, age, city

# 2. Iterating Over Values
# To iterate over the values of a dictionary:
for value in my_dict.values():
    print(value)  # Output: Alice, 25, New York

# 3. Iterating Over Key-Value Pairs
# To access both keys and values simultaneously, use the items() method:
for key, value in my_dict.items():
    print(f"{key}: {value}")
# Output:
# name: Alice
# age: 25
# city: New York

# 4. Iterating in Sorted Order
# If you need to iterate in a sorted order (by keys):
for key in sorted(my_dict.keys()):
    print(f"{key}: {my_dict[key]}")
# Output:
# age: 25
# city: New York
# name: Alice

# 5. Using enumerate with Dictionaries
# You can combine enumerate with items() to get the index while iterating through key-value pairs:
for index, (key, value) in enumerate(my_dict.items()):
    print(f"{index}: {key} -> {value}")
# Output:
# 0: name -> Alice
# 1: age -> 25
# 2: city -> New York

# 6. Looping Through Nested Dictionaries
# If a dictionary contains nested dictionaries, use nested for loops to iterate:
nested_dict = {
    "person1": {"name": "Alice", "age": 25},
    "person2": {"name": "Bob", "age": 30}
}

for key, sub_dict in nested_dict.items():
    print(f"{key}:")
    for sub_key, sub_value in sub_dict.items():
        print(f"  {sub_key}: {sub_value}")
# Output:
# person1:
#   name: Alice
#   age: 25
# person2:
#   name: Bob
#   age: 30

# 7. Dictionary Comprehension (Bonus)
# You can use dictionary comprehension to create or transform dictionaries in a concise way:

# Example: Square values of a dictionary
numbers = {"a": 1, "b": 2, "c": 3}
squared_numbers = {key: value**2 for key, value in numbers.items()}
print(squared_numbers)  # Output: {'a': 1, 'b': 4, 'c': 9}
