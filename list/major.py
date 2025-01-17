'''to add, remove, and copy elements in a list in Python, you can use the following methods:

1. Adding elements:
Using append(): Adds an element to the end of the list.'''


my_list = [1, 2, 3]
my_list.append(4)
print(my_list)  # Output: [1, 2, 3, 4]
'''Using insert(): Adds an element at a specific index.'''

my_list = [1, 2, 3]
my_list.insert(1, 5)  # Insert 5 at index 1
print(my_list)  # Output: [1, 5, 2, 3]
'''Using extend(): Adds multiple elements from another iterable (e.g., list) to the end of the list.'''


my_list = [1, 2, 3]
my_list.extend([4, 5, 6])
print(my_list)  # Output: [1, 2, 3, 4, 5, 6]
'''2. Removing elements:
Using remove(): Removes the first occurrence of a specific element.
'''

my_list = [1, 2, 3, 2, 4]
my_list.remove(2)  # Removes the first occurrence of 2
print(my_list)  # Output: [1, 3, 2, 4]
'''Using pop(): Removes and returns an element at a specific index (default is the last element).'''


my_list = [1, 2, 3, 4]
popped_element = my_list.pop(2)  # Removes element at index 2 (3)
print(my_list)  # Output: [1, 2, 4]
print(popped_element)  # Output: 3
'''Using clear(): Removes all elements from the list.'''


my_list = [1, 2, 3]
my_list.clear()
print(my_list)  # Output: []
'''3. Copying elements:
Using copy() method: Creates a shallow copy of the list.'''

my_list = [1, 2, 3]
copied_list = my_list.copy()
print(copied_list)  # Output: [1, 2, 3]
'''Using list slicing: Copies the list by slicing.'''


my_list = [1, 2, 3]
copied_list = my_list[:]
print(copied_list)  # Output: [1, 2, 3]
'''Using copy module: Creates a deep copy (useful if the list contains mutable objects).'''


import copy
my_list = [[1, 2], [3, 4]]
deep_copy = copy.deepcopy(my_list)
print(deep_copy)  # Output: [[1, 2], [3, 4]]
'''These are the basic methods to manipulate lists in Python.'''







'''

1. Using + Operator (Concatenation):
This combines two lists into one by concatenating them.'''


list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2
print(combined_list)  # Output: [1, 2, 3, 4, 5, 6]
'''2. Using extend() Method:
This adds all elements of list2 to the end of list1.'''


list1 = [1, 2, 3]
list2 = [4, 5, 6]
list1.extend(list2)
print(list1)  # Output: [1, 2, 3, 4, 5, 6]
'''3. Using append() Method in a Loop:
You can also use append() in a loop to add each element from one list to another.'''


list1 = [1, 2, 3]
list2 = [4, 5, 6]
for item in list2:
    list1.append(item)
print(list1)  # Output: [1, 2, 3, 4, 5, 6]
'''4. Using * Unpacking (Python 3.5+):
This method allows you to unpack the lists and combine them into one.'''

list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = [*list1, *list2]
print(combined_list)  # Output: [1, 2, 3, 4, 5, 6]
'''Each method can be used based on the context, with extend() modifying the original list and others creating a new combined list.
'''





