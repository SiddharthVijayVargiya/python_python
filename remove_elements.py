my_list = [1, 2, 3, 2, 4]
my_list.remove(2)  # Removes the first occurrence of 2
print(my_list)     # Output: [1, 3, 2, 4]

# Raises ValueError if element not found
# my_list.remove(5)  # Uncommenting this will raise ValueError




my_list = [10, 20, 30, 40]
last_element = my_list.pop()  # Removes and returns the last element
print(last_element)           # Output: 40
print(my_list)                # Output: [10, 20, 30]

second_element = my_list.pop(1)  # Removes and returns element at index 1 (20)
print(second_element)            # Output: 20
print(my_list)                   # Output: [10, 30]




my_list = [5, 10, 15, 20]
del my_list[1]  # Deletes element at index 1 (10)
print(my_list)  # Output: [5, 15, 20]

# Deleting a slice of elements
my_list = [1, 2, 3, 4, 5]
del my_list[1:4]  # Deletes elements from index 1 to 3 (i.e., 2, 3, 4)
print(my_list)     # Output: [1, 5]




my_list = [1, 2, 3, 4, 5]
my_list.clear()  # Removes all elements from the list
print(my_list)   # Output: []





my_list = [1, 2, 3, 4, 5]
my_list = [x for x in my_list if x != 3]  # Removes all occurrences of 3
print(my_list)  # Output: [1, 2, 4, 5]




my_list = [1, 2, 3, 4, 5]
my_list = list(filter(lambda x: x != 3, my_list))  # Removes all occurrences of 3
print(my_list)  # Output: [1, 2, 4, 5]
