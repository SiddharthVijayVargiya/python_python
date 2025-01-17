# Define a class named MyClass
class MyClass:
    # Constructor method to initialize object attributes
    # This method is automatically called when an object of this class is created
    def __init__(self, name, age):
        self.name = name  # Attribute 'name' is assigned the value passed during object creation
        self.age = age    # Attribute 'age' is assigned the value passed during object creation

    # A method inside the class to display a greeting message
    def greet(self):
        # Access the object's 'name' and 'age' attributes using 'self'
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

'''
Explanation of Each Part
class MyClass:

Defines a new class named MyClass. A class is a blueprint for creating objects.
def __init__(self, name, age):

This is the constructor method, denoted by __init__.
It is called automatically when an object of the class is created.
self refers to the current instance of the class, allowing access to the attributes and methods.
self.name = name and self.age = age

These lines initialize the attributes of the object.
self.name and self.age are instance attributes, which hold the values passed as name and age during object creation.
def greet(self):

This defines a method named greet for the class.
Methods are functions that belong to a class and operate on its objects.
print(f"Hello, my name is {self.name} and I am {self.age} years old.")

This line prints a personalized greeting message.
self.name and self.age fetch the values of the respective attributes for the current object.
'''


# Define the class
class MyClass:
    # Constructor to initialize attributes
    def __init__(self, name, age):
        self.name = name  # Assign the name provided to the 'name' attribute
        self.age = age    # Assign the age provided to the 'age' attribute

    # A method to display a greeting message
    def greet(self):
        # Use the object's attributes to create a personalized message
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

# Create an object of MyClass
# The 'name' and 'age' attributes are initialized as 'Alice' and 25
person = MyClass("Alice", 25)

# Access attributes directly
print(person.name)  # Output: Alice
print(person.age)   # Output: 25

# Call the greet method to display the message
person.greet()  # Output: Hello, my name is Alice and I am 25 years old.


"or"

MyClass.greet(person)# another way of function calling 




'''
so in class wer use constructor to define the structure of data ?
ChatGPT said:
ChatGPT
Yes, exactly! In a class, the constructor (defined as __init__() in Python) 
is used to define and initialize the structure of data that an object of the class will have. 
It serves as a blueprint for creating objects with specific attributes.


'''