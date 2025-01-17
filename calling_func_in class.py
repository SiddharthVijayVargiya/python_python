# Summary of When to Use Each Type of Method in Python Classes

# Static Method
# - How to Call: ClassName.function()
# - When to Use:
#   - Use a static method when the function does not require access to instance-specific data (`self`) or class-specific data (`cls`).
#   - These methods are general-purpose utilities that are logically related to the class but do not depend on the class or instance itself.
#   - You can define a static method using the `@staticmethod` decorator.

class MathHelper:
    @staticmethod
    def add(a, b):
        return a + b

# Example Usage of Static Method:
result = MathHelper.add(5, 3)  # Call the static method directly using the class name
print(result)  # Output: 8

# Class Method
# - How to Call: ClassName.function() or cls.function()
# - When to Use:
#   - Use a class method when the function works with class-level data or needs to modify the class state.
#   - These methods are defined using the `@classmethod` decorator and take `cls` as the first parameter.
#   - The `cls` parameter refers to the class itself and can be used to access or modify class variables.

class ExampleClass:
    class_variable = "I am a class variable."

    @classmethod
    def modify_class_variable(cls, value):
        cls.class_variable = value

# Example Usage of Class Method:
ExampleClass.modify_class_variable("New value")
print(ExampleClass.class_variable)  # Output: New value

# Instance Method
# - How to Call: object_name.function()
# - When to Use:
#   - Use an instance method when the function needs to operate on instance-specific data.
#   - These methods are the most common and are defined with `self` as the first parameter.
#   - The `self` parameter refers to the specific instance of the class and allows the method to access or modify the instance’s attributes.

class Student:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, {self.name}!")

# Example Usage of Instance Method:
student = Student("Alice")
student.greet()  # Output: Hello, Alice!

# Additional Notes:
# - Static methods do not have access to `self` or `cls`.
# - Class methods have access to `cls`, which refers to the class, but not `self`.
# - Instance methods have access to `self`, which refers to the specific instance of the class.
#
# - You can technically call an instance method using the class name, but you need to pass the instance explicitly as the first argument.
#   Example:
#       Student.greet(student)  # Same as student.greet(), but less readable and not recommended.
