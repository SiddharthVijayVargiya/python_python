'''
Question:
You need to design a Book class to represent a book in a library. 
The class should include the following functionality:

Attributes:

title: The title of the book.
author: The author of the book.
year: The publication year of the book.
is_available: A boolean attribute that indicates whether the book is available for borrowing (default value: True).
Methods:

borrow(): Marks the book as borrowed, i.e., sets is_available to False.
return_book(): Marks the book as returned, i.e., sets is_available to True.
book_info(): Returns a string with the book's title, author, year, and availability status.

'''


class Book:
    def __init__(self, title, author, year, is_available=True):
        # Initialize the attributes
        self.title = title
        self.author = author
        self.year = year
        self.is_available = is_available

    def borrow(self):
        # Mark the book as borrowed
        if self.is_available:
            self.is_available = False
            print(f"You have borrowed the book: '{self.title}' by {self.author}.")
        else:
            print(f"Sorry, the book '{self.title}' by {self.author} is not available.")

    def return_book(self):
        # Mark the book as returned
        self.is_available = True
        print(f"The book '{self.title}' by {self.author} has been returned.")

    def book_info(self):
        # Return the book's details and availability
        availability = "available" if self.is_available else "not available"
        return f"Title: {self.title}\nAuthor: {self.author}\nYear: {self.year}\nAvailability: {availability}"

# Example usage:
book1 = Book("1984", "George Orwell", 1949)

# Display the book information
print(book1.book_info())  # Should show that the book is available

# Borrow the book
book1.borrow()  # Should mark the book as borrowed

# Display the updated book information
print(book1.book_info())  # Should show that the book is not available

# Return the book
book1.return_book()  # Should mark the book as returned

# Display the final book information
print(book1.book_info())  # Should show that the book is available again
