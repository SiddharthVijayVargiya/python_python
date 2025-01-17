class Node:
    def __init__(self, data):
        self.data = data  # Data part
        self.next = None  # Pointer to the next node

class LinkedList:
    def __init__(self):
        self.head = None  # Initialize the head of the linked list
    
    def append(self, data):
        new_node = Node(data)  # Create a new node
        if not self.head:
            self.head = new_node  # If the list is empty, set the new node as the head
            return
        
        current = self.head
        while current.next:  # Traverse to the last node
            current = current.next
        current.next = new_node  # Link the new node at the end
    
    def display(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None") 

# Example usage
ll = LinkedList()
ll.append(10)
ll.append(20)
ll.append(30)
ll.display()
