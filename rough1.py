'''def smallest_array(arr,s):
    left = 0
    window_sum = 0
    min_length = float('inf')
    for right in range(len(arr)):
        window_sum += arr[right]
        while s<= window_sum:
            min = min(min_length,right-left+1)
    return min_length if min_length != '''
    
'''def smallest_sum(arr, s):
    left = 0
    min_length = float('inf')
    window_sum = 0
    
    for right in range(len(arr)):
        window_sum += arr[right]  # Add the current element to the window sum

        # Shrink the window from the left until window_sum is smaller than s
        while window_sum >= s:
            # Update the minimum length of the subarray
            min_length = min(min_length, right - left + 1)
            window_sum -= arr[left]  # Remove the leftmost element
            left += 1  # Shrink the window by moving the left pointer

    # If no valid window was found, return 0
    return min_length if min_length != float('inf') else 0

# Example usage
arr = [1, 2, 3, 4, 5, 6]
s = 7
print(smallest_sum(arr, s))  # Output: 1 (subarray [7] or [4, 3])'''

'''
def matrix_multiplication(mat1,mat2):
    a_rows = len(mat1)
    a_cols = len(mat1[0])
    b_rows = len(mat2)
    b_cols = len(mat2[0])
    if a_cols != b_cols :
        return f"this matrix is not valid"
    for i in range(a_rows):
        for j in  range(b_cols):
            for k in range(b_rows):
               '''
               
'''def matrix_multiplication(m1,m2):
    m1r = len(m1)
    m1c =  len(m1[0])
    
    m2r = len(m2)
     
    m2c = len(m2[0])
    result = [[0 for _ in range(m2c)] for _ in range(m1r)]

    
    if m1c != m2r:
        return f"this is nt valid"
    for i in range(m1r):
        for j in range(m2c):
            for k in range(m2r):
                result[i][j]+= m1[i][k]*m2[k][j]
    '''
'''def variable_size(arr,k):
    left = 0 
    window_sum =0 
    min_length =float("inf")
    for right in range(len(arr)):
        window_sum += arr[right]
        while window_sum>=k:
            right+=1
            min = min(min_length,right-left+1)
            window_sum -= arr[left]
            left += 1'''
'''def two_Pointer(arr,s):
    left = 0 
    right = len(arr)-1
    while left < right :
        if arr[left] == arr[right]:
            return f"afanmfdakf"
        elif :
            right-=1
        else:
            left +=1'''
'''def binary_search(arr,s):
    left =0
    right =0
    middle_value =len[arr]//2 #odd'''
'''def fibonacci(n):
    if n<1:
        return n
    return fibonacci(n-1)+fibonacci(n-2)
n = 4
print(fibonacci(n))'''
'''arr = ["ababa","abhhh","abfjfjaj","ab","a",""]
def longest_Common_prefix(arr):
    prefix = arr[0]
    for string in arr[:1]:
        while 
    '''
'''def longest_prefix(arr):
    if not arr:
        return ""
    prefix = arr[0]
    for string in arr[1:]:
        while string[:len(prefix)] != prefix and prefix :
            prefix = prefix[:-1]
        if not prefix:
            break
    return prefix
arr = ['aaa','aaannn','aaahaha']
print(longest_prefix(arr))'''
'''def matrix_multiplication(m1,m2):
    m1r = len(m1)
    m1c = len(m1[0])
    m2r = len(m2)
    m2c = len(m2[0])
    result = [[0 for _ in range(m2c)]for _ in range(m2r)]    
    for i in range(m1r):
        for j in range (m2c):
            for k in range(m1c):
                result[i][j] += m1[i][k]*m2[k][j]
    return result
m1 = [[2,2],[2,2]]
m2 = [[2,2],[2,2]]
print(matrix_multiplication(m1,m2))'''
'''arr = ['aaaa','aabbbb','aaaabbb']
def longest_substring_prefix(arr):
    prefix = arr[0]
    if prefix not in arr:
        return
    for string in arr[1:]:
        while string[:len(prefix)] != prefix and prefix :
            prefix = prefix[:-1]
        if not prefix :
            break
    return prefix     
print(longest_substring_prefix(arr))'''
'''for i in range(3):
    for j in range (i+1):
        print("*", end=" ")'''
        
        

'''def variable_sliding_window(arr,k):
    
    left = 0
    window_Sum = 0
    minimum_length = float('inf')
    for right in range(len(arr)):
        window_sum += arr[right]
        while S>= window_Sum:'''
        
'''lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(lst[::3])'''
'''import re
import json

# Function to clean and convert WhatsApp chat to desired format
def parse_whatsapp_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        chat_data = f.readlines()

    conversation = []
    parsed_data = []

    # Regex to match chat lines (adjust based on your chat format)
    message_pattern = re.compile(r"^\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M - (.*?): (.*)$")
    
    for line in chat_data:
        match = message_pattern.match(line)
        if match:
            sender, message = match.groups()
            
            # Add messages to conversation history
            conversation.append(f"{sender}: {message}")

            # Create a new entry when we have both User and Friend message
            if len(conversation) >= 2:
                parsed_data.append({
                    "conversation": " <sep> ".join(conversation[:-1]),  # conversation history (all but last)
                    "response": conversation[-1]  # the latest response
                })
                conversation = [conversation[-1]]  # Keep the latest message for context
    
    # Save the parsed data to a JSON file
    with open('whatsapp_conversations.json', 'w', encoding='utf-8') as outfile:
        json.dump(parsed_data, outfile, ensure_ascii=False, indent=4)

# Usage
parse_whatsapp_chat('your_whatsapp_chat.txt')'''

'''lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
del lst[::3]  # Removes every 3rd element
print(lst)  '''
'''lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
lst[::2] 
print(lst[::2])  '''
'''import torch
x = torch.linspace(start=1,end=2,steps=3).unsqueeze(0)
print(x)'''
'''
import torch 
import torch.nn as nn 
import torch.optim as optim
class LinearRegression(nn.Module):
    def __init__(self):
        super (LinearRegression,self).__init__()
        self.layer1 = nn.Linear(10,5)
        self.layer2  = nn.Linear(5,3)
        self.layer3  = nn.Linear(3,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.relu(self.layer3(x))
        return x
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)
x = torch.randn(100,10)
y = torch.randn(100,1)
epochs = 1000
for epoch in range(epochs):
    
    y_pred = model(x)
    loss = criterion(y_pred , y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
if (epoch+1)%100 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')'''
''''import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 3)
        self.layer3 = nn.Linear(3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x

# Instantiate the model
model = LinearRegression()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data (replace with your actual data)
x = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randn(100, 1)   # 100 target values

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass: compute predicted y by passing x to the model
    y_pred = model(x)
    
    # Compute the loss
    loss = criterion(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()  # Zero out the gradients from the previous step
    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
'''
'''import torch
import torch.nn as nn
import torch.optim as optim
class SimpleLinearRegression(nn.module):
    def __init__(self):
        super(SimpleLinearRegression,self).__init__()
        self.layer1 = nn.Linear(10,5)
        self.layeer2 = nn.Linear(5,1)
        self.relu = nn.ReLU()
    def forward(self):
        x = self.layer1(x)
        x = self.layeer2(x)
        return self.relu(x)
model = SimpleLinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameter(),lr = 0.01)
epochs =1000
for epoch in range(epochs):
    y = model(x)
    loss = criterion()
    '''
'''import torch
x =torch.randn(1,10)
y = torch.rand(1,10)
print(x)
print(y)'''
'''import torch

# Loop to generate 50 feature tensors and corresponding labels
for i in range(50):
    feature = torch.rand(1, 10)  # Generate a random feature tensor of shape (1, 10)
    
    # Example: Compute label by summing the feature values (as an example label generation process)
    label = feature.sum()  # or you can perform other operations on the feature

    # Print the feature and the computed label
    print("Feature:", feature)
    print("Label:", label)'''

'''import requests
from bs4 import BeautifulSoup

# URL of the page containing the table
url = 'your_target_url_here'

# Send a GET request to fetch the content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table
table = soup.find('table', class_='table-striped')

# Extract table headers
headers = []
for th in table.find_all('th'):
    headers.append(th.text.strip())

# Extract table rows
rows = []
for tr in table.find_all('tr')[2:]:  # Skip header and filter row
    cells = tr.find_all('td')
    if len(cells) > 1:  # Ensure it's a valid row
        row = {
            'S.No': cells[0].text.strip(),
            'ISBN': cells[1].text.strip(),
            'Title': cells[2].text.strip(),
            'Author': cells[3].text.strip(),
            'Year': cells[4].text.strip(),
            'Publisher': cells[5].text.strip(),
            'Link': cells[6].find('a')['href']
        }
        rows.append(row)

# Print the scraped data
for row in rows:
    print(row)
'''
'''import torch
import torch.nn as nn
import torch.optim as optim
x = torch.tensor([[1],[2],[3],[4]],dtype = torch.float32)
y = torch.tensor([5],[6],[7],[8],dtype= torch.float32)
class Linearregression(nn.Module):
    def __init__(self):
        super(Linearregression,self).__init__()
        self.x = nn.Linear(1,1)
    def forward(self):
        return '''
'''import torch
import torch.nn as nn 
import torch.optim as optim

features = torch.randn(1000,10)
weights = torch.randn(10,1)
labels = features@weights + features*weights+0.01
class LinearRegressiontry(nn.Module):
    def __init__(self):
        super(LinearRegressiontry,self).__init__()
        self.layer1 = nn.Linear(10,1)
    def forward(self,input):
        return self.layer1(input)
model = LinearRegressiontry()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr =0.01)
epochs = 1000
for epoch in range(epochs):
    prediction = model(features)
    loss = criterion(prediction,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%100 ==0 :
        print(epoch)'''
'''arr = [1,2,3,4,4,6,6]
left = 0
right= len(arr)-1
mid = (left+right)//2
while left > right :
    '''
'''sr =[1,2,3,4,5,6,9,8]
k = len(sr)
print(k)    
for i in range(k-1):
    print(i)'''
'''import torch
import torch.nn as nn
import torch.optim as optim

features = torch.randn(1000,10)
labels = torch.randn(10,1)
weights = features@labels + torch.randn(1000,1)*0.01


class Linearregression(nn.Module):
    def __init__(self):
        super(Linearregression,self).__init__()
        self.layer = nn.Linear(in_features=features,out_features=labels)
    def forward(self,input):
        return self.layer(input)
model = Linearregression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters,lr= 0.01)
epochs = 1000
for epoch in range(epochs):
    prediction = model(features)
    loss = criterion(prediction,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        with torch.no_grad():  # Disable gradient computation for validation.
            val_predictions = model(val_features)  # Generate predictions on the validation data.
            val_loss = criterion(val_predictions, val_labels)  # Calculate the validation loss.
        # Print the current training loss and validation loss.
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Step 7: Test the model
with torch.no_grad():  # Disable gradient computation for testing.
    test_predictions = model(val_features)  # Generate predictions on the validation set.
    # Display the first 5 predictions and their corresponding actual labels for comparison.
    print("\nSample Predictions:\n", test_predictions[:5])
    print("Actual Labels:\n", val_labels[:5])
'''

'''def matrixmultiplication(m1,m2):
    m1c = len(m1[0])
    m1r = len(m1)
    m2c = len(m2[0])
    m2r = len(m1)
    result = [ [0 for i in range(m1c)]for j in range(m2r)]
    for i in range(m1r):
        for j in range(m2c):
            for k in range(m2r):
                result += '''
'''s = "{[()]}"
x = [x for x in s]
for i in x :
    if x[i] == '[' and x[i] == '{' and x =='(':
        
'''




'''head = [1,1,2,3,3]
y=set(head)
x = []
for i in y :

    z=x.append(i)
    print(z)
    
'''
'''import torch
import torch.nn as nn 
import torch.optim as optim



features = torch.randn(1000,10)
weights = torch.randn(10,1)
labels = features@weights + torch.randn(1000,1)*0.01
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.layer = nn.Linear(in_features=10,out_features=1)
    def forward (self,input):
        return self.layer(input)
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01 )       
epochs = 1000
for epoch in range(epochs):
    prediction = model(features)
    loss = criterion(prediction,labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%100 ==0:
        print(f" EPOCH : {epoch+1}/{epochs}, LOSS : {loss.item():.4f}")
with torch.no_grad():
    test_input = torch.randn(1, 10)  # Single sample with 10 features
    prediction = model(test_input)
    print(f"Prediction for input {test_input.tolist()}: {prediction.item():.4f}")
 j in range(m2c):
            for k in range(m2r):
                result[i][j] += m1[i][k]*m2[k][j]
    return result 
m1 = [[1,1],[1,1]]
m2 = [[1,1],[1,1]]
print(f"{matmul(m1,m2)}")
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        current = head 
        while current and current.next:
            if current.val == current.val.next:
                current.next = current.next.next
            else :
                current = current.next
        return head
                '''
                
'''my_dict = {'a': [1,2,3], 'b': 2, 'c': 3}
for key, value in zip(my_dict.keys(), my_dict.values()):
    print(key, value)
import pandas as pd
df = pd.DataFrame(my_dict)
print(df)'''
'''def matmul(m1,m2):
    m1r = len(m1)
    m1c = len(m1[0])
    m2r = len(m2)
    m2c = len(m2[0])
    if m1r != m2c :
        print('the mat mul is not possible ')
    results = [[0 for _ in range(m2c)]for _ in range(m1r)]
    for i in range(m1r):
        for j in range (m2c):
            for k in range (m1c):
                results[i][j] += m1[i][k]* m2[k][j]
    return results       
m1 = [
    [1, 2],
    [3, 4],
    [5, 6]
]

m2 = [
    [7, 8, 9],
    [10, 11, 12]
]
print(f"{matmul(m1,m2)}")'''
'''def matmul(m1,m2):
    m1r = len(m1)
    m1c = len(m1[0])
    m2r = len(m2)
    m2c = len(m2[0])
    if m1r != m2c:
        print(f"invalid matric {m1} and {m2}")
    result = [[0 for _ in range(m2c)]for _ in range(m1r)]
    for i in range(m1r):
        for'''
'''x = [1,1,1]
print(len(x))
for i in range(len(x)):
    print(i)
'''
'''n = 10
lst = [1]*n
lst1 = [1]*n
print(lst)'''
'''divisible_by_3 = [num for num in range(1, 1001) if num % 3 == 0]
print(len(divisible_by_3))'''
'''greet =lambda num :  [num for num in range(1, 1001) if num % 3 == 0]
print(len(greet(3)))'''
'''string = "hello"
char_count = {}

for char in string:
    char_count[char] = char_count.get(char, 0)+1

print(char_count)'''
'''strng = 'hellow'
char_count={}
for char in strng:
    char_count[char]= char_count.get(char,0)+1
print (char_count)
'''
'''def myAtoi(self, s: str) -> int:
    return int(s)
s = '42'
print(type(s))
for i in s :
    print(i,end= "")'''
'''def threeSumClosest(self, nums: List[int], target: int) -> int:
        left = nums[target]-1
        right = nums[target]+1
        

        if target not in nums :
            return " the string is empty "
        sum = abs(nums[left]) +abs(nums[right])
        return sum  '''
'''nums = [-1,2,1,-4], target = 1
for i in range(len(nums)):
    if nums[i] == target :
        left = i -1
        right = i +1
        sum = abs (nums[left])+abs(nums[right])+nums[i]'''
'''nums = [1,2,3,4,5,6,7,8]
for i in range(len(nums)-2):
    print(i,end = " ")
cloestest_sum = float('inf')
print(cloestest_sum)'''
'''nums = [1,2,3,4,5,6,7,8]
val = 4

nums[:] = [x for x in nums if x != val]
len(nums)
print(len(nums))''''''
''''''class Solution:
    def removeElement(self, nums: list[int], val: int) -> int:'''
'''nums = [1,2,3,4,5,6,7,8]
val = 8
x = [y for y in nums if y != val ]
print(x)'''
'''nums = [0,1,0,2,3,4]
nums.sort()
z = [a for a in nums if a == 0]
x = [y for y in nums if y >0  ]

c = x +z
print(c)'''
'''x = [0,1,0,3,4,12]
non_zero =0
for i in range(len(x)):
    if x[i] ==0:
        
        x[non_zero]= x[i]
        non_zero+=1'''
'''s = 'hellow'
char_count={}
for char in s:
    char_count[char]= char_count.get(char,0)+1
print(char_count)
def zeroreplacement(l:list)->list:
    ''''''
def removezeros(s:list)->list:
    non_zero =0
    for i in range(len(s)):
        if s[i] !=0:
            s[non_zero]= s[i]
            non_zero+=1
    for j in range(non_zero,len(s)):
        s[j] =0
    return s
s = [0,1,0,1,2,3,4]
print(removezeros(s))
sting = 'hellow'
char_count ={}
for char in sting:
    char_count[char]= char_count.get(char,0)+1
print(char_count)
       ''' 
'''class Node():
    def __init__(self,data):
        self.data = data 
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append():'''
'''class Node:
    def __init__(self, data):
        self.data = data  # Store the data
        self.next = None  # Reference to the next node (initially None)

class LinkedList:
    def __init__(self):
        self.head = None  # Initially, the list is empty

# Creating a Linked List
linked_list = LinkedList()
linked_list.head = Node(10)  # Create the first node, set head to point to it
second_node = Node(20)
linked_list.head.next = second_node  # Link the first node to the second node

        
        
'''
'''class Node :
    def  __init__(self,data):
        self.data = data 
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,data ):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return'''
'''class Node :
    def __init__(self,data):
        self.data = data 
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,data):
        new_node = Node(data)
        if self.head is None:
            
            self.head = new_node
            return
        current = self.head
        while current.next:
            '''
'''class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append (self,data):
        new_node = Node(self)
        if self.head is None:
            self.head = new_node
            return 
        current = self.head 
        while current.next:
            current = current.next
            current.next = new_node
        '''
'''class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,data):
        new_node = Node(self)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next :
            current = current.next
        current.next = new_node'''
'''class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append (self,data):
        new_node = Node(self)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
            current.next = new_node'''
'''class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,data):
        new_node = Node(self)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next :
            current = current.next
            current.next = new_node
            
class Node :
    def __init__(self,data):
        self.data= data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,data):
        new_node = Node(self)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next :
            current = current.next
            current.next =  new_node'''
'''class Node :
    def __init(self,data):
        self,data = data 
        self,head = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,data):
        new_node = Node (self)
        if self.head is None:
            self.head = new_node
            return
        current   = self.head
        while current.next:
            current = current.next
            current.next = new_node
            '''
'''class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return 
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    def display (self):
        current = self.head
        while current :
            print(current.data , end ="->" )
            current = current.next
        print(None)
linked_list = LinkedList()
linked_list.append(10)
linked_list.append(20)
linked_list.append(30)
linked_list.display()  # Output: 10 -> 20 -> 30 -> None'''
'''def recure(x):
    for i in range(len(x)):
        if x != 23 :
            print(i*recure(x))'''

        
        
'''def recur(n):
    if n == 0:
        return 1
    return n *recur(n-1)
print(recur(5))

def reverse_string(s):
    # Base case
    if len(s) <= 1:
        return s
    # Recursive step
    return reverse_string(s[1:]) + s[0]

# Example usage
print(reverse_string("hello"))  # Output: "olleh"
ar = 'abcd'
print(ar[1:])'''

'''
class Node :
    def __init__(self,data):
        self.data = data 
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return 
        current = self.head
        while current.next:
            '''
'''class Node:
    def __init__(self,data):
        self.data = data #data 
        self.next = None #pointer
class LinkedList:
    def __init__(self):
        self.head = None
    def append (self,data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return 
        current = self.head
        while current.next :
            current = '''
'''            
class Node :
    def __init__(self,data):
        self.data = data 
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head 
        while current.next :
            current = current.next
            current.next = new_node'''