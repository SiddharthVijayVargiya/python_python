'''
def smallest_array(arr,s):
    left = 0
    window_sum = 0
    min_length = float('inf')
    for right in range(len(arr)):
        window_sum += arr[right]
        while s<= window_sum:
            min = min(min_length,right-left+1)
    return min_length if min_length != '''
    
'''
def smallest_sum(arr, s):
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
               
'''
def matrix_multiplication(m1,m2):
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
'''
def variable_size(arr,k):
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
'''
def two_Pointer(arr,s):
    left = 0 
    right = len(arr)-1
    while left < right :
        if arr[left] == arr[right]:
            return f"afanmfdakf"
        elif :
            right-=1
        else:
            left +=1'''
'''
def binary_search(arr,s):
    left =0
    right =0
    middle_value =len[arr]//2 #odd'''
'''
def fibonacci(n):
    if n<1:
        return n
    return fibonacci(n-1)+fibonacci(n-2)
n = 4
print(fibonacci(n))'''
'''
arr = ["ababa","abhhh","abfjfjaj","ab","a",""]
def longest_Common_prefix(arr):
    prefix = arr[0]
    for string in arr[:1]:
        while 
    '''
'''
def longest_prefix(arr):
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
'''
def matrix_multiplication(m1,m2):
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
'''
arr = ['aaaa','aabbbb','aaaabbb']
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
        
        

'''
def variable_sliding_window(arr,k):
    
    left = 0
    window_Sum = 0
    minimum_length = float('inf')
    for right in range(len(arr)):
        window_sum += arr[right]
        while S>= window_Sum:'''
        
'''lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(lst[::3])'''
'''
import re
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
'''
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(lst[::2])  # Output: [0, 2, 4, 6, 8]'''
'''
def three_sum(arr):
    arr.sort()  # Sort the array first
    result = []
    
    for i in range(len(arr) - 2):  # The loop runs till len(arr) - 2 to leave space for left and right pointers
        if i > 0 and arr[i] == arr[i - 1]:  # Skip duplicates for the first element
            continue
        
        left = i + 1
        right = len(arr) - 1
        
        while left < right:
            total = arr[i] + arr[left] + arr[right]
            
            if total == 0:
                result.append([arr[i], arr[left], arr[right]])  # Append the triplet to result
                
                # Skip duplicates for the second and third elements
                while left < right and arr[left] == arr[left + 1]:
                    left += 1
                while left < right and arr[right] == arr[right - 1]:
                    right -= 1
                
                # Move both pointers inward
                left += 1
                right -= 1
            
            elif total < 0:
                left += 1  # If sum is less than 0, move the left pointer right to increase the sum
            else:
                right -= 1  # If sum is greater than 0, move the right pointer left to decrease the sum
    
    return result  # Return the list of valid triplets

# Example usage:
arr = [-1, 0, 1, 2, -1, -4]
print(three_sum(arr))'''

'''
def fixed_size_slidng(arr,k):
    current_sum = sum(arr[:k])
    window_sum = current_sum
    for i in range(k,len(arr)):
        current_sum+= arr[i]
        current_sum -= arr[i-k]
        if current_sum > window_sum:
            current_sum = window_sum
    return window_sum

'''
'''import torch
import torch.nn as nn
import torch.optim as optim
class SimpleNN(nn.Module):
    def __init__(self):
        #supercalss
        super(SimpleNN,self),__init__():
            
'''
'''
import torch
import torch.nn as nn
import torch.optim as optim 
class SimpleNN(nn.module):
    def _init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(4,2)
        self.fc2 = nn.Linear(2,4)
        self.fc4 = nn.ReLU()
        '''
'''
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Define the Neural Network class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(4, 3)  # Input layer with 4 inputs, hidden layer with 3 units
        self.fc2 = nn.Linear(3, 2)  # Hidden layer with 3 inputs, output layer with 2 units
        self.relu = nn.ReLU()       # Activation function (ReLU)

    def forward(self, x):
        # Define forward pass
        x = self.relu(self.fc1(x))  # Pass through first layer and apply ReLU
        x = self.fc2(x)             # Pass through second layer
        return x

# Step 2: Create model, define loss function and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 3: Forward pass with dummy data
input_data = torch.randn(1, 4)  # 1 sample, 4 features
target = torch.tensor([1])      # Dummy target for classification

# Forward propagation
output = model(input_data)

# Calculate loss
loss = criterion(output, target)

# Backpropagation and optimization
optimizer.zero_grad()  # Zero the gradients
loss.backward()        # Backpropagation
optimizer.step()       # Update the weights

print("Output:", output)
print("Loss:", loss.item())
'''
'''
import torch                
import torch.nn as nn       
import torch.optim as optim 

class SimpleNN(nn.Module):    
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        self.fc1 = nn.Linear(4, 3)  
        self.fc2 = nn.Linear(3, 2)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        
        x = self.relu(self.fc1(x))  
        x = self.fc2(x)
        return x

model = SimpleNN()  

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)'''
''''
import torch
import torch.nn as nn
import torch.optim as optim
class SimpleNN(nn.Module):
    def __init__(self):
        super (SimpleNN,self).__init__()
        self.layer1 = nn.Linear(in_features=10,out_features=5)
        self.layer2 = nn.Linear(in_features=5,out_features=3)
        self.layer3 = nn.Linear(in_features=3,out_features=1)
        self.relu = nn.ReLU()
    def forward(self,x):#pass the data 
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        return x
model = SimpleNN(nn.Module) 
criterion  = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
input_data = torch.randn(1,4)
target = torch.tensor([1])
output = model(input_data)
'''
'''
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(in_features=10, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=3)
        self.layer3 = nn.Linear(in_features=3, out_features=3)  # Adjust the output size to match the number of classes
        self.relu = nn.ReLU()  # Corrected: ReLU should be called with parentheses

    def forward(self, x):
        x = self.relu(self.layer1(x))  # Apply ReLU after layer1
        x = self.relu(self.layer2(x))  # Apply ReLU after layer2 for non-linearity
        x = self.layer3(x)             # No ReLU after the final output layer
        return x

# Create an instance of the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Cross entropy loss for classification
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Use SGD optimizer with a learning rate of 0.01

# Dummy input data (1 sample with 10 features) and target label
input_data = torch.randn(1, 10)
target = torch.tensor([1])  # Ensure the target is within the range [0, num_classes-1]

# Forward pass: get model predictions
output = model(input_data)

# Calculate the loss
loss = criterion(output, target)

# Backpropagation and optimization step
optimizer.zero_grad()  # Clear the previous gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update model parameters

# Print the output and loss
print("Output:", output)
print("Loss:", loss.item())
'''
'''def longest_common_prefix(strs):
    prefix = strs[0]
    for i in strs[1:]:
        while i[:len(prefix)]!= prefix and prefix:
            prefix = prefix[:-1]
    return prefix 
strs= ["flower", "flow", "flight"]
print(longest_common_prefix(strs))'''
'''def longest_prefix(arr):
    prefix = arr[0]
    for string in arr[1:]:
        while string[:len(prefix)] != prefix and prefix :
            prefix = prefix [:-1]
            '''
'''arr = "alaalal"
arr = arr[:-1]
print(arr)'''
'''n =4
for i in range(1,n+1):
    print('*'*i)'''
'''def fibonacci(n):
    if n == 0:  # Base case
        return 0
    elif n == 1:  # Base case
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)  # Recursive case
print (fibonacci(4))'''
'''import torch
import torch.nn as nn
import torch.optim as optim 
X = torch.tensor([[1],[2],[3]])
Y = torch.tensor([[1],[4],[9]])
class SimpleLinearRegression(nn.module):
    def __init__(self):
        super (SimpleLinearRegression,self).__init__()
        self.layer1 = nn.Linear(10,5)
        self.layer2 = nn.Linear(5,2)
        self.layer3 = nn.Linear(2,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.layer1)'''
'''import torch

# Step 1: Create a tensor with requires_grad=True
x = torch.tensor(2.0, requires_grad=True)

# Step 2: Define a function of x
# Let's use a simple function: y = x^3
y = x ** 3

# Step 3: Perform backpropagation
# Since y is a scalar, we can directly call backward() to compute the gradient of y w.r.t x
y.backward()

# Step 4: Get the gradient
print(f'Gradient of y with respect to x: {x.grad}')'''
'''for i in range(1,10):
    print(i)
x = [0,1,2,3,4]
print(len(x))
n = int(input())
for i in range(1,n-1):
    print()'''
'''x = "ankakak"
y = ''
for i in x :
    if x == ["a",'e','i','o','u']:
        x = y
print(y)
        '''
'''def matrixmultiplication(m1,m2):
    m1r = len(m1)
    m1c = len(m1[0])
    m2r = len(m2)
    m2c = len(m2[0])
    if m1r != m2c :
        return 'this is the invalid matric '
    result = [[0 for _ in range(m2c)]for _ in range(m1r) ]
    for i in range(m1r):
        for j in range(m2c):
            for k in range(m1c):
                result[i][j] += m1[i][k] * m2[k][j]
    return result
# Example usage:
m1 = [
    [1, 2],
    [3, 4],
    [5, 6]
]

m2 = [
    [7, 8, 9],
    [10, 11, 12]
]

result = matrixmultiplication(m1, m2)

for row in result:
    print(row)'''
'''import torch
from torch.utils.data import random_split, TensorDataset, DataLoader

features = torch.randn(100, 10)
labels = (torch.randn(100) > 0).float()

# Create a dataset from the tensors
dataset = TensorDataset(features, labels)
print(dataset)'''
'''import torch
from torch.utils.data import dataset , random_split ,TensorDataset , DataLoader
features = torch.randn(100,10)
labels = torch.randn(100,10).float()
dataset = TensorDataset(features,labels)
print(dataset)'''
'''n = 4
for i in range(4):
    print(i)'''
'''n = 5 
for i in range(1,n+1):
    print(i*"*")'''
'''n = 5
for i in range(1, n + 1):
    print(' ' * (n - i)+ '*' * (2 * i - 1))
''' 
'''
n = 4
for i in range(n,0,-1):
    print(" "*(n-i)+ "*" * (2*i-1))'''
'''import torch
x = torch.tensor([[1,2],[1,2]])
print(x)
zeros = torch.zeros((2, 3))  # 2x3 matrix of zeros
ones = torch.ones((2, 3))    # 2x3 matrix of ones
print(zeros)
print(ones)'''
'''import torch
import torch.nn as nn 
import torch.optim as optim
torch.manual_seed(1000)
features = torch.randn(1000,10)
weights = torch.rand(10,1)
labels = torch.sigmoid(features@weights+torch.rand(1000,1)*0.01)

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.layer = nn.Linear(in_features= 10,out_features=1)
        self.layer1 = nn.Sigmoid()
    def forward(self,inputs):
        return self.layer1(self.layer(inputs))
model = LogisticRegression()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)
epochs = 1000
for epoch in range(epochs):
    prediction = model(features)
    loss = criterion(prediction,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch)%100 ==0:
        print(f"EPOCH{epoch}, LOSS :{loss.item():.4f}")
        '''
'''import torch
import torch.nn as nn 
import torch.optim as optim 
features = torch.randn(1000,10)
weights = torch.randn(10,1)
labels = torch.randint(0,2,(1000,1)).float()
#labels = torch.sigmoid(features@weights + torch.randn(1000,1)*0.01)
class LogisticRegressiontry(nn.Module):
    def __init__(self):
        super(LogisticRegressiontry,self).__init__()
        self.layer = nn.Linear(in_features=10,out_features=1)
        #self.layer1 = nn.Sigmoid()
    def forward(self,input):
        return self.layer(input)
model = LogisticRegressiontry()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.06)
epochs = 1000
for epoch in range(epochs):
    prediction = model(features)
    loss = criterion(prediction,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%100 == 0:
        print(f"epoch:[{(epoch+1)/epochs}], loss :{loss.item():.4f}")
with torch.no_grad():  # No gradient computation for testing.
    test_predictions = model(features[:5])  # Predict the first 5 samples.
    print("Sample Predictions:", test_predictions.flatten())  # Predicted values.
    print("Actual Labels:\n", labels[:5].flatten())'''
'''import torch
import torch.nn as nn 
import torch.optim as optim 
features = torch.randn(377,27)
weights = torch.randn(27,1)
bias  = torch.randn(377,1)
labels = features@weights +bias
class Linearr(nn.Module):
    def __init__(self):
        super (Linearr,self).__init__()
        self.layer = nn.Linear(in_features= 27,out_features= 1)
    def forward(self,input):
        return self.layer(input)
model = Linearr()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)


epochs = 1000
for epoch in range(epochs):
    prediction = model(features)
    loss = criterion(prediction,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%100 ==0:
        print(f"epoch {(epoch+1)/epochs}, loss = {loss.item():.4f}")
with torch.no_grad():
    test_prediction = prediction[:5]
    print("/nsample prediction", test_prediction)           
    print("sample labels", labels[:5])     '''
'''import torch
import torch.nn as nn
import torch.optim as optim

# Data generation and normalization
features = torch.randn(377, 27)
features = torch.nn.functional.normalize(features, dim=1)  # Normalize features
weights = torch.randn(27, 1)
bias = torch.randn(377, 1)
labels = features @ weights + bias

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(27, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer

# Training loop with early stopping and learning rate scheduling
epochs = 1000
patience = 10
best_loss = float('inf')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience/2)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(features)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Testing
with torch.no_grad():
    test_predictions = model(features[:5])
    print("\nSample Predictions:", test_predictions)
    print("Sample Labels:", labels[:5])'''
'''list = ["aa","aaa"]
x = [ "conatin_z"
     if 'y' in word 
     else word
     for word in list ]
print(x)'''
'''result = [[0 for _ in range(2)] for _ in range(3)]
print(result)'''
'''def matmul(m1,m2):
    m1r = len(m1)
    m1c = len(m1[0])
    m2r = len(m2)
    m2c = len(m2[0])
    if m1c != m2r :
        print(f"not a valid matrix")
        return None
    result = [[0 for _ in range(m2c)] for _ in range(m1r)]
    for i in range(m1r):
        for j in range(m2c):
            for k in range(m1c):
                result[i][j] += m1[i][k] * m2[k][j]
    return result
m1 = [
    [1, 2],
    [3, 4],
    [5, 6]
]

m2 = [
    [7, 8, 9],
    [10, 11, 12]
]

result = matmul(m1, m2)
print(result)

'''
'''def matmul(m1,m2):
    m1r = len(m1)
    m1c = len(m1[0])
    m2r = len(m2)
    m2c = len(m2[0])
    if m1r != m2c :
        print("not a valid matrix")
    result = [[0 for _ in range(m1r)]for _ in range(m2c)]
    for i in range(m1r):
        for j in range(m2c):
            for k in range(m2r):
                result[i][j] += m1[i][k] * m2[k][j]
    return result 
m1 = [
    [1, 2],
    [3, 4],
    [5, 6]
]

m2 = [
    [7, 8, 9],
    [10, 11, 12]
]

result = matmul(m1, m2)
print(result)'''


'''import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import hamming_loss, f1_score

# Example binary matrix
data = torch.tensor([
    [1, 1, 0, 0, 0,0,0,0,0,0,1,0,0,1],  # 2013
    [1, 0, 0, 0, 1,1,0,1,1,0,0,0,1,1],  #2014
    [0, 1, 0, 0, 0,1,1,1,1,0,0,1,1,1],  # 2015
    [1, 1, 1, 0, 0,0,1,0,0,0,1,0,0,1],#2016
    
    
], dtype=torch.float32)

# Split data into training and labels
X_train = data[:-1]  # Past years
y_train = data[1:]   # Next year's topics

# Define a simple feedforward neural network
class BinaryPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(BinaryPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),  # Hidden layer
            nn.ReLU(),
            nn.Linear(64, output_size),  # Output layer
            nn.Sigmoid()  # For binary outputs
        )
    
    def forward(self, x):
        return self.fc(x)

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
output_size = y_train.shape[1]
model = BinaryPredictor(input_size, output_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Predict next year's topics
with torch.no_grad():
    y_pred = model(X_train[-1].unsqueeze(0))
    y_pred_binary = (y_pred > 0.5).float()

# Evaluate
y_true = y_train[-1].numpy()
y_pred_numpy = y_pred_binary.numpy()[0]
print("Hamming Loss:", hamming_loss(y_true, y_pred_numpy))
print("F1 Score:", f1_score(y_true, y_pred_numpy, average='macro'))



# Print the raw probabilities
print("Raw Predictions (Probabilities):", y_pred.numpy())

# Print the binary predictions
print("Binary Predictions (0 or 1):", y_pred_binary.numpy())'''





'''lst = ['a','b','c']
for i in lst:
    print(i)
for j in range(len(lst)):
    print(j)'''
'''x= 14
y = 4
z = x/y
print(z)
e = x//y
print(e)
f = x%y
print(f)'''
'''x = [1,1,1,1,1,1,1]
for i in range(len(x)):
    print (i)'''


'''x= 8
y = 9

if x>y:
    
    print("hellow wolrd")
else:
    print("jio sher")
x = 10 
'''

'''x = [1,1,1,1,1]

z = ()
x = [1,2,3,4,44,44]
set = {}'''
a = "abcd"

'''print(a[0])
y = len(x)
print(y)
for i in range(len(x)):
    print(x)'''
    
    
'''a = "abcd"
for j in a :
    print(j)
x = len(a)
print(x)

'''


'''DICT = {"KEY": [1,2,3,3,0,0],
        "KEY2":[1,2,3,4,5,6,7,0,0],
        "KEY4":[1,2,3,4,5,6,7,8,0]}
import pandas as pd 
X = pd.DataFrame(DICT)
print(X)'''
'''x = "racecar"
y = x[::-1]
if x == y :
    print("its a plaindrome" )
else :
    print("not a plindrome")'''
'''import torch
import torch.nn as nn
import torch.optim as optim


features = torch.randn(1000,10)
weights = torch.randn(10,1)
labels = features@weights + torch.randn(1000,1)*0.01
'''


'''def threesum(nums):
    result = set()
    n = len(nums)
    for i in range(n):
        for j in range(i+1,n):
            for k in range (j+1,n):
                if nums[i] +nums[j] + nums[k] == 0:
                    triplet = tuple(sorted([nums[i],nums[j],nums[k]]))
                    result.add(triplet)
    return [ list(triplet) for triplet in result]'''
    
'''class Node:
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
ll.display()'''
'''import torch
import torch.nn as nn
import torch.optim as optim
features = torch.randn(1000, 10)  # 1000 samples, 10 features
weights = torch.randn(10, 1)  # 10 weights (one per feature)
labels = features @ weights + torch.randn(1000, 1)  # Linear combination + noise

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.layer = nn.Linear(10,1)
    def forward(self,input):
        return self.layer(input)
model = LinearRegression()
criterion = nn.MSELoss()
optmizer = optim.SGD(model.parameters(),lr = 0.01)
epochs = 1000
for epoch in range(epochs):
    prediction = model(features)
    loss = criterion(prediction,labels)
    optmizer.zero_grad()
    loss.backward()
    optmizer.step()
    if (epoch+1)%100 == 0:
        
        print(f"[EPOCH {(epoch+1)}/{epochs}],Loss : {loss.item():.4f}")'''
        
'''
epochs = 1000
for epoch in range(epochs):
    if (epoch+1)%100 == 0:
        print(f"[epoch{epoch+1}/{epochs}]")'''
'''epochs = 1000
for epoch in range(epochs):
    if (epoch+1)%100 ==0:
        print(f"[epoch:{epoch+1}/{epochs}]")'''
'''class Epochs():
    def __init__(self,epochs):
        epochs = 1000
        for epoch in range(epochs):
            if (epoch+1)%100 == 0:
                print(f"[epoch{epoch+1}/{epochs}]")'''
'''class Epochs:
    def __init__(self, epochs):
        self.epochs = epochs

    def run(self, interval=100):
        for epoch in range(self.epochs):
            if (epoch + 1) % interval == 0:
                print(f"[epoch {epoch + 1}/{self.epochs}]")

# Example usage
e = Epochs(1000)
e.run(interval=50)'''

'''class Students:
    def __init__(self,name,standard,age):
        self.name = name
        self.age = age
        
        self.standard = standard
    def greet(self):
        print(f"my name is {self.name} and i am from {self.standard} and i am {self.age} year old")
person = Students("xyxyx","12th",12)
print(person.name)
person.greet()

# OR

Students.greet(person)'''
'''
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

'''class Book:
    def __init__(self,title,author,year,is_available):
        self.title = title
        self.author = author
        self.year = year
        self.is_available = is_available
    def borrow(self):
        print(f"the book {self.title}{ self.author} is availabe")'''
'''class Student:
    def __init__(self,name):
        self.name = name
    def greet(self):
        print(f"the neme is {self.name}")
class Class(Student):
    def __init__(self,name,age):
        super.__init__(name)
        self.name = name
        self.age = age
    def dhreet (self):
        print(f"this is how iyt works ")'''
'''class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):  # Inheriting from Animal
    def bark(self):
        print("Dog barks")

dog = Dog()
dog.speak()  # Inherited method
dog.bark()   # Dog's own method'''
'''class Animal:
    def speak(self):
        print("Animal speaks")

class Canine:
    def hunt(self):
        print("Canine hunts")

class Dog(Animal, Canine):  # Inheriting from both Animal and Canine
    def bark(self):
        print("Dog barks")

dog = Dog()
dog.speak()  # Inherited from Animal
dog.hunt()   # Inherited from Canine
dog.bark()   # Dog's own method
'''
'''class MatrixMultiplication:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
    
    def multiply(self):
        m1r, m1c = len(self.m1), len(self.m1[0])
        m2r, m2c = len(self.m2), len(self.m2[0])
        
        # Check if multiplication is possible
        if m1c != m2r:
            print("Matrix multiplication is not possible because the number of columns in m1 is not equal to the number of rows in m2.")
            return None
        
        # Initialize the result matrix with zeros
        result = [[0 for _ in range(m2c)] for _ in range(m1r)]
        
        # Perform matrix multiplication
        for i in range(m1r):
            for j in range(m2c):
                for k in range(m1c):
                    result[i][j] += self.m1[i][k] * self.m2[k][j]
        
        return result

# Example Usage
m1 = [[1, 2, 3], [4, 5, 6]]
m2 = [[7, 8], [9, 10], [11, 12]]

matrix_obj = MatrixMultiplication(m1, m2)
result = matrix_obj.multiply()

if result:
    for row in result:
        print(row)
'''
'''x= [1,1,1,1]
def sliding_window(nums):
    nums.sort()  # Sorting the array for two-pointer approach
    left = 0
    right = len(nums) - 1
    result = []
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if current_sum == 0:  # Found a pair that sums to zero
            result.append([nums[left], nums[right]])
            left += 1
            right -= 1
        elif current_sum < 0:
            left += 1  # Increase sum by moving the left pointer
        else:
            right -= 1  # Decrease sum by moving the right pointer
    
    return result

# Example Usage
nums = [1, 1, 1, -1, -1]
pairs = sliding_window(nums)
print("Pairs that sum to zero:", pairs)
'''
'''class Student:
    def __init__(self,name,age):
        self.name = name
        self.age = age 
    def greet (self):
        print(f"the person named {self.name} is the master of {self.age}")
x = Student("arun",23)
Student.greet(x)
x .greet()'''
'''
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim 
features = torch.randn(1000,10)
weights = torch.randn(10,1)
labels = features@weights +torch.randn(1000,1)*0.01
class Linearregression(nn.Module):
    def __init__(self):
        super(Linearregression,self).__init__()
        self.layer1 = nn.Linear(in_features=10,out_features=1)
    def forward(self,input):
        return self.layer1(input)
model = Linearregression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)
epochs = 1000
for epoch in range(epochs):
    prediction = model(features)
    loss = criterion(prediction,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%100 == 0:
        print(f"[epoch : {epoch+1}/{epochs}], loss : {loss.item()}")
with torch.no_grad():
    predictions = model(features)  # Predict values for the entire dataset.
    test_predictions = model(features[:5])  # Predict the first 5 samples.
    print("\nSample Predictions:\n", test_predictions.flatten())  # Predicted values.
    print("Actual Labels:\n", labels[:5].flatten())  # Actual labels.
'''
'''def longestValidParentheses(s: str) -> int:
    left = 0
    right = 0
    max_length = 0
    for char in s :
        if char == "(":
            left +=1
        elif char ==")":
            right +=1
        elif right == left :
            max_length = max(max_length,2*right)
        elif right > left :
            right = 0,left =0'''
'''lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
del lst[::3]  # Removes every 3rd element
print(lst)  # Output: [2, 3, 5, 6, 8, 9]
lky = [ 2,4,6,8,10]
print(lky[::3])'''

'''x = "heloow wolrd "
y =x.split(" ")
print(y)
right= len(y)-1
for i in range(right):
    if len(y[right]) == 0:
        right -=1
    else :
        print(len(y[-1])) '''
'''class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        x=s.split(" ")
        for i in x :
            if len(x[i]) == 0:
                return len(x[-2])
            else :
                return (len(x[-1]))
'''



'''def wordPattern(self, pattern: str, s: str) -> bool:'''
'''s = 'abba'
for i in enumerate(s,start=1):
    print(i)
pattern = "cat dog dog cat"
for j in enumerate(pattern.split(),start=1):
    print(j)
'''
'''s = [1,2,3,4,54,56,7]
s += s[::-3]
print(s)'''
'''import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate random data for features and labels
features = torch.randn(1000, 9)
weights = torch.randn(9, 1)
labels = features @ weights + torch.randn(1000, 1)

# Define the Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.layer1 = nn.Linear(9, 1)

    def forward(self, input):
        return self.layer1(input)

# Initialize the model, loss function, and optimizer
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()  # Ensure the model is in training mode
    prediction = model(features)
    loss = criterion(prediction, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()'''
'''my_string = "hellol"
x = my_string.count(char)
print(x)
char_count = {}
for char in my_string:
    char_count[char] = char_count.get(char, 0) + 1  # Default value for new keys is 0
print(char_count)  # Output: {'h': 1, 'e': 1, 'l': 2, 'o': 1}'''

    
'''    
x = [num for num in range(1,1001) if (num)/3 == 0 ]
print(x)'''
'''
list = {'a':'1','b':'21'}
x = { lambda key:int(value)+1 for key, value in list.items()  }
print(x)'''

'''arr =[1,2,1,2,4]
k = 2
left = 0
count ={}
for right in range(len(arr)):
    count[arr[right]]= count.get(arr[right],0)+1

while len(count) > k:
            count[arr[left]] -= 1
            if count[arr[left]] == 0:
                del count[arr[left]]
            left += 1
print(count)
def longest_subarray_with_k_distinct(arr, k):
    count= {}
    left = 0
    max_length =0
    for right in range(len(arr)):
        count[arr[right]]= count.get(arr[right],0)+1'''
        
'''def variable_size_sliding_window(arr, k):
    left = 0
    count = {}
    for right in range(len(arr)):
        # Expand the window by including the element at 'right'
        count[arr[right]] = count.get(arr[right], 0) + 1
        
        # Shrink the window if the number of distinct elements exceeds 'k'
        while len(count) > k:
            count[arr[left]] -= 1
            if count[arr[left]] == 0:
                del count[arr[left]]  # Remove the element completely when count reaches zero
            left += 1
            
    return count

# Test case
arr = [1, 2, 1, 2, 4]
k = 2
print(variable_size_sliding_window(arr, k))

        '''
'''lst = [1,2,3,4,5]
lst.append((1,2,3))
print(lst)
lst.append
lst.extend
lst.insert
lst.pop
lst.'''
'''my_list = [10, 20, 30, 20]
my_list.remove(20)'''

'''import torch
import torch.nn as nn
import torch.optim as optim 
features = torch.randn(1000,10)
weights = torch.randn(10,1)
labels = features@weights +torch.randn(1000,1)*0.1
class Linearregression(nn.Module):
    def __init__(self):
        super(Linearregression,self).__init__()
        self.layer = nn.Linear(in_features=10,out_features=1)
    def forward(self,input):
        return self.layer(input)
model = Linearregression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)
EPOCHS = 1000    
for epoch in range(EPOCHS):
    prediction = model(features)
    loss = criterion(prediction,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%100 == 0:
        
        print(f"[Epoch {epoch+1}/{EPOCHS}, LOSS : {loss.item():.4f}]")
with torch.no_grad():
    test_predictions = model(features[:5])  # Predict the first 5 samples.
    print("\nSample Predictions:\n", test_predictions.flatten())  # Predicted values.
    print("Actual Labels:\n",labels[:5].flatten())
with torch.no_grad():
    prediction = model(features[:5])
    print("\n Sample prediction:\n", prediction.flatten())
    print("actual labels\n",labels[:5].flatten())
''''''x= [0,1,0,2,3,4,12]
non_zero = 0
for i in range(len(x)):
    if x[i] != 0:
        x[non_zero] = x[i]
        non_zero+=1
for i in range(non_zero,len(x)):
    x[i] =0
print(x)''''''

with torch.no_grad():
    prediction = model(features[:5])
    print(prediction.flatten())
    print(labels[:5].flatten)
with torch.no_grad():
    prediction = model(features[:5])
    print(f"SAMPLE prediction {prediction.flatten()}")
    print(f"Actual labels{labels[:5].flatten()}")
fearutures = torch.randn(1000,10)'''
'''import torch
import torch.nn as nn
import torch.optim as optim 
features = torch.randn(1000,9)
weights = torch.randn(9,1)
labels = features@weights+torch.randn(1000,1)*0.01
class Linear_regression(nn.Module):
    def __init__(self):
        super(Linear_regression,self).__init__()
        self.layer = nn.Linear(in_features=9,out_features=1)
    def forward(self,input):
        return self.layer(input)
MODEL= Linear_regression()
CRITERION = nn.MSELoss()
OPTIMIZER = optim.SGD(MODEL.parameters(),lr = 0.01)
EPOCH = 1000
for epoch in range(EPOCH):
    PREDICTION = MODEL(features)
    LOSS = CRITERION(PREDICTION,labels)
    OPTIMIZER.zero_grad()
    LOSS.backward()
    OPTIMIZER.step()
    if (epoch+1)%100 == 0:
        print(f"[EPOCH : {epoch+1}/{EPOCH},Loss : {LOSS.item():.10f}]")
with torch.no_grad():
    test_prediction = MODEL(features[:5])
    print(f"sample prediction are {test_prediction.flatten()}")
    print(f"actual labels are {labels[:5].flatten()}") 
    absolute_error = torch.abs(labels - test_prediction)
    percentage_error = (absolute_error / torch.abs(labels)) * 100
    mape = torch.mean(percentage_error)
    accuracy = 100 - mape.item()
    print(f"\nAccuracy: {accuracy:.2f}%")'''

'''import torch
import torch.nn as nn
import torch.optim as optim 

# Generating data
features = torch.randn(1000, 9)
weights = torch.randn(9, 1)
labels = features @ weights + torch.randn(1000, 1) * 0.01

# Linear Regression Model
class Linear_regression(nn.Module):
    def __init__(self):
        super(Linear_regression, self).__init__()
        self.layer = nn.Linear(in_features=9, out_features=1)

    def forward(self, input):
        return self.layer(input)

# Model, Loss, and Optimizer
MODEL = Linear_regression()
CRITERION = nn.MSELoss()
OPTIMIZER = optim.SGD(MODEL.parameters(), lr=0.01)
EPOCH = 10000

# Training Loop
for epoch in range(EPOCH):
    PREDICTION = MODEL(features)
    LOSS = CRITERION(PREDICTION, labels)
    OPTIMIZER.zero_grad()
    LOSS.backward()
    OPTIMIZER.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"[EPOCH: {epoch+1}/{EPOCH}, Loss: {LOSS.item()}]")

# Testing and Accuracy Calculation
with torch.no_grad():
    test_prediction = MODEL(features)
    
    # Calculate MAPE for accuracy
    absolute_error = torch.abs(labels - test_prediction)
    percentage_error = (absolute_error / torch.abs(labels)) * 100
    mape = torch.mean(percentage_error)
    accuracy = 100 - mape.item()

    # Display Results
    print("\nSample Predictions:")
    print(f"Predicted: {test_prediction[:5].flatten()}")
    print(f"Actual: {labels[:5].flatten()}")
    print(f"\nAccuracy: {accuracy:.2f}%")'''
'''import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generating synthetic data
features = torch.randn(1000, 9)
weights = torch.randn(9, 1)
labels = features @ weights + torch.randn(1000, 1) * 0.01

# Split data into training and validation sets
train_features = features[:800]
train_labels = labels[:800]
val_features = features[800:]
val_labels = labels[800:]

# Linear Regression Model
class Linear_regression(nn.Module):
    def __init__(self):
        super(Linear_regression, self).__init__()
        self.layer = nn.Linear(in_features=9, out_features=1)

    def forward(self, input):
        return self.layer(input)

# Model, Loss, and Optimizer
MODEL = Linear_regression()
CRITERION = nn.MSELoss()
OPTIMIZER = optim.SGD(MODEL.parameters(), lr=0.01)
EPOCH = 1000

# Lists to store training and validation losses
train_losses = []
val_losses = []

# Training Loop
for epoch in range(EPOCH):
    # Training phase
    MODEL.train()
    train_prediction = MODEL(train_features)
    train_loss = CRITERION(train_prediction, train_labels)
    OPTIMIZER.zero_grad()
    train_loss.backward()
    OPTIMIZER.step()

    # Validation phase
    MODEL.eval()
    with torch.no_grad():
        val_prediction = MODEL(val_features)
        val_loss = CRITERION(val_prediction, val_labels)
    
    # Store losses
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"[EPOCH: {epoch+1}/{EPOCH}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}]")

# Plotting the Bias-Variance Tradeoff Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCH + 1), train_losses, label="Training Loss")
plt.plot(range(1, EPOCH + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Bias-Variance Tradeoff Curve")
plt.legend()
plt.grid(True)
plt.show()
'''

'''class Node :
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self,head):
        self.head = None

        
    def append():
        
    def display():
    '''
'''class Node :
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList :
    def __init__(self,head):
        self.head = None
        '''
'''class Node:
    def __init__(self,data):
        self.data = data
        self,next = None
class LinkedList:
    def __init__(self,head):
        self.head = None
    def append(self):'''
'''class Node :
    def __init__(self,data):
        self.data = data 
        self.next = None
class LinkedList :
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
    def display(self):
        current = self.head
        while current:
            print(current.data, end ="->")
            current = current.next
        print("None")
ll = LinkedList()
ll.append(10)
ll.append(20)
ll.append(30)
ll.display()       
            '''
'''class Node :
    def __init__(self,data):
        self.data = data 
        self.next = None
class LinkedList:
    def __init__(self):
        self.head =None
    def append(self,data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return 
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    def display(self):
        current = self.head
        while current:
            print(current.data,end=" -> ")
            current = current.next
        print(None)
                            '''  
'''import torch
import torch.nn as nn
import torch.optim as optim 
features = torch.randn(1000,10)
weights = torch.randn(10,1)
labels = features @ weights + torch.randn(1000,1)


class LinearRegression :
    def _init__(self,)'''

'''def recur(x):
    return recur(x)*recur(x)
x =100
print(recur(x))'''
'''def recur(n: str):
    for i in range(len(n)):
        if n[i] == 0 :
            return n[1]
    return n[-1] + recur(n[-1]) 
print(recur("hellow"))'''
'''class Node :
    def __init__(self,data):
        self.data = data 
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,data):
        new_node = Node(data)
        if not self.head :
            
            self.head = new_node
            return
        current = self.head
        while current.next :
            current = current.next
        current.next = new_node
    def display(self):
        current = self.head
        while current :
            print(current.data, end="->")
            current = current.next
        print("None")
ll = LinkedList()
ll.append(10)
ll.append(20)
ll.append(30)
ll.display()'''
'''import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------- Step 1: Generate Synthetic Data --------------------
# Create synthetic data for regression (continuous labels).
features = torch.randn(1000, 10)  # Random feature matrix (1000 samples, 10 features).
weights = torch.randn(10, 1)      # True weights (10 features mapped to 1 output).
bias = torch.rand(1)              # Random bias.
labels = features @ weights + bias + torch.randn(1000, 1) * 0.1  # Add some noise


class Linearreg(nn.Module):
    def __init__(self):
        super(Linearreg,self).__init__()
        self.layer = nn.Linear(in_features=10,out_features=1)
    def forward (self,input):
        return self.layer(input)
model = Linearreg()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr= 0.01)
epochS = 1000
for epoch in range(epochS):
    prediction = model(features)
    loss = criterion(prediction ,labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%100 == 0:
        print(f"[EPOCH {epoch+1}/{epochS},loss : {loss.item():.4f}]")
with torch.no_grad():
    test_prediction = model(features[:5])
    print('sample_prediction',test_prediction.flatten())
    print("actual_labels",labels[:5].flatten())'''
'''count = 1

while count <5 and count:  # The loop runs as long as count is less than or equal to 5
    print("Count:", count)
    count += 1  # Increment count by 1

print("Loop finished!")
'''



'''class Node :
    def __init__(self,data):
        self.data = data 
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append (self,data):
        new_node = Node (data)
        if not self.head :
            self.head = new_node
            return
        '''
'''def recur(s:str):
    if len(s) ==0:
        return s
    else :
        return recur(s[1:]) + s[0]

  

print(recur("hellwo"))'''
def feboinnaci(n):
    if n == 0:
        return 0
    else :
        feboinnaci(n +n[n-1]) 
print(feboinnaci(6))