'''x =[1,2,3]
y = [4,5,6]
nested_list = [ [x for x in range(1,4)] for y in range(4,6)]
print(nested_list)'''
'''dict = {'key':[1,2,3],'hello':[1,2,3]}
dict1 ={}
for _ in range(int(input())):
    name = input()
    number = int(input())
    dict1.append([name,number])
print(dict1)'''
'''a=["apple", "banana", "cherry", "kiwi", "mango"]
new =[x for x in a if 'a' in x]
print(new)'''
'''
Problem:
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and 
nums[i] + nums[j] + nums[k] == 0.
Notice that the solution set must not contain duplicate triplets.'''
'''arr ='sidds'
def palindrome(arr):
    left = 0
    right = len(arr)-1
    while left<right:
        if arr[left] != arr[right]:
            return False
        left +=1
        right -=1
        
        return True
x= palindrome(arr)
print(x)'''
'''arr = [1, 2, 3, 4, 6, 8]
def sor(arr):
    left =0
    right = len(arr)-1
    while left<right:
        if arr[left]+arr[right]!=10:
            
            return (arr[left],arr[right])
        elif arr[left] + arr[right] < 10:
            left += 1
        
        # If the sum is greater than 10, move the right pointer to the left
        else:
            right -= 1
x = sor(arr)
print(x) '''

'''
Given an array of non-negative integers where each element represents the height of a
vertical line on the x-axis, use the two-pointer technique to find two lines that together
with the x-axis form a container that holds the most water.

Example: heights = [1,8,6,2,5,4,8,3,7]
'''
'''def sor(arr):
    arr = arr.sort()
    left =0
    right = len(arr)-1
    while left<right:
        return arr[right-1]+arr[right]
arr =[1,8,6,2,5,4,8,3,7]
print(sor(arr))

        '''''''''
        
        
'''def longets_substring(s):
    def two_pointers(left,right):
        while left >=0 and right <len(s) and s[left]== s[right]:
            left -=1
            right +=1
        return s[left+1,right]
    longest = ""
    for i in range(len(s)):
        odd_palindrom = longets_substring(i,i)
        even_palindrome = longets_substring(i-1,i)
    
        maximum_longest = max(longest ,odd_palindrom,even_palindrome,key = len)
    return longest'''
    
'''def expanding_window_algo(s):
    def setting_two_pointers(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        # Return the valid palindrome substring
        return s[left+1:right]

    # We can call the setting_two_pointers for testing an expanding window
    return setting_two_pointers(0, len(s) - 1)

# Example input
s = 'abbbbbba'
print(expanding_window_algo(s))'''
'''def binary_search(arr,target):
    def two_pointer(left,right):
        left =0
        right = len(arr)-1
        mid_point = len(arr)%2
        while left <= right:
            if target > arr[mid_point]:
                right -=1
            elif target < arr[mid_point]:
                left += 1
            else :
                return arr[left]
                '''
'''def smallest_subarray_with_given_sum(S, nums):
    # Initialize variables
    window_sum = 0
    min_length = float('inf')  # Start with a large number
    left = 0

    # Traverse the array with the right pointer
    for right in range(len(nums)):
        window_sum += nums[right]  # Add the current element to the window sum

        # While the window sum is at least S, try to find the smallest window
        while window_sum >= S:
            # Update the minimum length of the subarray
            min_length = min(min_length, right - left + 1)
            # Shrink the window from the left
            window_sum -= nums[left]
            left += 1

    # If no valid window was found, return 0; otherwise, return the minimum length
    return min_length if min_length != float('inf') else 0

# Example usage
nums = [4, 2, 2, 7, 1, 2, 3, 6]
S = 8
print(smallest_subarray_with_given_sum(S, nums))  # Output: 1

def smallest_subarray(arr,s):
    left =0
    minlength = float('inf')
    window_sum = 0
    for right in range (len(arr)):
        window_sum += right[arr]
        while s>= window_sum:
            min = min(minlength,right-left+1)
            window_sum -= arr[left]
    return minlength if minlength != float('inf') else 0'''


'''def smallest (arr,sum):
    left =0
    right = 0 
    ws  = 0
    ml =float('inf')
    for right in range(len(arr)):
        ws += right[arr]
        while '''
'''def matrix_multiplication(m1,m2):
    m1r = len(m1)
    m1c = len(m1[0])
    m2r = len(m2)
    m2c = len(m2[0])
    if m1c != m2r :
        return f"not valid"
    result =[[0 for _ in range(m1c)]for _ in range(m1r)]
    for i in range(m1r):
        for j in range(m1c):
            for k in range(m2r):
                result[i][j]+= m1[i][k]*m2[k][j]
    return result
m1 =[[2,2],[2,2]]
m2 = [[2,2],[2,2]]

print(matrix_multiplication(m1,m2))
'''

'''def matrix_transpose (m3):
    x = [[m3[j][i]for j in range(len(m3))]for i in range(len(m3[0]))]'''
'''def palindrom(s):
    left =0
    right = len(s)-1
    while left <right :
        if s[left]==s[right]:
            left +=1
            right -=1
        else :
            return f"notpalindrome"
    return f"palindrome"'''
'''def reverse_string(s):
    x = s[::-1]
    return x
s ='aasss'
print(reverse_string(s))
'''
'''def  variable_sliding_window(arr,k):
    window_sum =0
    left =0
    min_length  = float('inf')
    for right in range(len(arr)):
        window_sum += arr[right] # move right 
        while window_sum >= k:
            min = min(min_length,right-left+1)
            window_sum -= arr[left] #shrink
            left +=1 # move towrds right
    return min_length if min_length!= float('inf') else min_length'''
'''def matrix_multiplication(m1,m2):
    m1r = len(m1) 
    m1c = len(m1[0])
    m2r = len(m2)
    m2c = len(m2[0])
    result = [[0 for _ in range(len(m1c))]for _ in range(len(m2r))]
    for i in range(m1r):
        for j in range(m2c):
            for k in range(m2r):
                result[i][j] += m1[i][k]*m2[k][j]
    return result
    '''
'''arr = ['array','area','arbiratory','ar','arar']
def longest_prefix(arr):
    prefix  = arr[0]
    for item in arr[1:]:
        while item[:len(prefix)]!= prefix and prefix :
            prefix =prefix[:-1]
    return prefix
            
print(longest_prefix(arr))'''
'''def longest_prefix(arr):
    left = 0
    right = len(arr)-1
    while left< right :
        if arr[left] == arr[right]:
            left +=1
            right -=1
        elif arr[left]!= arr[right]:
            break
        return f"is the right"
arr = "toot"
print(longest_prefix(arr))'''
'''import torch
import torch.nn as nn
import torch.optim as optim

# Create synthetic data for linear regression
features = torch.randn(1000, 10)  # 1000 samples, 10 features
weights = torch.randn(10, 1)      # Weights for the true linear relationship
labels = features @ weights + torch.rand(1000, 1) * 0.01  # True labels with some noise

# Define the Linear Regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.layer = nn.Linear(in_features=10, out_features=1)  # 10 input features, 1 output
    
    def forward(self, x):
        return self.layer(x)  # Perform a linear transformation

# Initialize the model
model = LinearRegression()

# Define the loss function (Mean Squared Error) and optimizer (Stochastic Gradient Descent)
criterion = nn.MSELoss()  # MSE Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # SGD optimizer with learning rate 0.01

# Training the model
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    prediction = model(features)  # Model prediction
    loss = criterion(prediction, labels)  # Calculate loss
    
    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Update weights
    
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Test the model
with torch.no_grad():
    test_predictions = model(features[:5])  # Predict for the first 5 samples
    print("\nSample Predictions:\n", test_predictions.flatten())
    print("Actual Labels:\n", labels[:5].flatten())

'''

'''import torch
import torch.nn as nn
import torch.optim as optim
features = torch.randn(1000,10)
weights = torch.randn(10,1)
labels = torch.randint(0,2,(1000,1)).float()



class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.layer= nn.Linear(in_features=10,out_features=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,inputs):
        return self.sigmoid(self.layer(inputs))

model = LogisticRegression()
countrier = nn.BCELoss()
optimzer = optim.SGD(model.parameters(),lr = 0.01)
epochs = 1000
for epoch in range(epochs):
    inputs = features
    prediction = model(features)
    loss = countrier(prediction,labels)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if (epoch+1)%100 ==0 :
        print(f" Epoch[{epoch}/{epochs}, loss :{loss.item():.4f}]")'''
'''
import torch
import torch.nn as nn 
import torch.optim as optim
features = torch.randn(1000,10)
weights = torch.randn(10,1)
labels = torch.randint(0,2,(1000,1)).float()

class LinearRegressionretry(nn.Module):
    def __init__(self):
        super(LinearRegressionretry,self).__init__()
        self.layer = nn.Linear(in_features=10,out_features=1)
        self.layerone= nn.Sigmoid()
    def forward (self,input):
        return self.layerone(self.layer())'''

'''def maxsum(arr,k):
    max_sum = sum(arr[:k])
    curr_sum = max_sum
    for i in range(k,len(arr)):
        curr_sum += arr[i]
        curr_sum -= arr[i-k]
        if curr_sum > max_sum:
           max_sum = curr_sum
    return max_sum   
arr = [1, 2, 3, 4, 5, 6, 7, 8]
k = 3
print(maxsum(arr, k)) '''
'''import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss, f1_score
from sklearn.model_selection import train_test_split

# Seed for reproducibility
np.random.seed(42)

# Generate data: 10 years (rows) and 10,000 columns (features)
years = list(range(2013, 2023))  # Years from 2013 to 2022
data = np.random.randint(0, 2, size=(len(years), 10000))  # Random binary data

# Convert data to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)

# Split into train and test sets (80% train, 20% test)
train_data, test_data = train_test_split(data_tensor, test_size=0.2, random_state=42, shuffle=False)
X_train = train_data[:-1]  # Past years (train set)
y_train = train_data[1:]   # Next year's topics (train labels)
X_test = test_data[:-1]    # Past years (test set)
y_test = test_data[1:]     # Next year's topics (test labels)

# Define a simple feedforward neural network
class BinaryPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(BinaryPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),  # Hidden layer
            nn.ReLU(),
            nn.Linear(16, output_size),  # Output layer
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

# Train the model and track loss
epochs = 100
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    losses.append(loss.item())  # Store loss
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Plot training loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model on the test set
with torch.no_grad():
    y_test_pred = model(X_test)  # Predict on test data
    y_test_pred_binary = (y_test_pred > 0.5).float()  # Convert probabilities to binary

# Evaluate metrics on test set
y_test_true = y_test.numpy()  # Ground truth
y_test_pred_numpy = y_test_pred_binary.numpy()  # Predicted binary values
print("Test Set Hamming Loss:", hamming_loss(y_test_true, y_test_pred_numpy))
print("Test Set F1 Score:", f1_score(y_test_true, y_test_pred_numpy, average='macro'))'''

'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Step 1: Generate data
years = list(range(2013, 2023))  # Years from 2013 to 2022
data = np.random.randint(0, 2, size=(len(years), 10000))  # Random binary data

# Convert data to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)

# Labels (example: sum of features per row)
labels = data_tensor.sum(dim=1)  # Summing across features (adjust for your task)

# Step 2: Reshape data for LSTM
# Add a time dimension to the data (sequence_length = 1 for simplicity)
data_tensor = data_tensor.unsqueeze(1)

# Step 3: Split data into training and testing sets
dataset = TensorDataset(data_tensor, labels)
train_size = int(0.8 * len(dataset))  # 80% training data
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# Step 4: Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # Final hidden state
        out = self.fc(h_n[-1])     # Fully connected layer
        return out

# Step 5: Hyperparameters
input_dim = 10000  # Number of features
hidden_dim = 64    # Number of hidden units in LSTM
output_dim = 1     # Output dimension
num_layers = 2     # Number of LSTM layers
learning_rate = 0.001
epochs = 10

# Step 6: Instantiate model, loss, and optimizer
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 7: Training Loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs.squeeze(), y_batch)  # Match dimensions
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Step 8: Testing
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():  # No gradients during evaluation
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            test_loss += loss.item()
    return test_loss / len(test_loader)

test_loss = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}")
'''
'''import numpy as np
import torch

# Step 1: Generate data
years = list(range(2013, 2023))  # Years from 2013 to 2022
data = np.random.randint(0, 2, size=(len(years), 10000))  # Random binary data

# Normalize data (features between 0 and 1)
data_tensor = torch.tensor(data, dtype=torch.float32) / 1.0  # Binary data is already 0 or 1

# Labels (sum of features per row, normalized)
labels = data_tensor.sum(dim=1)  # Summing across features
labels = labels / 10000.0        # Normalize labels to range 0 to 1

# Step 2: Reshape data for LSTM
data_tensor = data_tensor.unsqueeze(1)  # Add a time dimension
print(data_tensor)'''



'''import torch
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate years from 2014 to 2023
years = np.arange(2014, 2024)

# Number of topics
num_topics = 10000

# Convert the years into a tensor
years_tensor = torch.tensor(years)

# Generate random data for 10,000 topics
topic_data_tensor = torch.randint(0, 2, (len(years), num_topics))

# Combine the years and topic data into a single tensor
dataset_tensor = torch.cat((years_tensor.unsqueeze(1), topic_data_tensor), dim=1)

# Show the first few rows of the dataset
print(dataset_tensor[:5])'''


'''import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate years from 2014 to 2023 (10 years)
years = np.arange(2014, 2024)

# Number of topics (features)
num_topics = 10000

# Generate random topic data (0 or 1)
topic_data = np.random.randint(0, 2, (len(years), num_topics))

# Convert years to tensor for labels
years_tensor = torch.tensor(years - 2014)  # Encoding years as 0 to 9

# Convert topic data to tensor for input features
topic_data_tensor = torch.tensor(topic_data, dtype=torch.float32)

# Combine the years and topic data into a single dataset (for the features and target)
dataset_tensor = torch.cat((topic_data_tensor, years_tensor.unsqueeze(1)), dim=1)

# Split into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(dataset_tensor, test_size=0.2, shuffle=True, random_state=42)

# Separate features and labels for train and test data
X_train = train_data[:, :-1]  # All columns except the last (which is the label)
y_train = train_data[:, -1]   # Only the last column (the label)
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Convert to torch tensors
X_train = X_train.float()
y_train = y_train.long()  # Labels are categorical (years encoded as integers)
X_test = X_test.float()
y_test = y_test.long()

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Initialize the model
input_size = num_topics  # Number of topics
hidden_size = 128       # Number of LSTM units
output_size = 10        # Number of years (0 to 9)
model = LSTMModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Reshape data for LSTM (add a sequence dimension)
X_train = X_train.unsqueeze(1)  # Add a dummy sequence dimension (batch_size, seq_len, input_size)
X_test = X_test.unsqueeze(1)

# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    _, predicted_classes = torch.max(y_pred, 1)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
'''


'''import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate years from 2014 to 2023 (10 years)
years = np.arange(2014, 2024)

# Number of topics (features)
num_topics = 10000

# Generate random topic data (0 or 1)
topic_data = np.random.randint(0, 2, (len(years), num_topics))

# Convert years to tensor for labels
years_tensor = torch.tensor(years - 2014)  # Encoding years as 0 to 9

# Convert topic data to tensor for input features
topic_data_tensor = torch.tensor(topic_data, dtype=torch.float32)

# Combine the years and topic data into a single dataset (for the features and target)
dataset_tensor = torch.cat((topic_data_tensor, years_tensor.unsqueeze(1)), dim=1)

# Split into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(dataset_tensor, test_size=0.2, shuffle=True, random_state=42)

# Separate features and labels for train and test data
X_train = train_data[:, :-1]  # All columns except the last (which is the label)
y_train = train_data[:, -1]   # Only the last column (the label)
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Convert to torch tensors
X_train = X_train.float()
y_train = y_train.long()  # Labels are categorical (years encoded as integers)
X_test = X_test.float()
y_test = y_test.long()

# Define the Linear Model (instead of LSTM for this classification task)
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model
input_size = num_topics  # Number of topics
hidden_size = 128       # Number of hidden units
output_size = 10        # Number of years (0 to 9)
model = SimpleModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    _, predicted_classes = torch.max(y_pred, 1)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")'''
    
    
    
'''import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate years from 2014 to 2023 (10 years)
years = np.arange(2014, 2024)

# Number of topics (features)
num_topics = 1000

# Generate random topic data (0 or 1)
topic_data = np.random.randint(0, 2, (len(years), num_topics))

# Convert years to tensor for labels
years_tensor = torch.tensor(years - 2014)  # Encoding years as 0 to 9

# Convert topic data to tensor for input features
topic_data_tensor = torch.tensor(topic_data, dtype=torch.float32)

# Combine the years and topic data into a single dataset (for the features and target)
dataset_tensor = torch.cat((topic_data_tensor, years_tensor.unsqueeze(1)), dim=1)

# Split into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(dataset_tensor, test_size=0.2, shuffle=True, random_state=42)

# Separate features and labels for train and test data
X_train = train_data[:, :-1]  # All columns except the last (which is the label)
y_train = train_data[:, -1]   # Only the last column (the label)
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Convert to torch tensors
X_train = X_train.float()
y_train = y_train.long()  # Labels are categorical (years encoded as integers)
X_test = X_test.float()
y_test = y_test.long()

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)  # hn[-1] for the final hidden state
        out = self.fc(hn[-1])  # Pass the last hidden state to the fully connected layer
        return out

# Initialize the model
input_size = num_topics  # Number of topics
hidden_size = 128       # Number of LSTM units
output_size = 10        # Number of years (0 to 9)
model = LSTMModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Reshape data for LSTM (add a sequence dimension)
X_train = X_train.unsqueeze(1)  # Add a dummy sequence dimension (batch_size, seq_len, input_size)
X_test = X_test.unsqueeze(1)

# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    _, predicted_classes = torch.max(y_pred, 1)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

'''



'''import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate years from 2014 to 2023
years = np.arange(2014, 2024)

# Number of topics
num_topics = 10000

# Convert the years into a tensor
years_tensor = torch.tensor(years)

# Generate random data for 10,000 topics
topic_data_tensor = torch.randint(0, 2, (len(years), num_topics))

# Combine the years and topic data into a single tensor
dataset_tensor = torch.cat((years_tensor.unsqueeze(1), topic_data_tensor), dim=1)

# Show the first few rows of the dataset
print(dataset_tensor[:5])

# Define Autoencoder Architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()  # Sigmoid for binary data (0 or 1)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Input and output dimensions
input_dim = dataset_tensor.shape[1]  # Number of features in dataset (years + topics)

# Initialize Autoencoder Model with a larger latent dimension
latent_dim = 128  # Increase the latent dimension
model = Autoencoder(input_dim, latent_dim)

# Define the optimizer with a lower learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Split the dataset into train and test sets
train_data, test_data = train_test_split(dataset_tensor, test_size=0.2, random_state=42)

# Convert to tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)

# Define loss function (Mean Squared Error Loss)
criterion = nn.MSELoss()

# Training the model (you can increase num_epochs here)
num_epochs = 100  # Increase the number of epochs for better convergence
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    reconstructed = model(train_data)
    
    # Compute the loss (Mean Squared Error)
    loss = criterion(reconstructed, train_data)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test data
model.eval()
with torch.no_grad():
    test_reconstructed = model(test_data)
    test_loss = criterion(test_reconstructed, test_data)
    print(f'Test Loss: {test_loss.item():.4f}')



'''


'''import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate years from 2014 to 2023
years = np.arange(2014, 2024)

# Number of topics
num_topics = 10000

# Convert the years into a tensor
years_tensor = torch.tensor(years)

# Generate random data for 10,000 topics
topic_data_tensor = torch.randint(0, 2, (len(years), num_topics))

# Combine the years and topic data into a single tensor
dataset_tensor = torch.cat((years_tensor.unsqueeze(1), topic_data_tensor), dim=1)

# Show the first few rows of the dataset
print(dataset_tensor[:5])

# Define CNN-LSTM Architecture
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, num_filters, latent_dim, lstm_hidden_dim, output_dim):
        super(CNNLSTM, self).__init__()
        
        # 1D Convolutional Layer
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1)
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=lstm_hidden_dim, batch_first=True)
        
        # Fully Connected Layer to output predicted topics
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
    
    def forward(self, x):
        # Apply 1D convolution (expecting input shape: [batch_size, input_dim, seq_len])
        x = self.conv1d(x)
        x = torch.relu(x)
        
        # Reshape the data for LSTM: [batch_size, seq_len, num_filters]
        x = x.permute(0, 2, 1)
        
        # LSTM Layer
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use the output from the last timestep
        output = self.fc(hn[-1])
        
        return output


# Reshape dataset into [batch_size, seq_len, input_dim]
input_dim = dataset_tensor.shape[1] - 1  # Subtract the year column
dataset_tensor = dataset_tensor.float()

# Define train-test split
train_data, test_data = train_test_split(dataset_tensor, test_size=0.2, random_state=42)

# Convert to tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)

# Define model hyperparameters
num_filters = 64  # Number of filters for CNN layer
latent_dim = 128  # Latent dimension
lstm_hidden_dim = 256  # Hidden dimension for LSTM layer
output_dim = num_topics  # Same as the number of topics

# Initialize the CNN-LSTM model
model = CNNLSTM(input_dim=1, num_filters=num_filters, latent_dim=latent_dim, lstm_hidden_dim=lstm_hidden_dim, output_dim=output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Reshape train and test data to [batch_size, input_dim, seq_len]
train_data = train_data[:, 1:].unsqueeze(1)  # Adding extra dimension for channels (input_dim, which is 1 here)
test_data = test_data[:, 1:].unsqueeze(1)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(train_data)
    
    # Compute the loss (compare predicted to actual topic data for each year)
    loss = criterion(output, train_data.squeeze(1))  # Remove the extra dimension for comparison
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test data
model.eval()
with torch.no_grad():
    test_output = model(test_data)
    test_loss = criterion(test_output, test_data.squeeze(1))  # Use all features except the year
    print(f'Test Loss: {test_loss.item():.4f}')

# Predict the topics for the year 2024 using the trained model
with torch.no_grad():
    predicted_topics = model(test_data)
    print(f'Predicted topic data for the year 2024: {predicted_topics[0]}')







import matplotlib.pyplot as plt

# Track loss for visualization
train_losses = []
test_losses = []

# Training the model with visualization
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(train_data)
    
    # Compute the loss
    loss = criterion(output, train_data.squeeze(1))
    train_losses.append(loss.item())
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        test_output = model(test_data)
        test_loss = criterion(test_output, test_data.squeeze(1))
        test_losses.append(test_loss.item())
    
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Plot the training and test loss
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), train_losses, label="Train Loss")
plt.plot(range(num_epochs), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Test Loss Over Epochs")
plt.legend()
plt.show()

# Visualize predictions vs actual for the test set
predicted_topics = test_output.numpy()[0]  # First test sample predictions
actual_topics = test_data.squeeze(1).numpy()[0]  # First test sample actual values

plt.figure(figsize=(10, 6))
plt.plot(predicted_topics, label="Predicted Topics", alpha=0.7)
plt.plot(actual_topics, label="Actual Topics", alpha=0.7)
plt.xlabel("Topic Index")
plt.ylabel("Presence (0 or 1)")
plt.title("Predicted vs Actual Topics for a Test Sample")
plt.legend()
plt.show()


# Function to visualize intermediate feature maps
def visualize_feature_maps(model, input_data, layer_name="conv1d"):
    model.eval()
    with torch.no_grad():
        # Forward pass up to the specified layer
        for name, layer in model.named_children():
            input_data = layer(input_data)
            if name == layer_name:
                break
        feature_maps = input_data.squeeze(0).numpy()
    
    # Plot feature maps
    num_features = feature_maps.shape[0]
    plt.figure(figsize=(15, 15))
    for i in range(min(num_features, 16)):  # Visualize up to 16 feature maps
        plt.subplot(4, 4, i + 1)
        plt.plot(feature_maps[i])
        plt.title(f"Feature Map {i + 1}")
        plt.tight_layout()
    plt.show()

# Function to visualize error distribution
def plot_error_distribution(actual, predicted):
    errors = actual - predicted
    plt.figure(figsize=(10, 6))
    plt.hist(errors.flatten(), bins=50, alpha=0.7, color="red", label="Prediction Errors")
    plt.xlabel("Error Value")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")
    plt.legend()
    plt.show()

# Function to visualize correlation heatmap
def plot_correlation_heatmap(actual, predicted):
    import seaborn as sns

    # Correlation matrix
    actual = actual.flatten()
    predicted = predicted.flatten()
    correlation = np.corrcoef(actual, predicted)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap Between Actual and Predicted Topics")
    plt.show()

# Visualize feature maps for a single example from test data
print("Visualizing Feature Maps from CNN Layer:")
visualize_feature_maps(model, test_data[:1])

# Calculate and visualize prediction errors
actual_topics = test_data.squeeze(1).numpy()
predicted_topics = test_output.numpy()
print("Visualizing Prediction Error Distribution:")
plot_error_distribution(actual_topics, predicted_topics)

# Visualize correlation heatmap
print("Visualizing Correlation Heatmap:")
plot_correlation_heatmap(actual_topics, predicted_topics)'''



'''def lentesting(nums):
    
    return len(nums)
nums = [ 1,1,1,1]
print(lentesting(nums))'''
'''class Student:
    def __init__(self,name,age,year):
        self.name = name 
        self.age = age 
        self.year = year 
    def okok(self):
        print(f"the age of the person is {self.age} and he has black hair named{self.name} and also works from {self.year}")
objec = Student("yogesh",23,1991)
class Teacher(Student):
    def __init__(self,name,age,year,height):
        super(Student,self).__init__()
        self.name = name 
        self.age = age
        self.year = year
        self.height = height 
    def koko(self):
        print(f"ok so this man name {self.name} is the pilot ")
ob = Teacher("sid",23,24,15)
ob.koko()'''
'''def triangle(n):
   for i in range(n+1):
        print(" "*(n-i)+"*"*(2*i))
   for i in range(n+1):
        print(" "*(i) + "*"*((2*n)-i))
triangle(4)        '''
'''class Solution:
    def maximumLength(self, s: str) -> int:
        left = 0
        right = 0
        max_length  = 0
        while left < len(s) and right < len(s):
            if s[left] == s[right]:
                left +=1'''
                
'''class Node :
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append (self,data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next:
            if current.next is 
        '''
'''class Node :
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self,data):
        new_node = Node(data)
        self.data = data 
            '''
'''
def lonpre(s):
    prefix = s[:1]
    for char in prefix:
        return char 
    for word in s[1:] :
        return word 
    for character in word :
        if character == char :
            character += 1
        
s = ['aabbc','aabbccd','aabbcd','aabb']
print(lonpre(s))  '''


'''def febonnaci (n):
    if n <=1:
        return n 
    else:
        return febonnaci(n-1) + febonnaci(n-2)
print(febonnaci(6))'''


'''def febonnaci (n):
    series = [0,1]
    for _ in range(2,n):
        series.append(series[-1]+series[-2])
    return series [:n]

        
        

print(febonnaci(4))'''
'''def cou(n):
    
    for i in range(n):
        print(i)
    return 
print(cou(6))'''



'''l = [1,2,3,4,5,6,1,1,3,5]
count= {}
count[i] = count.get(i,0)+1
print (count)'''

l = [7,1,5,9,3,5]
'''l = set(l)
x = [i for i in l ]
highest = x[-1]
second_highest = x[-2]
print(highest)
print(second_highest)'''


'''lis = [1,2,3,4,5,6]
x =lis[-3:]
print(x)'''
'''def febonacci(n):
    if n <=1:
        return n 
    else :
        return febonacci(n-2)+ febonacci(n-1)
def generate_fibonacci_series(n):
    return [febonacci(i) for i in range(n)]
print(generate_fibonacci_series(4))'''
'''original_list = [1, 5, 10, 20]
filtered_list = list(filter(lambda x: x > 10, original_list))  # Using filter with a lambda function

print(filtered_list)
# Output: [20]'''

'''def ff(dict):
    dict1 = {}
    for key,value in dict.items():
        if key == 2 :
            return value+1
        dict1[key]= value
dict= {"a": 10, "b": 20, "c": 5, "d": 30}
print(ff(dict))   '''


'''# Declare an empty dictionary
count_dict = []

# List of numbers
numbers = [10, 15, 22, 33, 10, 22, 15, 10]

# Count occurrences using a loop and save them in the dictionary
for num in numbers:
    count_dict = count_dict.get(num, 0) + 1

print(count_dict)
# Output: {10: 3, 15: 2, 22: 2, 33: 1}
'''


'''def cc(words):
    dict1 = {}  # Initialize an empty dictionary
    for item in words:
        # If the item already exists in the dictionary, increment the count, otherwise set it to 1
        dict1[item] = dict1.get(item, 0) + 1
    return dict1  # Return the dictionary after the loop

words = ['apple', 'app', 'apple', 'cherry', 'banana', 'applle']
print(cc(words))
'''

'''def longest_prefix(lst):
    if not lst :
        return ""
    prefix = lst[0]
    for i in range(len(prefix)):
        for j in lst[1:]:
            if prefix[i] != j[i]:
                return prefix[:i]
    return prefix
lst= ['apple', 'app', 'apple', 'cherry', 'banana', 'applle']
print(longest_prefix(lst))       '''

'''
nums = [10, 5, 8, 20, 3]
x = set(nums)
y = [i for i in x]
print(y[-1])
lst1= [1, 2, 3, 4, 5]
lst2= [4, 5, 6, 7, 8]
for i in lst1:
    for j in lst2:
        if i == j :
            print(i)''''''
def longest_substring(strs:list):
    if not strs :
        return ""
    prefix = strs[0]
    for i in prefix :
        for item in strs[1:]:
            if i< len(prefix) or '''
'''prefix = "sabkbccb"
for i in range(len(prefix)):
    print(prefix[i])
'''
'''def longest_prefix(stsr:list):
    if not stsr :
        return ""
    prefix = stsr[0]
    for i in range(len(prefix)):
        for item in stsr[1:]:
            if i >= len(item) or prefix[i] != item[i]:
                return prefix[:i]
    return prefix 
stsr = ['apple','app,','apppl']
print(longest_prefix(stsr))'''
'''bubble sort '''
'''def bubble_Sort(lst:list):
    n = len(lst)#6
    for i in range(n-1):#n-1 = 5
        for j in range(n-i-1): # 5-1-1 = ,5-2-1,5-3-1
            if lst[j]>lst[j+1]:
                lst[j], lst[j+1] = lst[j+1] , lst[j]
    return lst'''
'''def sort(lst):
    if not lst :
        return ""
    n = len(lst)
    for i in range(n):
        for j in range(n-i-1):
            if lst[j]> lst[j+1]:
                lst[j],lst[j+1] = lst[j+1],lst[j]
                
    return 
lst = [1,3,4,7,3,6,89]
x =sort.lst
y = x[-1]
print(y)


def xnd(dhdhh):
    if conditon :
        return output
    for i in iterable :
        return i+1
    return '''
    
'''n = [1,2,3,4,5,6,7,8,9]
for i in range(len(n)):
    for j in range(len(n)-i-1):
        print(i,j,end =" ")'''
'''def my_decorator(func):
    def wrapper(a, b):
        print(f"Adding {a} and {b}...")  # Message before calling the function
        result = func(a, b)  # Call the original function
        print(f"The result is {result}.")  # Message after calling the function
        return result
    return wrapper
@my_decorator
def add(a, b):
    return a + b

print(add(3, 5))  # Call the decorated function
'''
'''
def longest_subs(l):
    dic = {}
    for key, value in l.items():
        if key == (value / 2):
            if key not in dic:  # Initialize the key in 'dic' if it doesn't exist
                dic[key] = 0
            dic[key] += 1  # Increment the count for the key
    return dic  # Return the dictionary after the loop

l = {1: 4, 2: 4, 3: 6, 4: 8}
print(longest_subs(l))
'''
'''def sliding_windw(lst,target):
    lst1 = []
    left = 0
    for right in range(len(lst)):
        
        if lst[right]+lst[left]<target:
            
            left-=1
        else:
            right+=1    
        '''
        
'''s = "abcdbcuwcbwkfcsbvjecksckfhkfnlcsfjckcwdefgzgijklmnopqrst"
char_count ={}
for char in  s:
    char_count[char]= char_count.get(char,0)+1
    
print(len(char_count))
    '''
'''def decorator(func):
    def wrapper():
        func()
        print(f"{func.name}")
        
    return
@decorator
def do_this():
    return "ok ok"'''
'''import functools

def decorator(func):
    @functools.wraps(func)
    def wrapper():
        result = func()
        print(f"{func.__name__}")
        return result
    return wrapper

@decorator
def do_this():
    return "ok ok"
'''
'''l = ['aaple','absxka','xidxk','xjax ja d']
for num in l :
    if num == 'absxka':
        l.remove(num)
        l.append(num)
    print(l)'''
'''class Node :
    def __init__(self,data ):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append (self,data):
        new_node = Node(data)
        if self.head is None:
            self.head= new_node
            return 
        current = self.head
        while current.next:
            current = current.next
            current.next = new_node
    def display(self,data):
        current = self.head
        while current :
            print(current.data,end ="->")
            current = self.data 
            '''
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
        while current.next :
            current = current.next 
        current.next = new_node
    def display(self):
        current = self.head 
        while current :
            print(current.data ,end ="->")
            current = current.next
        print("none")
        
linked_list = LinkedList()
linked_list.append(10)
linked_list.append(20)
linked_list.append(30)
linked_list.display()
dict1 = {1:1}
dict={} 
for key,value in dict1.items():
    dict[key]=1
print(dict)'''

'''def insertion_sort(arr:list):
    key = arr[1]
    for i in arr :
        if i < key :
            i,key = i ,key 
        else :
            i,key = key,i 

    '''
'''class Node :
    def __init__(self,data):
        self.data = data 
        self.next = None
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
            current = current.next
        current.next = new_node
    def display(self):
        current = self.head
        while current :
            print(current.data,end="->")
            current = current.next
        print('none')
linked_list = LinkedList()
linked_list.append(10)
linked_list.append(20)
linked_list.append(30)
linked_list.display() '''
'''
l ={1,2,3,4,5}
x = len(l)
print(x)'''
l = [1,2,3,4,5,6,5,5]
s = set(l)
if len(s)<len(l):
    print('it contains duplicates')
elif  len(l) ==len(s):
    print('the lsit dont coantian duplicates ')
    