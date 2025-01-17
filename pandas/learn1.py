'''
1. From a Dictionary
a) Dictionary of Lists
Each key represents a column, and the values are the data for that column.


data = {"a": [1, 2, 3], "b": [4, 5, 6]}
df = pd.DataFrame(data)
print(df)
Output:


   a  b
0  1  4
1  2  5
2  3  6
b) Dictionary of Series
Keys are column names, and pd.Series provides indexed data.


data = {"a": pd.Series([1, 2, 3], index=[0, 1, 2]),
        "b": pd.Series([4, 5], index=[0, 1])}
df = pd.DataFrame(data)
print(df)
Output:

     a    b
0  1.0  4.0
1  2.0  5.0
2  3.0  NaN



2. From a List of Lists
Each inner list represents a row.


data = [[1, 2, 3], [4, 5, 6]]
df = pd.DataFrame(data, columns=["a", "b", "c"])
print(df)
Output:


   a  b  c
0  1  2  3
1  4  5  6


3. From a List of Dictionaries
Each dictionary represents a row, and keys become column names.


data = [{"a": 1, "b": 2}, {"a": 3, "b": 4, "c": 5}]
df = pd.DataFrame(data)
print(df)
Output:


   a  b    c
0  1  2  NaN
1  3  4  5.0



4. From a NumPy Array
Create a DataFrame from a NumPy array, specifying column names.


import numpy as np
data = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(data, columns=["a", "b"])
print(df)
Output:


   a  b
0  1  2
1  3  4



5. From a Scalar Value
Generate a DataFrame with repeated scalar values.


df = pd.DataFrame(5, index=range(3), columns=["a", "b"])
print(df)
Output:


   a  b
0  5  5
1  5  5
2  5  5



6. From an Existing DataFrame
Copy or modify an existing DataFrame.

original_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
df = pd.DataFrame(original_df, columns=["a", "b", "c"], dtype=float)
print(df)
Output:


     a    b   c
0  1.0  3.0 NaN
1  2.0  4.0 NaN



7. From an Index or Series
You can pass an index or pd.Series to specify row/column data.


index = pd.Index(["row1", "row2"])
df = pd.DataFrame({"a": [1, 2]}, index=index)
print(df)
Output:


      a
row1  1
row2  2


8. With MultiIndex
Create DataFrame with hierarchical indexing.


index = pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1)])
data = [[1, 2], [3, 4], [5, 6]]
df = pd.DataFrame(data, index=index, columns=["X", "Y"])
print(df)
Output:


       X  Y
A 1    1  2
  2    3  4
B 1    5  6



9. From a CSV or External Source
Although not directly using pd.DataFrame(), data can be loaded from external sources.


df = pd.read_csv("example.csv")
print(df)
Tips for Customization:
index: Custom row labels.
columns: Custom column labels.
dtype: Set a specific data type for all elements.
Handling Missing Data: Use np.nan or None for missing values.
The pd.DataFrame() function adapts to a variety of input formats, making it a powerful tool for creating tabular data structures! Let me know if you'd like examples in a specific context. 😊

'''