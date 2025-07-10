import pandas as pd

# Define numerical data
data = {
    'A': [10, 20, 30, 40],
    'B': [1.5, 2.5, 3.5, 4.5],
    'C': [100, 200, 300, 400]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
print(df,"\n--------------")
print(df.min())
print(df.max())
print(df.count())
print(df.mean())


