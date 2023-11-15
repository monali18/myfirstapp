import pandas as pd

# Create a sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [6, 7, 8, 9, 10],
    'C': [11, 12, 13, 14, 15],
    'D': ['Group1', 'Group1', 'Group2', 'Group2', 'Group3']
}

df = pd.DataFrame(data)

# Group by column D and calculate the sum of A, B, and C for each group
result = df.groupby('D').sum({'A', 'B', 'C'}).reset_index()

# Print the result
print(result)
