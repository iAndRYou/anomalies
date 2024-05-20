import pandas as pd

# Assuming the CSV data is stored in a file named 'data.csv'
file_path = './BTC-2021min.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame
print(df.head())

df.plot()