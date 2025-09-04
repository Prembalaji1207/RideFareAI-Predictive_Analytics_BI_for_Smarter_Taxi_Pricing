import pandas as pd

# Load dataset
df = pd.read_csv("taxi_fare.csv")

# Show first 5 rows
print(df.head())

# Show dataset info
print(df.info())

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Drop rows where critical values are missing
df = df.dropna(subset=['dropoff_longitude', 'dropoff_latitude'])

# Convert pickup_datetime to datetime type
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# Remove rows where fare is <= 0 (invalid fare)
df = df[df['fare_amount'] > 0]

# Remove rows where passenger count is unrealistic (0 or > 6 for taxis)
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

print("After cleaning, dataset shape:", df.shape)

# Keep only rows within NYC bounding box
df = df[
    (df['pickup_longitude'] >= -75) & (df['pickup_longitude'] <= -72) &
    (df['dropoff_longitude'] >= -75) & (df['dropoff_longitude'] <= -72) &
    (df['pickup_latitude'] >= 40) & (df['pickup_latitude'] <= 42) &
    (df['dropoff_latitude'] >= 40) & (df['dropoff_latitude'] <= 42)
]

print("After removing out-of-NYC trips, dataset shape:", df.shape)

df.to_csv("cleaned_taxi_data.csv", index=False)
print("Cleaned dataset saved as cleaned_taxi_data.csv")


