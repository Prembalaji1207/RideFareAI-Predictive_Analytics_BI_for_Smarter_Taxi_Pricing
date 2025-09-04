#EDA

import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("cleaned_taxi_data.csv")

# -----------------------------
# 1. Overview of the dataset
# -----------------------------
print("Dataset shape:", df.shape)
print("\nSummary statistics:\n", df.describe())
print("\nData types:\n", df.dtypes)

# -----------------------------
# 2. Fare amount distribution
# -----------------------------
plt.figure(figsize=(8,5))
df['fare_amount'].hist(bins=50)
plt.xlabel("Fare Amount ($)")
plt.ylabel("Number of Trips")
plt.title("Distribution of Taxi Fares")
plt.show()

# -----------------------------
# 3. Passenger count distribution
# -----------------------------
plt.figure(figsize=(6,4))
df['passenger_count'].value_counts().sort_index().plot(kind='bar')
plt.xlabel("Passenger Count")
plt.ylabel("Number of Trips")
plt.title("Passenger Count Distribution")
plt.show()

# -----------------------------
# 4. Pickup time analysis
# -----------------------------
# Convert pickup_datetime to datetime format
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# Extract hour, day of week
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.day_name()

# Trips by hour
plt.figure(figsize=(8,4))
df['hour'].value_counts().sort_index().plot(kind='bar')
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.title("Trips by Hour of Day")
plt.show()

# Trips by day of week
plt.figure(figsize=(7,4))
df['day_of_week'].value_counts().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
).plot(kind='bar')
plt.xlabel("Day of Week")
plt.ylabel("Number of Trips")
plt.title("Trips by Day of Week")
plt.show()

# -----------------------------
# 5. Average fare by passenger count
# -----------------------------
avg_fare_by_passenger = df.groupby('passenger_count')['fare_amount'].mean()
print("\nAverage Fare by Passenger Count:\n", avg_fare_by_passenger)



#EXTRA EDA

##6. Trip Distance Feature

import numpy as np

# Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['distance_km'] = haversine(
    df['pickup_longitude'], df['pickup_latitude'],
    df['dropoff_longitude'], df['dropoff_latitude']
)

# Plot distance distribution
df['distance_km'].hist(bins=50, figsize=(8,5))
plt.xlabel("Distance (km)")
plt.ylabel("Number of Trips")
plt.title("Distribution of Trip Distances")
plt.show()

# Average fare by distance buckets
df['distance_bucket'] = pd.cut(df['distance_km'], bins=[0,2,5,10,20,50])
print(df.groupby('distance_bucket')['fare_amount'].mean())



##7. Fare vs. Distance Relationship

sample = df.sample(5000)  # take a sample to avoid heavy plotting
plt.scatter(sample['distance_km'], sample['fare_amount'], alpha=0.3)
plt.xlabel("Distance (km)")
plt.ylabel("Fare Amount ($)")
plt.title("Fare vs Distance")
plt.show()


#8.Correlation Heatmap

import seaborn as sns

corr = df[['fare_amount','passenger_count','distance_km','hour']].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
