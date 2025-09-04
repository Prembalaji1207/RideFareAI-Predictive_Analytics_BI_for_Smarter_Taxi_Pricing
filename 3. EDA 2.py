import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your cleaned CSV
df = pd.read_csv("cleaned_taxi.csv")   # change file name if different

# 2. Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# 3. Create distance column
df['distance_km'] = haversine(
    df['pickup_longitude'], df['pickup_latitude'],
    df['dropoff_longitude'], df['dropoff_latitude']
)

# 4. Plot distribution
df['distance_km'].hist(bins=50, figsize=(8,5))
plt.xlabel("Distance (km)")
plt.ylabel("Number of Trips")
plt.title("Distribution of Trip Distances")
plt.show()

# 5. Average fare by distance bucket
df['distance_bucket'] = pd.cut(df['distance_km'], bins=[0,2,5,10,20,50])
print(df.groupby('distance_bucket')['fare_amount'].mean())
