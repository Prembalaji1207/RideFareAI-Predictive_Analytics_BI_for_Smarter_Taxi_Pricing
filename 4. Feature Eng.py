import pandas as pd
import numpy as np

# Load your cleaned dataset
df = pd.read_csv("cleaned_taxi_data.csv", parse_dates=["pickup_datetime"])

#cal dist
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

df["distance_km"] = df.apply(lambda row: haversine(
    row["pickup_latitude"], row["pickup_longitude"],
    row["dropoff_latitude"], row["dropoff_longitude"]
), axis=1)


# --- Feature Engineering ---

# 1. Log-transform distance
df["log_distance"] = np.log1p(df["distance_km"])

# 2. Time-based features
df["hour"] = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek   # Monday=0, Sunday=6
df["month"] = df["pickup_datetime"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

# 3. Interaction features
df["distance_passenger"] = df["distance_km"] * df["passenger_count"]

# --- Save updated dataset ---
df.to_csv("feature_engineered_taxi_data.csv", index=False)

print("✅ Feature engineering complete. New file saved as feature_engineered_taxi_data.csv")
print("New columns added:", ["log_distance", "hour", "day_of_week", "month", "is_weekend", "is_night", "distance_passenger"])
