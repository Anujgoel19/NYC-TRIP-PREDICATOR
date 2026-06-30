import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("nyc_taxi_trip_duration.csv")

# Convert datetime columns
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

if 'dropoff_datetime' in df.columns:
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

# Create time features
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_day'] = df['pickup_datetime'].dt.dayofweek
df['pickup_month'] = df['pickup_datetime'].dt.month

# Distance in KM (approximate)
df['distance'] = (
    np.sqrt(
        (df['pickup_latitude'] - df['dropoff_latitude'])**2 +
        (df['pickup_longitude'] - df['dropoff_longitude'])**2
    ) * 111
)

# Remove unrealistic trip durations
df = df[
    (df['trip_duration'] >= 60) &
    (df['trip_duration'] <= 7200)
]

# Remove invalid passenger counts
df = df[
    (df['passenger_count'] > 0) &
    (df['passenger_count'] <= 6)
]

# Remove very large distances (optional)
df = df[df['distance'] <= 50]

# Save cleaned dataset
df.to_csv("cleaned_taxi_data.csv", index=False)

print("Cleaning completed!")
print("Shape:", df.shape)

print("\nDistance Statistics:")
print(df["distance"].describe())