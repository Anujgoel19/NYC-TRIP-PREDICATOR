"""
Preprocessing and Feature Engineering Module
==============================================
Handles data cleaning, feature engineering, and preparation for model training.

Author: ML Project
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Class for preprocessing and feature engineering of taxi data.
    
    Features engineered:
    - Distance in kilometers and meters
    - Trip fare estimation
    - Speed metrics
    - Time-based features
    - Geographical features
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the preprocessor.
        
        Args:
            df (pd.DataFrame): Raw dataframe
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def haversine_distance(self, lat1: np.ndarray, lon1: np.ndarray, 
                          lat2: np.ndarray, lon2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate haversine distance between two GPS coordinates.
        
        Args:
            lat1, lon1: Starting coordinates
            lat2, lon2: Ending coordinates
            
        Returns:
            Tuple of (distance_km, distance_meters)
        """
        # Earth's radius in kilometers
        R = 6371.0
        
        # Convert degrees to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        distance_km = R * c
        distance_meters = distance_km * 1000
        
        return distance_km, distance_meters
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer new features from raw data.
        
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        print("Engineered Features:")
        print("-" * 60)
        
        df = self.df.copy()
        
        # 1. Convert datetime columns
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
        print("✓ Datetime conversion")
        
        # 2. Distance calculations (verify with existing distance)
        dist_km, dist_meters = self.haversine_distance(
            df['pickup_latitude'].values,
            df['pickup_longitude'].values,
            df['dropoff_latitude'].values,
            df['dropoff_longitude'].values
        )
        df['distance_km'] = dist_km
        df['distance_meters'] = dist_meters
        print("✓ Distance in kilometers and meters calculated")
        
        # 3. Trip fare estimation (NYC taxi rate: $2.50 base + $2.50 per mile + time charges)
        # Base fare + distance charge + time charge (assuming rush hour multiplier)
        BASE_FARE = 2.50
        COST_PER_KM = 1.56  # ~$2.50 per mile
        COST_PER_MINUTE = 0.35  # Time-based charge
        
        trip_duration_minutes = df['trip_duration'] / 60
        distance_miles = df['distance_km'] * 0.621371  # Convert km to miles
        
        # Calculate estimated fare
        df['estimated_fare'] = (BASE_FARE + 
                               (distance_miles * 2.50) + 
                               (trip_duration_minutes * COST_PER_MINUTE))
        print("✓ Trip fare estimation")
        
        # 4. Speed-related features
        df['speed_kmh'] = df['distance_km'] / (df['trip_duration'] / 3600)  # km/h
        df['speed_kmh'] = df['speed_kmh'].replace([np.inf, -np.inf], 0)  # Handle division by zero
        print("✓ Speed metrics calculated")
        
        # 5. Advanced time features
        df['pickup_is_weekend'] = df['pickup_datetime'].dt.dayofweek >= 5
        df['pickup_is_weekend'] = df['pickup_is_weekend'].astype(int)
        print("✓ Weekend flag")
        
        # Time periods
        df['time_period'] = pd.cut(df['pickup_hour'], 
                                   bins=[0, 6, 12, 18, 24],
                                   labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                   include_lowest=True)
        print("✓ Time period categorization")
        
        # 6. Geographical features (pickup zone estimates)
        df['pickup_lat_zone'] = pd.cut(df['pickup_latitude'], bins=10)
        df['pickup_lon_zone'] = pd.cut(df['pickup_longitude'], bins=10)
        print("✓ Geographical zones")
        
        # 7. Passenger-related features
        df['is_solo_trip'] = (df['passenger_count'] == 1).astype(int)
        df['is_large_group'] = (df['passenger_count'] >= 5).astype(int)
        print("✓ Passenger-related features")
        
        # 8. Vendor identifier encoding
        df['vendor_is_1'] = (df['vendor_id'] == 1).astype(int)
        print("✓ Vendor encoding")
        
        print("-" * 60 + "\n")
        
        self.df = df
        return df
    
    def prepare_features(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Prepare and separate features for modeling.
        
        Returns:
            Tuple of (processed_df, numerical_features, categorical_features)
        """
        df = self.df.copy()
        
        # Define feature categories
        numerical_features = [
            'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 'pickup_hour', 'pickup_day',
            'pickup_month', 'distance', 'distance_km', 'distance_meters',
            'estimated_fare', 'speed_kmh', 'is_solo_trip', 'is_large_group',
            'vendor_is_1', 'pickup_is_weekend'
        ]
        
        categorical_features = ['store_and_fwd_flag', 'time_period']
        
        # Encode categorical variables
        for col in categorical_features:
            if col in df.columns:
                encoder = LabelEncoder()
                df[col + '_encoded'] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
                numerical_features.append(col + '_encoded')
        
        # Drop non-essential columns
        drop_columns = [col for col in df.columns if col in 
                       ['id', 'pickup_datetime', 'dropoff_datetime', 
                        'store_and_fwd_flag', 'time_period', 'pickup_lat_zone', 
                        'pickup_lon_zone'] and col not in numerical_features]
        
        df = df.drop(columns=drop_columns, errors='ignore')
        
        print(f"✓ Prepared {len(numerical_features)} numerical features")
        print(f"✓ Encoded {len(categorical_features)} categorical features\n")
        
        return df, numerical_features, categorical_features
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X (pd.DataFrame): Features to scale
            fit (bool): Whether to fit the scaler
            
        Returns:
            np.ndarray: Scaled features
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            print("✓ Scaler fitted and features scaled")
        else:
            X_scaled = self.scaler.transform(X)
            print("✓ Features scaled using existing scaler")
        
        return X_scaled
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Complete preprocessing pipeline.
        
        Returns:
            Tuple of (X_scaled, y, feature_names)
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Engineer features
        self.engineer_features()
        
        # Prepare features
        df_processed, numerical_features, _ = self.prepare_features()
        
        # Separate features and target
        X = df_processed[numerical_features].copy()
        y = df_processed['trip_duration'].values
        
        # Handle any remaining infinite or NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scale_features(X, fit=True)
        
        print(f"✓ Final features shape: {X_scaled.shape}")
        print(f"✓ Target variable shape: {y.shape}")
        print("="*60 + "\n")
        
        return X_scaled, y, numerical_features


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("cleaned_taxi_data.csv")
    preprocessor = DataPreprocessor(df)
    X, y, features = preprocessor.prepare_data()
    print(f"Data prepared successfully!")
    print(f"Features: {features}")