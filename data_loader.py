"""
Data Loader Module
==================
Handles loading, basic exploration, and initial inspection of the taxi dataset.

Author: ML Project
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class TaxiDataLoader:
    """
    Class for loading and exploring taxi dataset.
    
    Attributes:
        filepath (str): Path to the CSV file
        df (pd.DataFrame): Loaded dataframe
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the data loader.
        
        Args:
            filepath (str): Path to the CSV file
        """
        self.filepath = filepath
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the taxi dataset from CSV.
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)
        print(f"✓ Data loaded successfully!")
        print(f"  - Shape: {self.df.shape}")
        print(f"  - Columns: {len(self.df.columns)}")
        return self.df
    
    def explore_data(self) -> None:
        """
        Perform initial data exploration and print statistics.
        """
        if self.df is None:
            print("Error: Data not loaded. Call load_data() first.")
            return
        
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        
        print("\n1. Dataset Shape:")
        print(f"   Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        print("\n2. Column Information:")
        print(self.df.dtypes)
        
        print("\n3. Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("   No missing values found! ✓")
        
        print("\n4. Statistical Summary:")
        print(self.df.describe())
        
        print("\n5. Categorical Features:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"   {col}: {self.df[col].nunique()} unique values")
        
        print("\n6. Target Variable (trip_duration):")
        print(f"   Min: {self.df['trip_duration'].min()} seconds")
        print(f"   Max: {self.df['trip_duration'].max()} seconds")
        print(f"   Mean: {self.df['trip_duration'].mean():.2f} seconds")
        print(f"   Median: {self.df['trip_duration'].median():.2f} seconds")
        print("="*60 + "\n")
        
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded dataframe.
        
        Returns:
            pd.DataFrame: The loaded dataframe
        """
        return self.df.copy() if self.df is not None else None


if __name__ == "__main__":
    # Example usage
    loader = TaxiDataLoader("cleaned_taxi_data.csv")
    df = loader.load_data()
    loader.explore_data()