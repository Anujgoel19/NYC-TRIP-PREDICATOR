"""
Model Training and Evaluation Module
====================================
Handles model building, training, evaluation, and performance metrics.

Author: ML Project
Date: 2024
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import joblib
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class TaxiDurationModel:
    """
    Class for building and training taxi trip duration prediction models.
    
    Models implemented:
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model handler.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target variable
            test_size (float): Proportion of test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"✓ Data split: {len(self.X_train)} train, {len(self.X_test)} test")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_linear_regression(self) -> None:
        """Train linear regression model."""
        print("\n  Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = lr
        print("  ✓ Linear Regression trained")
    
    def train_random_forest(self) -> None:
        """Train random forest model."""
        print("  Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        print("  ✓ Random Forest trained")
    
    def train_gradient_boosting(self) -> None:
        """Train gradient boosting model."""
        print("  Training Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )
        gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb
        print("  ✓ Gradient Boosting trained")
    
    def train_xgboost(self) -> None:
        """Train XGBoost model."""
        print("  Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        xgb_model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb_model
        print("  ✓ XGBoost trained")
    
    def train_all_models(self) -> None:
        """Train all models."""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        self.train_linear_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_xgboost()
        
        print("\n✓ All models trained successfully!")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict: Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Convert R² to accuracy-like metric (0-100%)
        # Clamp R² between 0 and 1, then convert to percentage
        accuracy_score = max(0, min(100, r2 * 100))
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Accuracy': accuracy_score
        }
    
    def evaluate_models(self) -> None:
        """Evaluate all trained models."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        best_accuracy = 0
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            
            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Training metrics
            train_metrics = self.calculate_metrics(self.y_train, y_train_pred)
            print(f"  Training Metrics:")
            print(f"    RMSE: {train_metrics['RMSE']:.2f} seconds")
            print(f"    MAE:  {train_metrics['MAE']:.2f} seconds")
            print(f"    R²:   {train_metrics['R2']:.4f}")
            print(f"    Accuracy Score: {train_metrics['Accuracy']:.2f}%")
            
            # Testing metrics
            test_metrics = self.calculate_metrics(self.y_test, y_test_pred)
            print(f"  Testing Metrics:")
            print(f"    RMSE: {test_metrics['RMSE']:.2f} seconds")
            print(f"    MAE:  {test_metrics['MAE']:.2f} seconds")
            print(f"    R²:   {test_metrics['R2']:.4f}")
            print(f"    Accuracy Score: {test_metrics['Accuracy']:.2f}%")
            
            # Store metrics
            self.metrics[model_name] = {
                'train': train_metrics,
                'test': test_metrics
            }
            
            # Track best model
            if test_metrics['Accuracy'] > best_accuracy:
                best_accuracy = test_metrics['Accuracy']
                self.best_model = model
                self.best_model_name = model_name
        
        print("\n" + "="*60)
        print(f"✓ BEST MODEL: {self.best_model_name} ({best_accuracy:.2f}% accuracy)")
        print("="*60)
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if self.best_model_name not in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            print("Feature importance available only for tree-based models.")
            return None
        
        importance = self.best_model.feature_importances_
        features_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return features_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save the best model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        joblib.dump(self.best_model, filepath)
        print(f"✓ Best model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.best_model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the best model.
        
        Args:
            X (np.ndarray): Features to predict on
            
        Returns:
            np.ndarray: Predictions
        """
        return self.best_model.predict(X)
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get a summary of all model metrics.
        
        Returns:
            pd.DataFrame: Summary dataframe
        """
        summary = []
        for model_name, metrics in self.metrics.items():
            summary.append({
                'Model': model_name,
                'Train Accuracy': metrics['train']['Accuracy'],
                'Test Accuracy': metrics['test']['Accuracy'],
                'Test RMSE': metrics['test']['RMSE'],
                'Test MAE': metrics['test']['MAE'],
                'Test R²': metrics['test']['R2']
            })
        
        return pd.DataFrame(summary)


if __name__ == "__main__":
    # Example usage
    print("ML Model Module")
    print("This module is meant to be imported and used in the main script.")