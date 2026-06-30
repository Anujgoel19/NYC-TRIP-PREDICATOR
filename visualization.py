"""
Visualization Module
====================
Handles creation of various plots and visualizations for data exploration
and model performance analysis.

Author: ML Project
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Class for creating various visualizations of taxi data and model performance.
    """
    
    def __init__(self, df: pd.DataFrame = None):
        """
        Initialize visualizer.
        
        Args:
            df (pd.DataFrame): Original dataframe (optional)
        """
        self.df = df
        
    def plot_trip_duration_distribution(self) -> plt.Figure:
        """
        Plot distribution of trip duration.
        
        Returns:
            plt.Figure: Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.df['trip_duration'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Trip Duration (seconds)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Distribution of Trip Duration', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Box plot
        axes[1].boxplot(self.df['trip_duration'], vert=True)
        axes[1].set_ylabel('Trip Duration (seconds)', fontsize=11, fontweight='bold')
        axes[1].set_title('Trip Duration Box Plot', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_distance_vs_duration(self) -> plt.Figure:
        """
        Plot relationship between distance and trip duration.
        
        Returns:
            plt.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sample data if dataset is large
        sample_df = self.df.sample(min(5000, len(self.df)), random_state=42)
        
        scatter = ax.scatter(sample_df['distance'], sample_df['trip_duration'], 
                           alpha=0.5, c=sample_df['passenger_count'], 
                           cmap='viridis', s=30, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Distance (km)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Trip Duration (seconds)', fontsize=11, fontweight='bold')
        ax.set_title('Distance vs Trip Duration (colored by Passenger Count)', 
                    fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Passenger Count', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_pickup_hour_analysis(self) -> plt.Figure:
        """
        Plot trip characteristics by pickup hour.
        
        Returns:
            plt.Figure: Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Average duration by hour
        hour_stats = self.df.groupby('pickup_hour').agg({
            'trip_duration': ['mean', 'count'],
            'distance': 'mean'
        })
        
        axes[0].bar(hour_stats.index, hour_stats['trip_duration']['mean'], 
                   color='coral', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Pickup Hour', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Average Trip Duration (seconds)', fontsize=11, fontweight='bold')
        axes[0].set_title('Average Trip Duration by Hour of Day', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='y')
        
        # Trip count by hour
        axes[1].bar(hour_stats.index, hour_stats['trip_duration']['count'], 
                   color='lightblue', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Pickup Hour', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Number of Trips', fontsize=11, fontweight='bold')
        axes[1].set_title('Trip Count by Hour of Day', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_passenger_count_analysis(self) -> plt.Figure:
        """
        Plot trip characteristics by passenger count.
        
        Returns:
            plt.Figure: Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Passenger count distribution
        passenger_counts = self.df['passenger_count'].value_counts().sort_index()
        axes[0].bar(passenger_counts.index, passenger_counts.values, 
                   color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Number of Passengers', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Distribution of Passenger Count', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='y')
        
        # Average duration by passenger count
        passenger_duration = self.df.groupby('passenger_count')['trip_duration'].mean()
        axes[1].plot(passenger_duration.index, passenger_duration.values, 
                    marker='o', linewidth=2, markersize=8, color='darkblue')
        axes[1].set_xlabel('Number of Passengers', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Average Trip Duration (seconds)', fontsize=11, fontweight='bold')
        axes[1].set_title('Average Duration by Passenger Count', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, metrics_summary: pd.DataFrame) -> plt.Figure:
        """
        Plot model performance comparison.
        
        Args:
            metrics_summary (pd.DataFrame): Summary of model metrics
            
        Returns:
            plt.Figure: Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Test Accuracy comparison
        axes[0, 0].barh(metrics_summary['Model'], metrics_summary['Test Accuracy'], 
                       color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Model Test Accuracy Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3, axis='x')
        for i, v in enumerate(metrics_summary['Test Accuracy']):
            axes[0, 0].text(v + 1, i, f'{v:.2f}%', va='center', fontweight='bold')
        
        # Test RMSE comparison
        axes[0, 1].barh(metrics_summary['Model'], metrics_summary['Test RMSE'], 
                       color='orange', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('RMSE (seconds)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Model Test RMSE Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3, axis='x')
        for i, v in enumerate(metrics_summary['Test RMSE']):
            axes[0, 1].text(v + 10, i, f'{v:.2f}s', va='center', fontweight='bold')
        
        # Test MAE comparison
        axes[1, 0].barh(metrics_summary['Model'], metrics_summary['Test MAE'], 
                       color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('MAE (seconds)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Model Test MAE Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3, axis='x')
        for i, v in enumerate(metrics_summary['Test MAE']):
            axes[1, 0].text(v + 3, i, f'{v:.2f}s', va='center', fontweight='bold')
        
        # Train vs Test Accuracy
        x = np.arange(len(metrics_summary))
        width = 0.35
        axes[1, 1].bar(x - width/2, metrics_summary['Train Accuracy'], width, 
                      label='Train', color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, 1].bar(x + width/2, metrics_summary['Test Accuracy'], width, 
                      label='Test', color='darkblue', edgecolor='black', alpha=0.7)
        axes[1, 1].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Train vs Test Accuracy', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics_summary['Model'], rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """
        Plot predicted vs actual values.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            plt.Figure: Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Trip Duration (seconds)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Predicted Trip Duration (seconds)', fontsize=11, fontweight='bold')
        axes[0].set_title('Predicted vs Actual Trip Duration', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Residuals
        residuals = y_true - y_pred
        axes[1].hist(residuals, bins=50, color='purple', edgecolor='black', alpha=0.7)
        axes[1].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {residuals.mean():.2f}')
        axes[1].set_xlabel('Residuals (seconds)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1].set_title('Distribution of Prediction Residuals', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            top_n (int): Number of top features to display
            
        Returns:
            plt.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_df.head(top_n)
        ax.barh(top_features['Feature'], top_features['Importance'], 
               color='teal', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis='x')
        
        for i, v in enumerate(top_features['Importance']):
            ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Save figure to disk.
        
        Args:
            fig (plt.Figure): Figure object
            filename (str): Filename to save as
        """
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved as {filename}")
        plt.close(fig)


if __name__ == "__main__":
    print("Visualization Module")
    print("This module is meant to be imported and used in the main script.")