"""
Main ML Pipeline Orchestrator
=============================
Orchestrates the complete machine learning pipeline for taxi trip duration prediction.
This script runs end-to-end: data loading → preprocessing → training → evaluation → visualization.

Author: ML Project
Date: 2024
Usage: python main.py <path_to_data.csv>
"""

import sys
import os
import numpy as np
import pandas as pd
from data_loader import TaxiDataLoader
from preprocessing import DataPreprocessor
from model import TaxiDurationModel
from visualization import Visualizer
import warnings
warnings.filterwarnings('ignore')


def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print(" " * 15 + "NYC TAXI TRIP DURATION PREDICTION")
    print(" " * 20 + "MACHINE LEARNING PROJECT")
    print("="*70 + "\n")
    
    # =====================================================================
    # STEP 1: LOAD DATA
    # =====================================================================
    
    # Use provided CSV file or default
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "cleaned_taxi_data.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: File '{data_path}' not found.")
        print("Usage: python main.py <path_to_data.csv>")
        sys.exit(1)
    
    loader = TaxiDataLoader(data_path)
    df = loader.load_data()
    loader.explore_data()
    
    # =====================================================================
    # STEP 2: PREPROCESSING & FEATURE ENGINEERING
    # =====================================================================
    
    preprocessor = DataPreprocessor(df)
    X_scaled, y, feature_names = preprocessor.prepare_data()
    
    # =====================================================================
    # STEP 3: SPLIT DATA
    # =====================================================================
    
    print("\n" + "="*60)
    print("TRAIN-TEST SPLIT")
    print("="*60)
    
    model_handler = TaxiDurationModel(random_state=42)
    X_train, X_test, y_train, y_test = model_handler.split_data(X_scaled, y, test_size=0.2)
    
    # =====================================================================
    # STEP 4: TRAIN MODELS
    # =====================================================================
    
    model_handler.train_all_models()
    
    # =====================================================================
    # STEP 5: EVALUATE MODELS
    # =====================================================================
    
    model_handler.evaluate_models()
    metrics_summary = model_handler.get_metrics_summary()
    
    print("\n" + "="*60)
    print("MODELS SUMMARY TABLE")
    print("="*60)
    print(metrics_summary.to_string(index=False))
    print("="*60 + "\n")
    
    # =====================================================================
    # STEP 6: GET FEATURE IMPORTANCE
    # =====================================================================
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (TOP 15)")
    print("="*60)
    
    importance_df = model_handler.get_feature_importance(feature_names)
    if importance_df is not None:
        print(importance_df.head(15).to_string(index=False))
    
    print("="*60 + "\n")
    
    # =====================================================================
    # STEP 7: CREATE VISUALIZATIONS
    # =====================================================================
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    visualizer = Visualizer(df)
    
    # Trip duration distribution
    print("Creating: Trip Duration Distribution...")
    fig1 = visualizer.plot_trip_duration_distribution()
    visualizer.save_figure(fig1, 'visualizations/01_trip_duration_distribution.png')
    
    # Distance vs Duration
    print("Creating: Distance vs Duration Analysis...")
    fig2 = visualizer.plot_distance_vs_duration()
    visualizer.save_figure(fig2, 'visualizations/02_distance_vs_duration.png')
    
    # Pickup Hour Analysis
    print("Creating: Pickup Hour Analysis...")
    fig3 = visualizer.plot_pickup_hour_analysis()
    visualizer.save_figure(fig3, 'visualizations/03_pickup_hour_analysis.png')
    
    # Passenger Count Analysis
    print("Creating: Passenger Count Analysis...")
    fig4 = visualizer.plot_passenger_count_analysis()
    visualizer.save_figure(fig4, 'visualizations/04_passenger_count_analysis.png')
    
    # Model Comparison
    print("Creating: Model Comparison...")
    fig5 = visualizer.plot_model_comparison(metrics_summary)
    visualizer.save_figure(fig5, 'visualizations/05_model_comparison.png')
    
    # Predictions vs Actual
    print("Creating: Predictions vs Actual...")
    y_pred = model_handler.predict(X_test)
    fig6 = visualizer.plot_predictions_vs_actual(y_test, y_pred)
    visualizer.save_figure(fig6, 'visualizations/06_predictions_vs_actual.png')
    
    # Feature Importance
    if importance_df is not None:
        print("Creating: Feature Importance Plot...")
        fig7 = visualizer.plot_feature_importance(importance_df, top_n=15)
        visualizer.save_figure(fig7, 'visualizations/07_feature_importance.png')
    
    print("\n✓ All visualizations created successfully!")
    
    # =====================================================================
    # STEP 8: SAVE MODELS AND PREPROCESSOR
    # =====================================================================
    
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60 + "\n")
    
    model_handler.save_model('models/best_model.pkl')
    
    # Save preprocessor scaler
    import joblib
    joblib.dump(preprocessor.scaler, 'models/scaler.pkl')
    print("✓ Scaler saved to models/scaler.pkl")
    
    # Save feature names
    with open('models/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    print("✓ Feature names saved to models/feature_names.txt")
    
    print("="*60 + "\n")
    
    # =====================================================================
    # STEP 9: FINAL SUMMARY
    # =====================================================================
    
    print("\n" + "="*70)
    print("PROJECT SUMMARY")
    print("="*70)
    
    best_accuracy = metrics_summary.loc[
        metrics_summary['Model'] == model_handler.best_model_name, 'Test Accuracy'
    ].values[0]
    
    print(f"\n✓ Best Model: {model_handler.best_model_name}")
    print(f"✓ Test Accuracy: {best_accuracy:.2f}% (Target: >75%)")
    print(f"✓ Models Trained: {len(model_handler.models)}")
    print(f"✓ Features Used: {len(feature_names)}")
    print(f"✓ Training Samples: {len(X_train):,}")
    print(f"✓ Test Samples: {len(X_test):,}")
    
    if best_accuracy > 75:
        print(f"\n✓ SUCCESS! Model exceeds 75% accuracy threshold! 🎉")
    else:
        print(f"\n⚠ Note: Model accuracy is {best_accuracy:.2f}%, slightly below 75% target")
    
    print("\nAll outputs saved to:")
    print(f"  - Models: models/")
    print(f"  - Visualizations: visualizations/")
    print(f"\nNext Step: Run 'streamlit run app.py' to launch the web interface")
    
    print("\n" + "="*70 + "\n")
    
    return model_handler, metrics_summary


if __name__ == "__main__":
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Run main pipeline
    model_handler, metrics = main()