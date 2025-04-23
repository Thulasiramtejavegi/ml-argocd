#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(n_normal=970, n_anomalies=30, random_state=42, output_file='dataset.csv'):
    """
    Generate synthetic data with normal samples and anomalies and save to CSV
    
    Parameters:
    -----------
    n_normal : int
        Number of normal data points to generate
    n_anomalies : int
        Number of anomaly data points to generate
    random_state : int
        Random seed for reproducibility
    output_file : str
        Path to save the CSV file
    """
    print(f"Generating synthetic dataset with {n_normal} normal points and {n_anomalies} anomalies...")
    
    # Create main cluster of normal data
    X_normal, _ = make_blobs(n_samples=n_normal, centers=1, random_state=random_state)
    
    # Create outliers (anomalies)
    X_outliers = np.random.uniform(low=-4, high=4, size=(n_anomalies, 2))
    
    # Combine normal and anomaly data
    X = np.vstack([X_normal, X_outliers])
    
    # Create labels (1 for normal, -1 for anomalies)
    y_true = np.ones(X.shape[0])
    y_true[n_normal:] = -1  # Last n_anomalies points are anomalies
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature1': X[:, 0],
        'feature2': X[:, 1],
        'label': y_true
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Generate dataset with 1000 examples (970 normal, 30 anomalies)
    df = generate_synthetic_data(n_normal=970, n_anomalies=30, output_file='dataset.csv')
    print(f"Generated dataset shape: {df.shape}")
    print(f"Number of normal samples: {np.sum(df['label'] == 1)}")
    print(f"Number of anomalies: {np.sum(df['label'] == -1)}")
