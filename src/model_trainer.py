#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

def load_data(csv_file):
    """
    Load data from CSV file
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
        
    Returns:
    --------
    X : array of shape (n_samples, n_features)
        Features
    y : array of shape (n_samples,)
        Labels (if available, otherwise None)
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Check if label column exists
    if 'label' in df.columns:
        X = df.drop('label', axis=1).values
        y = df['label'].values
    else:
        X = df.values
        y = None
    
    return X, y

def train_isolation_forest(X, contamination=0.1, n_estimators=100, max_samples='auto', random_state=42):
    """
    Train an Isolation Forest model
    
    Parameters:
    -----------
    X : array of shape (n_samples, n_features)
        Training data
    contamination : float
        Expected proportion of anomalies in the dataset
    n_estimators : int
        Number of base estimators (trees)
    max_samples : int or float or str
        Number of samples to draw to train each base estimator
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    model : IsolationForest
        Trained Isolation Forest model
    """
    print("Training Isolation Forest model...")
    
    # Initialize the model
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        verbose=0
    )
    
    # Train the model
    model.fit(X)
    
    return model

def save_model_and_scaler(model, scaler, model_path='/app/models/isolation_forest_model.joblib', scaler_path='/app/models/scaler.joblib'):
    """
    Save the trained model and scaler to disk
    
    Parameters:
    -----------
    model : trained model
        Model to save
    scaler : StandardScaler
        Fitted scaler
    model_path : str
        Path to save the model
    scaler_path : str
        Path to save the scaler
    """
    dump(model, model_path)
    dump(scaler, scaler_path)
    print(f"Model saved as '{model_path}'")
    print(f"Scaler saved as '{scaler_path}'")

if __name__ == "__main__":
    # Load data from CSV
    X, y = load_data('dataset.csv')
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = train_isolation_forest(X_scaled)
    
    # Save model and scaler
    save_model_and_scaler(model, scaler)
