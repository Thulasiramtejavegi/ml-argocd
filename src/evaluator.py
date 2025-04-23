#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from joblib import load

def evaluate_model(model_path='isolation_forest_model.joblib', scaler_path='scaler.joblib', data_path='dataset.csv'):
    """
    Evaluate a trained Isolation Forest model
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    scaler_path : str
        Path to the saved scaler
    data_path : str
        Path to the CSV file with labeled data
        
    Returns:
    --------
    metrics : dict
        Dictionary with evaluation metrics
    """
    print(f"Evaluating model from {model_path}...")
    
    # Load the model and scaler
    model = load(model_path)
    scaler = load(scaler_path)
    
    # Load the data
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1).values
    y_true = df['label'].values
    
    # Scale the data
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=-1)
    recall = recall_score(y_true, y_pred, pos_label=-1)
    f1 = f1_score(y_true, y_pred, pos_label=-1)
    
    # Format results
    metrics = {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return metrics

if __name__ == "__main__":
    metrics = evaluate_model()
