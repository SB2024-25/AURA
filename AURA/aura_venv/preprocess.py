import pandas as pd
import numpy as np
from scipy.stats import zscore

def clean_data(df, exceptions=None):
    """Enhanced data cleaning with outlier detection and handling"""
    df_clean = df.copy()
    
    # Handle missing values
    if exceptions and 'missing_values' in exceptions:
        df_clean.fillna(exceptions['missing_values'], inplace=True)
    else:
        df_clean.dropna(inplace=True)
    
    # Select relevant data types
    df_clean = df_clean.select_dtypes(include=["number", "category", "object"])
    
    # Handle outliers using z-score
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        z_scores = np.abs(zscore(df_clean[numeric_cols]))
        df_clean = df_clean[(z_scores < 3).all(axis=1)]
    
    # Handle exceptions for specific columns
    if exceptions:
        for col, condition in exceptions.get('column_rules', {}).items():
            if col in df_clean.columns:
                df_clean = df_clean.query(condition)
    
    # Convert object columns to category for efficiency
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype('category')
    
    return df_clean, numeric_cols, categorical_cols

def compute_correlation(df, numeric_cols):
    """Compute correlation matrix with efficient memory handling"""
    if len(numeric_cols) > 50:  # For large datasets, compute in chunks
        corr_dict = {}
        for i in range(0, len(numeric_cols), 50):
            cols_chunk = numeric_cols[i:i+50]
            corr_dict.update(df[cols_chunk].corr().to_dict())
        return corr_dict
    return df[numeric_cols].corr().to_dict()

def normalize_data(df, numeric_cols):
    """Normalize numeric columns for better analysis"""
    if numeric_cols:
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    return df
