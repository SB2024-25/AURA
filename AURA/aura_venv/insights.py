import ollama
import json
import pandas as pd
import numpy as np
from scipy.stats import zscore

def generate_ai_insights(df, user_context=None):
    """Generate comprehensive AI insights with visualization suggestions"""
    if user_context is None:
        user_context = {}
    
    # Prepare dataset summary
    dataset_summary = {
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['category', 'object']).columns.tolist(),
        "sample_data": df.head().to_dict(),
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict()
    }
    
    # Prepare prompt with user context
    prompt = f"""
    Analyze this dataset and provide structured insights based on the following context:
    - Dataset description: {user_context.get('description', 'Not provided')}
    - User expectations: {user_context.get('expectations', 'Not provided')}
    - Exceptions: {user_context.get('exceptions', 'None')}
    
    Dataset Summary:
    {json.dumps(dataset_summary, indent=2)}
    
    Provide:
    1. Key insights and patterns
    2. Suggested visualizations with recommended chart types and axis configurations
    3. Data quality assessment
    4. Recommendations for further analysis
    """
    
    try:
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        ai_output = response['message']['content']
        
        # Parse and structure the output
        insights = {
            "summary": ai_output,
            "visualization_suggestions": suggest_visualizations(df),
            "data_quality": assess_data_quality(df),
            "recommendations": []
        }
        
        return insights
        
    except Exception as e:
        return {"error": str(e)}

def suggest_visualizations(df):
    """Suggest appropriate visualizations based on data characteristics"""
    suggestions = []
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns
    
    # Suggest scatter plots for numeric relationships
    if len(numeric_cols) >= 2:
        suggestions.append({
            "type": "scatter",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "description": "Show relationship between two numeric variables"
        })
    
    # Suggest histograms for numeric distributions
    for col in numeric_cols:
        suggestions.append({
            "type": "histogram",
            "x": col,
            "description": f"Show distribution of {col}"
        })
    
    # Suggest bar charts for categorical data
    for col in categorical_cols:
        if df[col].nunique() < 20:  # Avoid columns with too many unique values
            suggestions.append({
                "type": "bar",
                "x": col,
                "description": f"Show distribution of {col}"
            })
    
    return suggestions

def assess_data_quality(df, method='auto', threshold=3.0):
    """Enhanced data quality assessment with configurable outlier detection"""
    quality_report = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "outliers": {},
        "outlier_details": {}
    }
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        # Detect outliers using selected method
        if method == 'auto':
            outliers, method_used = detect_outliers_auto(df[col], threshold)
        elif method == 'iqr':
            outliers = detect_outliers_iqr(df[col])
            method_used = 'iqr'
        elif method == 'zscore':
            outliers = detect_outliers_zscore(df[col], threshold)
            method_used = 'zscore'
        elif method == 'mad':
            outliers = detect_outliers_mad(df[col], threshold)
            method_used = 'mad'
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Store results
        quality_report["outliers"][col] = outliers.sum()
        quality_report["outlier_details"][col] = {
            "method": method_used,
            "threshold": threshold,
            "indices": outliers[outliers].index.tolist(),
            "values": df[col][outliers].tolist()
        }
    
    return quality_report

def detect_outliers_auto(series, threshold=3.0):
    """Automatically select best outlier detection method"""
    # Try Z-score first if data appears normally distributed
    if is_normal_distribution(series):
        return detect_outliers_zscore(series, threshold), 'zscore'
    # Use IQR for non-normal distributions
    return detect_outliers_iqr(series), 'iqr'

def detect_outliers_iqr(series):
    """Detect outliers using Interquartile Range method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))

def detect_outliers_zscore(series, threshold=3.0):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(zscore(series))
    return z_scores > threshold

def detect_outliers_mad(series, threshold=3.0):
    """Detect outliers using Median Absolute Deviation method"""
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    modified_z_scores = 0.6745 * (series - median) / mad
    return np.abs(modified_z_scores) > threshold

def is_normal_distribution(series):
    """Check if data appears normally distributed"""
    # Simple check based on skewness
    skewness = series.skew()
    return -1 < skewness < 1
