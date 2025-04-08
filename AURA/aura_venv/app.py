from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

import pandas as pd
import os
import json
import xml.etree.ElementTree as ET
from insights import generate_ai_insights, assess_data_quality
from preprocess import clean_data, compute_correlation, normalize_data



app = Flask(__name__, static_folder='templates/static')
UPLOAD_FOLDER = "datasets"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/description")
def description():
    return render_template('description.html')

@app.route("/results")
def results():
    return render_template('results.html')


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Store filename in localStorage via JavaScript
    return jsonify({
        "file": file.filename,
        "redirect": url_for('description')
    })



@app.route("/insights", methods=["POST"])
def get_insights():
    data = request.json
    file_path = os.path.join(UPLOAD_FOLDER, data["file"])
    
    # Handle different file types
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path) as f:
            data = json.load(f)
        df = pd.json_normalize(data)
    elif file_path.endswith('.xml'):
        tree = ET.parse(file_path)
        root = tree.getroot()
        data = [{child.tag: child.text for child in item} for item in root]
        df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    
    # Get user exceptions if provided
    exceptions = data.get("exceptions", {})
    
    # Clean and preprocess data
    df_clean, num_cols, cat_cols = clean_data(df, exceptions)
    
    # Normalize numeric data
    if num_cols:
        df_clean = normalize_data(df_clean, num_cols)
    
    # Generate correlation matrix
    correlation_matrix = compute_correlation(df_clean, num_cols)
    
    # Get outlier detection configuration
    outlier_config = {
        "method": data.get("outlier_method", "auto"),
        "threshold": float(data.get("outlier_threshold", 3.0))
    }

    # Generate AI insights with user context
    user_context = {
        "description": data.get("description", ""),
        "expectations": data.get("expectations", ""),
        "exceptions": exceptions,
        "outlier_config": outlier_config
    }
    
    # Perform data quality assessment
    data_quality = assess_data_quality(df_clean, 
                                      method=outlier_config['method'],
                                      threshold=outlier_config['threshold'])
    
    ai_insights = generate_ai_insights(df_clean, user_context)


    return jsonify({
        "correlation_matrix": correlation_matrix,
        "ai_insights": ai_insights,
        "data_quality": data_quality,

        "processed_columns": {
            "numeric": num_cols,
            "categorical": cat_cols
        },
        "redirect": url_for('results')
    })




if __name__ == "__main__":
    app.run(debug=True)
