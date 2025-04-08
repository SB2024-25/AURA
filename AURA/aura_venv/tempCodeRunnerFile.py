import os
import pandas as pd
import ollama
from flask import Flask, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_dataset(file_path):
    """Loads a CSV file and extracts numeric & categorical columns."""
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        return df, numeric_cols, categorical_cols
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, [], []

def compute_correlations(df, numeric_cols):
    """Computes correlation matrix for numeric features."""
    return df[numeric_cols].corr().to_dict()

def generate_ai_insights(df, numeric_cols, categorical_cols):
    """Generates AI-driven insights using Mistral via Ollama."""
    prompt = f"""
    Analyze the following dataset summary:
    Numeric columns: {numeric_cols}
    Categorical columns: {categorical_cols}
    Basic statistics: {df.describe().to_string()}
    Identify key trends, anomalies, and relationships.
    """
    response = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    df, numeric_cols, categorical_cols = load_dataset(file_path)
    if df is None:
        return jsonify({"error": "Invalid CSV file"}), 400

    return jsonify({
        "message": "File uploaded successfully",
        "filename": file.filename,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    })

@app.route('/correlation', methods=['GET'])
def correlation():
    file_name = request.args.get('file')
    if not file_name:
        return jsonify({"error": "File parameter missing"}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    df, numeric_cols, _ = load_dataset(file_path)
    if df is None:
        return jsonify({"error": "Invalid CSV file"}), 400

    correlations = compute_correlations(df, numeric_cols)
    return jsonify({"correlations": correlations})

@app.route('/insights', methods=['GET'])
def insights():
    file_name = request.args.get('file')
    if not file_name:
        return jsonify({"error": "File parameter missing"}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    df, numeric_cols, categorical_cols = load_dataset(file_path)
    if df is None:
        return jsonify({"error": "Invalid CSV file"}), 400

    insights = generate_ai_insights(df, numeric_cols, categorical_cols)
    return jsonify({"insights": insights})

if __name__ == "__main__":
    app.run(debug=True)
