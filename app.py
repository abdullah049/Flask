# app.py - Flask app for Safe/Unsafe URL Detection using REST API

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = Flask(__name__)

# Load and prepare the dataset (assuming 'phishing.csv' is in the same directory)
data = pd.read_csv('phishing.csv')

# Separate features and target
y = data['Label']
X = data.drop(['Domain', 'Label'], axis=1)  # Drop 'Domain' as it's not numerical

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Train the Decision Tree model (using max_depth=5 as in the notebook)
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)

# Get the list of feature columns expected by the model
feature_columns = X_train.columns.tolist()

def extract_features(url):
    """
    Extract features from a URL, setting unknown features to 0.
    """
    features = {col: 0 for col in feature_columns}
    
    # Extract known features (as per the notebook)
    features['Have_IP'] = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0
    features['Have_At'] = 1 if '@' in url else 0
    features['URL_Length'] = len(url)
    features['URL_Depth'] = url.count('/')
    features['Redirection'] = 1 if '//' in url[url.find('//') + 2:] else 0
    features['https_Domain'] = 1 if urlparse(url).scheme == 'https' else 0
    features['TinyURL'] = 1 if any(shortener in url for shortener in ['bit.ly', 'tinyurl', 'goo.gl']) else 0
    features['Prefix/Suffix'] = 1 if '-' in urlparse(url).netloc else 0
    
    # Other features like DNS_Record, Web_Traffic, etc., remain 0 as dummies
    
    # Convert to DataFrame for prediction
    return pd.DataFrame([features])

# Route for the homepage (serves the search bar UI)
@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Safe/Unsafe URL Detector</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
            input[type="text"] { width: 400px; padding: 10px; font-size: 16px; }
            button { padding: 10px 20px; font-size: 16px; margin-left: 10px; }
            #result { margin-top: 20px; font-size: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Safe/Unsafe URL Detector</h1>
        <input type="text" id="url" placeholder="Enter URL here...">
        <button onclick="checkURL()">Check</button>
        <div id="result"></div>
        
        <script>
            function checkURL() {
                const url = document.getElementById('url').value;
                if (!url) {
                    alert('Please enter a URL');
                    return;
                }
                fetch(`/predict?url=${encodeURIComponent(url)}`)
                    .then(response => response.json())
                    .then(data => {
                        const resultDiv = document.getElementById('result');
                        if (data.prediction === 1) {
                            resultDiv.innerHTML = '⚠️ This URL is UNSAFE (phishing/malicious)';
                            resultDiv.style.color = 'red';
                        } else {
                            resultDiv.innerHTML = '✅ This URL is SAFE';
                            resultDiv.style.color = 'green';
                        }
                    })
                    .catch(error => console.error('Error:', error));
            }
        </script>
    </body>
    </html>
    ''')

# REST API endpoint for prediction
@app.route('/predict', methods=['GET'])
def predict():
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Extract features
    url_features = extract_features(url)
    
    # Predict (0 = safe, 1 = unsafe/phishing)
    pred = tree.predict(url_features)[0]
    
    return jsonify({'prediction': int(pred)})

if __name__ == '__main__':
    # Render uses PORT env var
    import os
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
