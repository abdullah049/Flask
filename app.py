from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os

app = Flask(__name__)
CORS(app)  # This fixes the Flutter web error!

# Load and train model
data = pd.read_csv('phishing.csv')
y = data['Label']
X = data.drop(['Domain', 'Label'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)
feature_columns = X_train.columns.tolist()

def extract_features(url):
    features = {col: 0 for col in feature_columns}
    features['Have_IP'] = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0
    features['Have_At'] = 1 if '@' in url else 0
    features['URL_Length'] = len(url)
    features['URL_Depth'] = url.count('/')
    features['Redirection'] = 1 if '//' in url[url.find('//') + 2:] else 0
    features['https_Domain'] = 1 if urlparse(url).scheme == 'https' else 0
    features['TinyURL'] = 1 if any(x in url for x in ['bit.ly', 'tinyurl', 'goo.gl']) else 0
    features['Prefix/Suffix'] = 1 if '-' in urlparse(url).netloc else 0
    return pd.DataFrame([features])

@app.route('/predict', methods=['GET'])
def predict():
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'No URL'}), 400
    features = extract_features(url)
    pred = tree.predict(features)[0]
    return jsonify({'prediction': int(pred)})

@app.route('/')
def home():
    return "URL Safety Checker API - Use /predict?url=..."

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
