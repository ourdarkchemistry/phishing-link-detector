import pandas as pd
import re
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_urls(file):
    try:
        with open(file, 'r') as f:
            urls = f.read().splitlines()
        return pd.DataFrame(urls, columns=['url'])
    except Exception as e:
        print(f"Error loading URL file: {e}")
        return None

def extract_features(df):
    df['url_length'] = df['url'].apply(len)
    df['num_subdomains'] = df['url'].apply(lambda x: x.count('.') - 1)
    df['has_https'] = df['url'].apply(lambda x: 1 if "https" in x else 0)
    df['has_suspicious_words'] = df['url'].apply(lambda x: 1 if re.search(r'(secure|login|free|gift|win|claim)', x) else 0)
    return df[['url_length', 'num_subdomains', 'has_https', 'has_suspicious_words']]

def train_model():
    data = {
        'url_length': [20, 25, 35, 45, 60, 15, 10, 55],
        'num_subdomains': [1, 2, 1, 2, 3, 1, 0, 3],
        'has_https': [1, 1, 0, 0, 0, 1, 1, 0],
        'has_suspicious_words': [0, 1, 1, 1, 1, 0, 0, 1],
        'label': [0, 1, 1, 1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    X = df[['url_length', 'num_subdomains', 'has_https', 'has_suspicious_words']]
    y = df['label']
    
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def main():
    if len(sys.argv) < 3 or sys.argv[1] != '--file':
        print("Usage: python detector.py --file <path_to_url_file>")
        return

    file = sys.argv[2]
    df = load_urls(file)

    if df is not None:
        print("Extracting features from URLs...")
        features = extract_features(df)

        print("Loading classification model...")
        model = train_model()
        
        print("Detecting phishing URLs...")
        predictions = model.predict(features)
        df['prediction'] = predictions
        df['prediction'] = df['prediction'].apply(lambda x: 'Phishing' if x == 1 else 'Safe')
        
        print(df[['url', 'prediction']])

if __name__ == "__main__":
    main()
