from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load models and preprocessing objects
try:
    # Logistic Regression components
    lr_model = joblib.load('logistic_regression_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    
    # LSTM components
    lstm_model = load_model('lstm_model.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Constants for LSTM
    max_len = 200
    
    # Initialize text cleaning
    nltk.download('stopwords')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    print("All models and components loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = text.split()
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Join back to string
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input
            review = request.form['review']
            model_type = request.form['model_type']
            
            # Clean the review
            cleaned_review = clean_text(review)
            
            if model_type == 'lr':
                # Logistic Regression prediction
                features = tfidf.transform([cleaned_review])
                prediction = lr_model.predict(features)[0]
                confidence = np.max(lr_model.predict_proba(features))
                model_name = "Logistic Regression"
            else:
                # LSTM prediction
                sequence = tokenizer.texts_to_sequences([cleaned_review])
                padded = pad_sequences(sequence, maxlen=max_len)
                prediction = (lstm_model.predict(padded) > 0.5).astype("int32")[0][0]
                confidence = float(lstm_model.predict(padded)[0][0])
                if prediction == 0:
                    confidence = 1 - confidence
                model_name = "LSTM"
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence_percent = round(confidence * 100, 2)
            
            return render_template('index.html', 
                                 review=review,
                                 cleaned_review=cleaned_review,
                                 model_used=model_name,
                                 sentiment=sentiment,
                                 confidence=confidence_percent)
            
        except Exception as e:
            return render_template('index.html', 
                                 error_message=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)