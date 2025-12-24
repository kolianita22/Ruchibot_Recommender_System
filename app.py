
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib, os, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Load saved model and data
tfidf = joblib.load("vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
recipes = pd.read_csv("cleaned_reciped.csv")

# Create Flask app
app = Flask(__name__)

# Function for recommendation
"""def recommend_recipe(user_input, top_n=5):
    user_vec = tfidf.transform([user_input])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return recipes.iloc[top_indices][['title', 'ingredients_regional','ingredients', 'cuisine', 'diet', 'difficulty']]"""
import re

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation, numbers, and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize (convert words to their base form)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join back into a single string
    return " ".join(tokens)

recipes = pd.read_csv("cleaned_reciped.csv")
recipes.dropna(subset=['ingredients', 'instructions'], inplace=True)
recipes['clean_ingredients'] = recipes['ingredients'].apply(preprocess_text)
recipes['clean_instructions'] = recipes['instructions'].apply(preprocess_text)

# Combine them
recipes['combined_features'] = recipes['clean_ingredients'] + " " + recipes['clean_instructions']


def recommend_recipe(user_input, top_n=5):
    clean_input = preprocess_text(user_input)
    user_vec = tfidf.transform([clean_input])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    recipes['score'] = sim_scores
    top_indices = recipes['score'].argsort()[-top_n:][::-1]
    return recipes.iloc[top_indices][['title', 'ingredients_regional','ingredients', 'cuisine', 'diet','seasons','score']]



# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Chat endpoint
@app.route('/get', methods=['POST'])
def chat():
    user_input = request.form['msg']
    
    results = recommend_recipe(user_input)
    reply = ""
    for _, r in results.iterrows():
        reply += f"🍲 <b>{r['title']}</b><b>ingredients:</b>{r['ingredients']}<br><b>ingredients_regional:</b>{r['ingredients_regional']}<br>Cuisine: {r['cuisine']}<br>Diet: {r['diet']}<br><b>seasons:{r['seasons']}<br>"
    return reply

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
