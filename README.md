
# 🍛 RuchiBot - Recipe Recommendation Chatbot

RuchiBot is an **intelligent recipe recommendation system** that helps users discover recipes based on ingredients they have. It leverages **Natural Language Processing (NLP)**, **TF-IDF vectorization**, and **cosine similarity** to provide personalized recipe suggestions. The system also supports Marathi translations using Google Translate for regional ingredients.

---

## Features

- 🥘 **Ingredient-based Recommendations**: Enter the ingredients you have, and RuchiBot suggests recipes.
- 🔍 **Top Recipe Details**: Includes:
  - Recipe Name
  - Ingredients (Regional & Standard)
  - Cuisine Type
  - Diet Information
  - Suitable Seasons
- 💬 **Interactive Chatbot Interface**: Built with HTML, CSS, and JavaScript.
- 🌐 **Flask Backend**: Handles recommendation logic and real-time responses.
- 🧠 **NLP-based Preprocessing**:
  - Lowercasing, punctuation removal, tokenization
  - Stopwords removal
  - Lemmatization
- 📊 **Data Cleaning & Preprocessing**: Handles missing values, duplicates, and standardizes regional ingredients.

---

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Python, Flask
- **Data Processing**: Pandas
- **Machine Learning**: Scikit-learn (TF-IDF, Cosine Similarity)
- **NLP**: NLTK (tokenization, lemmatization, stopwords)
- **Translation**: Googletrans (for Marathi <-> English)
- **Model Serialization**: Joblib (`vectorizer.pkl`, `tfidf_matrix.pkl`)

---

## Installation & Setup

1.Clone the repository:

bash
git clone https://github.com/kolianita22/RuchiBot_Recommender_System.git
cd RuchiBot
___
**Create a virtual environment:**

>>python -m venv venv
>>source venv/bin/activate   # Linux/Mac
>>venv\Scripts\activate    # Windows
---

**Install dependencies:**

>>pip install -r requirements.txt

___
**Download NLTK resources:**

>>import nltk
>>nltk.download('punkt')
>>nltk.download('stopwords')
>>nltk.download('wordnet')

___
**Ensure the following files are in the project directory:**

cleaned_reciped.csv (preprocessed recipe dataset)
vectorizer.pkl (saved TF-IDF vectorizer)
tfidf_matrix.pkl (TF-IDF feature matrix)
___
**Run the Flask application:**

>>python app.py
___
**Usage**

1.Open the chatbot interface.

2.Enter ingredients you have (e.g., potatoes, tomatoes, garlic).

3.Press Enter or click Send.

4.RuchiBot returns the top 5 recipe recommendations with detailed information.
___
**Future Enhancements**

a.Add Marathi language support for instructions and ingredients.

b.Include images for recipes in the chatbot responses.

c.Implement voice input/output.

d.Deploy the chatbot on Heroku, AWS, or Azure.


e.Create a dashboard for analytics: top ingredients, popular cuisines, and diet types.
