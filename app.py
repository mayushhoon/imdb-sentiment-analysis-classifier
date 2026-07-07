
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present (for Streamlit deployment)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except nltk.downloader.DownloadError:
    nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

# Define the text cleaning function (must be the same as used during training)
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

# Load the trained model and TF-IDF vectorizer
@st.cache_resource
def load_resources():
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, tfidf_vectorizer = load_resources()

st.title("IMDB Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to get its sentiment (Positive/Negative).")

# Text input from user
user_input = st.text_area("Enter your movie review here:", "This movie was absolutely fantastic and I loved every minute of it!")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        cleaned_input = clean_text(user_input)
        stopped_input = remove_stopwords(cleaned_input)
        tokenized_input = word_tokenize(stopped_input)

        # Transform the preprocessed text using the loaded TF-IDF vectorizer
        # Note: TfidfVectorizer expects a list of strings (or a single string for one document)
        # If tokenizer was used in TfidfVectorizer, it expects tokenized input
        # Our TFIDF was initialized with tokenizer=lambda x:x, preprocessor=lambda x:x
        # So, it expects a list of tokens directly.
        input_vector = tfidf_vectorizer.transform([' '.join(tokenized_input)]) # Join tokens back to a string for vectorizer if it expects strings

        # Make prediction
        prediction = model.predict(input_vector)

        # Display result
        if prediction[0] == 1:
            st.success("Sentiment: Positive")
        else:
            st.error("Sentiment: Negative")
    else:
        st.warning("Please enter a movie review to analyze.")
