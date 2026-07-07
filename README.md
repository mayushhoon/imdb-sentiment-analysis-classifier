# IMDB Movie Review Sentiment Analysis

This project demonstrates a sentiment analysis model built using various machine learning algorithms to classify IMDB movie reviews as either positive or negative.

## Project Overview

The goal of this project is to build and evaluate different classification models for sentiment analysis on a dataset of IMDB movie reviews. The pipeline involves:

1.  **Data Loading and Initial Exploration**: Loading the dataset and understanding its structure.
2.  **Data Cleaning**: Handling duplicate entries.
3.  **Text Preprocessing**: Removing HTML tags, special characters, converting text to lowercase, performing stop word removal, and tokenization.
4.  **Feature Extraction**: Converting text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
5.  **Model Training and Evaluation**: Training and evaluating multiple classification models (Logistic Regression, Multinomial Naive Bayes, Linear SVM, Decision Tree, Random Forest, K-Nearest Neighbors).
6.  **Model Saving**: Saving the best performing model (Logistic Regression) and the TF-IDF vectorizer for deployment.
7.  **Streamlit Deployment**: Creating a web application using Streamlit to interactively classify new movie reviews.

## Files in this Repository

*   `IMDB Dataset.csv`: The original dataset containing movie reviews and their sentiments.
*   `logistic_regression_model.pkl`: The trained Logistic Regression model, saved in pickle format.
*   `tfidf_vectorizer.pkl`: The fitted TF-IDF vectorizer, saved in pickle format.
*   `app.py`: The Streamlit application script for the web interface.
*   `requirements.txt`: A list of Python dependencies required to run the project.
*   `README.md`: This README file.

## How to Run Locally

To run this project on your local machine, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd imdb-sentiment-analysis
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data (if not already present)**:
    The `app.py` script attempts to download necessary NLTK data. If you encounter issues, you might need to manually download them:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    ```

5.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

    This will open the application in your web browser.

## Deployment on Streamlit Community Cloud

To deploy this application on Streamlit Community Cloud:

1.  **Push your project to a GitHub repository** (ensuring all necessary files like `app.py`, `requirements.txt`, `logistic_regression_model.pkl`, `tfidf_vectorizer.pkl`, and `IMDB Dataset.csv` are included).
2.  Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3.  Click on "New app" and select your GitHub repository and branch.
4.  Specify `app.py` as the main file path.
5.  Click "Deploy!"

Your application will be deployed and accessible via a public URL.
