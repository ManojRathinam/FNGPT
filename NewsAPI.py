import os
from apikey import APIKEY
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from textblob import TextBlob

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = APIKEY

# Load your dataset (replace 'your_dataset.csv' with your dataset file)
data = pd.read_csv("news_dataset.csv", encoding='ISO-8859-1')

# Replace missing values in the 'text' column with an empty string
data['text'].fillna('', inplace=True)

# Split the dataset into features (X) and labels (y)
X = data['text']
y = data['label']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train-Test Split for TF-IDF Model
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier with TF-IDF
tfidf_classifier = MultinomialNB()
tfidf_classifier.fit(X_train_tfidf, y_train)
y_pred_tfidf = tfidf_classifier.predict(X_test_tfidf)

# App Framework
st.title("Fake News Detection")
news_prompt = st.text_area("Enter the News Article:")

# Prompt Template
template = PromptTemplate(
    input_variables=['news'],
    template="{news}"
)

# Memory
memory = ConversationBufferMemory(input_key='news', memory_key='chat history')

# Create an OpenAI-powered Langchain instance
llm = OpenAI(temperature=0.3)

# Fake News Detection Chain
chain = LLMChain(llm=llm, prompt=template,
                 output_key='response', memory=memory)

# Function to classify news with TF-IDF


def classify_news_tfidf(news_text):
    news_tfidf = tfidf_vectorizer.transform([news_text])
    result_tfidf = tfidf_classifier.predict(news_tfidf)
    return result_tfidf[0]

# Lie detection function using PassiveAggressiveClassifier


def news_detection(news_text):
    pac = PassiveAggressiveClassifier()
    pac.partial_fit(X_train_tfidf, y_train, classes=['FAKE', 'REAL'])
    news_tfidf = tfidf_vectorizer.transform([news_text])
    lie_result = pac.predict(news_tfidf)
    return lie_result[0]


if st.button("Detect Fake News"):
    if news_prompt:

        # TF-IDF model response
        tfidf_result = classify_news_tfidf(news_prompt)

        # Lie detection response
        lie_result = news_detection(news_prompt)

        # Langchain
        langchain_response = chain({'news': news_prompt})

        if tfidf_result != lie_result:  # If TF-IDF and PassiveAggressive disagree
            # Langchain (GPT) response for sentiment analysis
            sentiment_analysis = TextBlob(langchain_response['response'])
            sentiment = sentiment_analysis.sentiment.polarity

            if sentiment < 0:  # If sentiment is negative, consider it as "FAKE"
                sentiment_label = "FAKE"
            else:
                sentiment_label = "REAL"

            st.write("News Detection (Sentiment Analysis):", sentiment_label)

            # Model evaluation for sentiment analysis
            accuracy_sentiment = 1 if sentiment_label == y_test.iloc[0] else 0
            classification_rep_sentiment = classification_report(
                [y_test.iloc[0]], [sentiment_label], target_names=['REAL', 'FAKE'])

            st.header("Model Evaluation (Sentiment Analysis)")
            st.write("Accuracy (Sentiment Analysis):", accuracy_sentiment)
            st.text(classification_rep_sentiment)
        else:
            st.write("News Detection (TF-IDF): ", tfidf_result)

        # Provide additional information using Langchain response
        st.write("More Information (Langchain):")
        st.write(langchain_response['response'])

        # Calculate accuracy and display classification report for the TF-IDF model
        st.header("Model Evaluation (TF-IDF Model)")
        accuracy = accuracy_score(y_test, y_pred_tfidf)
        st.write("Accuracy (TF-IDF Model):", accuracy)
        classification_rep = classification_report(
            y_test, y_pred_tfidf, target_names=['REAL', 'FAKE'])
        st.text(classification_rep)
