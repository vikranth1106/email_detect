import streamlit as st
import pandas as pd
import numpy as np
import pickle
import spacy
import re
import nltk
from tensorflow.keras.models import load_model

# Download necessary resources
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Load models and vectorizer
spam_model = load_model("models/spam_model.keras")
cat_model = load_model("models/category_model.keras")

with open("vectorizer/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

category_labels = ["Work", "Education", "Promotions"]

# Email cleaning function
def clean_email(text):
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", '', text.lower())
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

# Streamlit UI
st.title("📧 Smart Mail AI")
st.write("🔍 Enter an email to detect if it's spam and classify its category.")

subject = st.text_input("Email Subject")
body = st.text_area("Email Body")

if st.button("Analyze"):
    if not subject.strip() or not body.strip():
        st.warning("Please enter both subject and body.")
    else:
        full_text = subject + " " + body
        cleaned = clean_email(full_text)
        vector = vectorizer.transform([cleaned]).toarray()

        spam_pred = spam_model.predict(vector)[0][0]
        cat_pred = cat_model.predict(vector)[0]
        category = category_labels[np.argmax(cat_pred)]

        st.subheader("📊 Analysis Result")
        st.write("🛡️ Spam:", "Spam" if spam_pred > 0.5 else "Not Spam", f"({round(spam_pred * 100, 2)}%)")
        st.write("📂 Category:", category, f"({round(np.max(cat_pred) * 100, 2)}%)")
