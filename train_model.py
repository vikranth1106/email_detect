import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pickle
import os

# Load dataset
df = pd.read_csv("smart_mail_dataset.csv")

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text']).toarray()

# Labels
y_spam = df['label_spam']
y_cat = pd.Categorical(df['label_category'])
y_cat_encoded = to_categorical(y_cat.codes)

# Train-test split
X_train, X_test, y_spam_train, y_spam_test = train_test_split(X, y_spam, test_size=0.2, random_state=42)
_, _, y_cat_train, y_cat_test = train_test_split(X, y_cat_encoded, test_size=0.2, random_state=42)

# Spam model
spam_model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(1, activation='sigmoid')
])
spam_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
spam_model.fit(X_train, y_spam_train, epochs=10, verbose=1)

# Category model
cat_model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(y_cat_encoded.shape[1], activation='softmax')
])
cat_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cat_model.fit(X_train, y_cat_train, epochs=10, verbose=1)

# Save everything
os.makedirs("models", exist_ok=True)
os.makedirs("vectorizer", exist_ok=True)

spam_model.save("models/spam_model.keras")
cat_model.save("models/category_model.keras")

with open("vectorizer/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
