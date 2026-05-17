# My-Ninth-Repository-
Create an Email Classification System(Spam,Promotion,Social,Important)
# ==========================================
# FILE: train_model.py
# ==========================================

import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# DOWNLOAD NLTK RESOURCES
# ==========================================

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("dataset.csv")

print("\n========== DATASET ==========")
print(df.head())

# ==========================================
# TEXT PREPROCESSING
# ==========================================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

print("\n========== CLEANED TEXT ==========")
print(df[['text', 'clean_text']].head())

# ==========================================
# FEATURE ENGINEERING
# ==========================================

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['clean_text'])

y = df['label']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MODEL TRAINING
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\n========== MODEL TRAINED ==========")

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== ACCURACY ==========")
print("Accuracy:", accuracy)

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================
# TOP KEYWORDS PER CATEGORY
# ==========================================

feature_names = vectorizer.get_feature_names_out()

for i, category in enumerate(model.classes_):

    top10 = np.argsort(model.coef_[i])[-10:]

    print(f"\nTop keywords for {category}:")

    for j in top10:
        print(feature_names[j])

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n========== MODEL SAVED ==========")