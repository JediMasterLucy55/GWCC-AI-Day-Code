import json
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# loads the JSON file
with open('books.json', 'r') as file:
    data = json.load(file)

synopsis = [book["synopsis"] for book in data]
genre = [book["genre"] for book in data]

X_train, Y_train, X_test, Y_test = train_test_split(
    synopsis, genre, train_size = 0.2, random_state = 42
)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)    

model = LogisticRegression(max_iter=1000)  
model.fit(X_train_tfidf, Y_train)  

prediction = model.predict(X_test_tfidf)

while True:
    user_input = input("\nType a book description (or 'quit' to stop): ")
    if user_input.lower() == "quit":
        break
    user_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_tfidf)
    print("Predicted genre:", prediction[0])