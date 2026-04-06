import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# loads the JSON file
with open('books.json', 'r') as f:
    data = json.load(f)

X = [book["synopsis"] for book in data]
y = [book["genre"] for book in data]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)    

model = LogisticRegression(max_iter=1000)  
model.fit(X_train_tfidf, y_train)  

prediction = model.predict(X_test_tfidf)

while True:
    user_input = input("\nType a book description (or 'quit' to stop): ")
    if user_input.lower() == "quit":
        break
    user_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_tfidf)
    print("Predicted genre:", prediction[0])