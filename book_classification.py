import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# loads the JSON file
with open('books.json', 'r') as f:
    data = json.load(f)

# makes a list of all the synopses in the dataset
X = [book["synopsis"] for book in data]
# makes a list of all the genres in the dataset
y = [book["genre"] for book in data]

# splits the data into two types: training and testing
# training allows the program to learn about the dataset, and testing quizzes it
# test_size tells us how much of the data we want to be used for testing
# random_state shuffles the data. leaving it constant is for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

# converts words into data the program can understand
vectorizer = TfidfVectorizer()

# learns what words are important and turns them into numbers
X_train_tfidf = vectorizer.fit_transform(X_train)
# applies what it learned
X_test_tfidf = vectorizer.transform(X_test)    

# creates the brain
# max_iter is so it is able to learn more accurately
model = LogisticRegression(max_iter = 1000)  
# trains the AI
model.fit(X_train_tfidf, y_train)  

# predicts the genre
prediction = model.predict(X_test_tfidf)

# interactivity
while True:
    # gets the synopsis
    user_input = input("\nType a book description (or 'quit' to stop): ")
    if user_input.lower() == "quit":
        break
    # turns the user_input into a number
    user_tfidf = vectorizer.transform([user_input])
    # predicts the genre using TFIDF
    prediction = model.predict(user_tfidf)
    # prints the result
    print("Predicted genre:", prediction[0])