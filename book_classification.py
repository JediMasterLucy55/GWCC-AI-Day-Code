from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer()

data = "books.json"

matrix = vectorizer.fit_transform(data)

print(matrix.toArray())