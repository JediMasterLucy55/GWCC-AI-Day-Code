import json
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


# ── CONFIG ────────────────────────────────────────────────────────────────────

JSON_FILE = "books.json"        # path to your dataset
MODEL_FILE = "book_model.pkl"   # where the trained model gets saved


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_data(filepath):
    """Load and validate the JSON dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find '{filepath}'.\n"
            "Make sure your JSON file is in the same folder as this script."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    required_fields = {"title", "synopsis", "genre"}
    missing = required_fields - set(df.columns)
    if missing:
        raise ValueError(f"Your JSON is missing these fields: {missing}")

    print(f"✔ Loaded {len(df)} books")
    print(f"✔ Genres found: {sorted(df['genre'].unique())}\n")

    return df


# ── PREPROCESS ────────────────────────────────────────────────────────────────

def preprocess(df):
    """Clean labels and combine title + synopsis into one text field."""
    df = df.copy()

    # Normalize genre labels
    df["genre"] = df["genre"].str.lower().str.strip()

    # Combine title and synopsis (title repeated for extra weight)
    df["text"] = df["title"] + " " + df["title"] + " " + df["synopsis"]

    # Drop rows with missing text or genre
    before = len(df)
    df = df.dropna(subset=["text", "genre"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"⚠ Dropped {dropped} rows with missing text or genre.\n")

    return df


# ── TRAIN ─────────────────────────────────────────────────────────────────────

def train(df):
    """Train a TF-IDF + Logistic Regression pipeline."""
    documents = df["text"].tolist()
    labels = df["genre"].tolist()

    # Need at least 2 samples per genre for stratified split
    genre_counts = df["genre"].value_counts()
    rare_genres = genre_counts[genre_counts < 2].index.tolist()
    if rare_genres:
        print(
            f"⚠ These genres have fewer than 2 examples and will be skipped: {rare_genres}\n"
        )
        df = df[~df["genre"].isin(rare_genres)]
        documents = df["text"].tolist()
        labels = df["genre"].tolist()

    if len(df) < 4:
        raise ValueError(
            "Not enough data to train. Add more books to your JSON (aim for 50+ per genre)."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        documents, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams
            max_features=10000,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,                # regularization strength
            solver="lbfgs",
            multi_class="auto",
        )),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    predictions = pipeline.predict(X_test)
    print("\n── Evaluation Results ──────────────────────────────")
    print(classification_report(y_test, predictions))

    print("── Confusion Matrix ────────────────────────────────")
    genres = sorted(set(labels))
    cm = confusion_matrix(y_test, predictions, labels=genres)
    cm_df = pd.DataFrame(cm, index=genres, columns=genres)
    print(cm_df)
    print()

    return pipeline


# ── SAVE / LOAD MODEL ─────────────────────────────────────────────────────────

def save_model(pipeline, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"✔ Model saved to '{filepath}'\n")


def load_model(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved model found at '{filepath}'. Train first.")
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ── PREDICT ───────────────────────────────────────────────────────────────────

def predict(pipeline, title, synopsis):
    """Predict the genre of a single book."""
    text = f"{title} {title} {synopsis}"
    genre = pipeline.predict([text])[0]
    probs = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    print(f"\nTitle:     {title}")
    print(f"Predicted: {genre.upper()}")
    print("\nConfidence scores:")
    for cls, prob in sorted(zip(classes, probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<15} {bar} {prob:.1%}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("       Book Genre Classifier — TF-IDF")
    print("=" * 52 + "\n")

    # 1. Load and preprocess
    df = load_data(JSON_FILE)
    df = preprocess(df)

    # 2. Train and save
    pipeline = train(df)
    save_model(pipeline, MODEL_FILE)

    # 3. Demo predictions on a few books
    #    Replace these with your own to test
    print("── Sample Predictions ──────────────────────────────")

    predict(
        pipeline,
        title="The Lost Planet",
        synopsis="A crew of astronauts crash lands on an unknown world filled with alien creatures and must find a way back to Earth before their oxygen runs out."
    )

    predict(
        pipeline,
        title="Midnight at Thornwood Manor",
        synopsis="A detective is called to a remote estate after the owner is found dead. Every guest has a motive and secrets run deep in the old manor's walls."
    )

    predict(
        pipeline,
        title="The Dragon's Promise",
        synopsis="A young mage must forge an alliance with the last dragon to defeat an ancient darkness threatening to swallow the kingdom whole."
    )

    print("\n" + "=" * 52)
    print("To predict your own book, call:")
    print("  predict(pipeline, title='...', synopsis='...')")
    print("=" * 52)


if __name__ == "__main__":
    main()