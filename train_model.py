# train_model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "data/news.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "news_classifier.joblib")

SAMPLE_DATA = [
    # Business
    ("Stocks rallied after the central bank signaled lower interest rates.", "Business"),
    ("Company announced quarterly earnings beating analyst expectations.", "Business"),
    ("Mergers and acquisition activity picked up in the tech sector.", "Business"),
    ("Inflation cooled this month, lifting investor confidence.", "Business"),
    ("New startup raised Series A funding to expand operations.", "Business"),

    # Sports
    ("The striker scored a hat-trick to win the derby match.", "Sports"),
    ("Olympic committee revealed the schedule for the next games.", "Sports"),
    ("Coach praised team for disciplined defense and quick counters.", "Sports"),
    ("Tennis champion defended her title in three sets.", "Sports"),
    ("Local club won the regional under-18 tournament.", "Sports"),

    # Tech
    ("New smartphone launches with a powerful chipset and OLED display.", "Tech"),
    ("Open-source project released a major update with performance improvements.", "Tech"),
    ("AI startup unveiled a model that translates speech in real-time.", "Tech"),
    ("Cybersecurity researchers disclosed a critical vulnerability in the library.", "Tech"),
    ("Cloud provider announced lower prices for GPU instances.", "Tech"),
]

def ensure_data():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("No dataset found — creating a small sample dataset at", DATA_PATH)
        df = pd.DataFrame(SAMPLE_DATA, columns=["text", "label"])
        df.to_csv(DATA_PATH, index=False)
        print("Sample dataset created with", len(df), "rows.")

def train():
    ensure_data()
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=500, solver="liblinear"))
    ])

    param_grid = {
        "tfidf__ngram_range": [(1,1), (1,2)],
        "clf__C": [0.5, 1.0, 5.0]
    }

    print("Starting GridSearchCV (this should be fast for small dataset)...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring="f1_weighted", verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    y_pred = best.predict(X_test)
    print("\n=== Classification Report ===\n")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    train()
