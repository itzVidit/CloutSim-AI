import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "dreams_validated.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# -----------------------------
# Parameters
# -----------------------------
CHUNK_SIZE = 100_000
MAX_FEATURES = 200_000

# -----------------------------
# Initialize vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    min_df=5
)

# -----------------------------
# FIRST PASS: Fit vectorizer
# -----------------------------
print("🔧 Fitting TF-IDF vectorizer (streaming)...")

text_iter = (
    text
    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE)
    for text in chunk["text"].astype(str)
)

vectorizer.fit(text_iter)

print("✅ Vocabulary size:", len(vectorizer.vocabulary_))

# -----------------------------
# Initialize models
# -----------------------------
career_model = MultinomialNB()
popularity_model = MultinomialNB()

career_classes = None
popularity_classes = None

# -----------------------------
# SECOND PASS: Train models
# -----------------------------
print("🚀 Training models (partial_fit)...")

total_rows = 0

for i, chunk in enumerate(pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE)):
    X = vectorizer.transform(chunk["text"].astype(str))

    y_career = chunk["career_primary"]
    y_popularity = chunk["popularity"]

    if i == 0:
        career_classes = y_career.unique()
        popularity_classes = y_popularity.unique()

        career_model.partial_fit(
            X, y_career, classes=career_classes
        )
        popularity_model.partial_fit(
            X, y_popularity, classes=popularity_classes
        )
    else:
        career_model.partial_fit(X, y_career)
        popularity_model.partial_fit(X, y_popularity)

    total_rows += len(chunk)
    print(f"Trained on {total_rows:,} rows")

# -----------------------------
# Save models
# -----------------------------
joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
joblib.dump(career_model, MODEL_DIR / "career_model.pkl")
joblib.dump(popularity_model, MODEL_DIR / "popularity_model.pkl")

print("\n🎉 TRAINING COMPLETE")
print("Models saved to:", MODEL_DIR)
