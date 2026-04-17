import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# updated semantic embeddings
from step18_semantic_embeddings_v2 import semantic_scores

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "dreams_validated.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load base models
# -----------------------------
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
career_model = joblib.load(MODEL_DIR / "career_model.pkl")

# -----------------------------
# Training config
# -----------------------------
POS_SAMPLES = 30_000
NEG_SAMPLES = 30_000

# -----------------------------
# Load data (sample)
# -----------------------------
df = pd.read_csv(DATA_FILE, nrows=POS_SAMPLES)

X_meta = []
Y_meta = []

# -----------------------------
# POSITIVE SAMPLES
# -----------------------------
for _, row in df.iterrows():
    text = row["text"]
    true_career = row["career_primary"]
    sample_popularity = row["popularity"]

    # TF-IDF
    X = vectorizer.transform([text])
    tfidf_probs = career_model.predict_proba(X)[0]
    tfidf_labels = career_model.classes_
    tfidf_score = dict(zip(tfidf_labels, tfidf_probs)).get(true_career, 0.0)

    # Semantic + expected popularity
    sem_data = semantic_scores(text)
    sem_score = sem_data[true_career]["semantic_score"]
    expected_popularity = sem_data[true_career]["expected_popularity"]

    # Popularity alignment
    pop_align = 1 - abs(sample_popularity - expected_popularity) / 4

    X_meta.append([tfidf_score, sem_score, pop_align])
    Y_meta.append(1)

# -----------------------------
# NEGATIVE SAMPLES
# -----------------------------
df_neg = df.sample(NEG_SAMPLES, random_state=42)

for _, row in df_neg.iterrows():
    text = row["text"]
    wrong_career = np.random.choice(career_model.classes_)
    sample_popularity = row["popularity"]

    X = vectorizer.transform([text])
    tfidf_probs = career_model.predict_proba(X)[0]
    tfidf_labels = career_model.classes_
    tfidf_score = dict(zip(tfidf_labels, tfidf_probs)).get(wrong_career, 0.0)

    sem_data = semantic_scores(text)
    sem_score = sem_data.get(wrong_career, {}).get("semantic_score", 0.0)
    expected_popularity = sem_data.get(wrong_career, {}).get(
        "expected_popularity", 3
    )

    pop_align = 1 - abs(sample_popularity - expected_popularity) / 4

    X_meta.append([tfidf_score, sem_score, pop_align])
    Y_meta.append(0)

# -----------------------------
# Train fusion model
# -----------------------------
X_meta = np.array(X_meta)
Y_meta = np.array(Y_meta)

fusion_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

fusion_model.fit(X_meta, Y_meta)

# -----------------------------
# Save model
# -----------------------------
joblib.dump(fusion_model, MODEL_DIR / "fusion_model_v2.pkl")

print("\n✅ Fusion Model v2 trained")
print("TF-IDF weight:", round(fusion_model.coef_[0][0], 4))
print("Semantic weight:", round(fusion_model.coef_[0][1], 4))
print("Popularity alignment weight:", round(fusion_model.coef_[0][2], 4))
