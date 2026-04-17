"""
Enhanced Fusion Model Training v3
Features:
- Multi-signal fusion (TF-IDF, semantic, context, metadata)
- Intelligent feature engineering
- Balanced sampling strategy
- Context-aware scoring
- Career similarity consideration
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import json

from step18b_semantic_embeddings_v3 import semantic_scores, get_similar_careers

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "dreams_validated_enhanced.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Training parameters
SAMPLE_SIZE = 100_000  # Larger sample for better training
POSITIVE_RATIO = 0.5  # Balanced dataset
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -----------------------------
# Load Base Models
# -----------------------------
print("📦 Loading base models...")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
career_model = joblib.load(MODEL_DIR / "career_model.pkl")
popularity_model = joblib.load(MODEL_DIR / "popularity_model.pkl")
motivation_model = joblib.load(MODEL_DIR / "motivation_model.pkl")
age_model = joblib.load(MODEL_DIR / "age_model.pkl")
intensity_model = joblib.load(MODEL_DIR / "intensity_model.pkl")
personality_model = joblib.load(MODEL_DIR / "personality_model.pkl")

print("✅ Base models loaded")

# -----------------------------
# Feature Engineering
# -----------------------------
def extract_advanced_features(text, true_career, sample_row):
    """
    Extract comprehensive feature set for fusion model
    """
    features = {}
    
    # 1. TF-IDF Predictions
    X = vectorizer.transform([text])
    
    # Career probabilities
    career_probs = career_model.predict_proba(X)[0]
    career_labels = career_model.classes_
    career_scores = dict(zip(career_labels, career_probs))
    features["tfidf_career"] = career_scores.get(true_career, 0.0)
    features["tfidf_max"] = max(career_probs)
    features["tfidf_entropy"] = -np.sum(career_probs * np.log(career_probs + 1e-10))
    
    # Popularity prediction
    pop_probs = popularity_model.predict_proba(X)[0]
    features["tfidf_popularity"] = max(pop_probs)
    
    # 2. Semantic Scores
    sem_data = semantic_scores(text, include_metadata=True)
    
    if true_career in sem_data:
        features["semantic_score"] = sem_data[true_career]["semantic_score"]
        features["keyword_matches"] = sem_data[true_career]["keyword_matches"]
        expected_pop = sem_data[true_career]["expected_popularity"]
    else:
        features["semantic_score"] = 0.0
        features["keyword_matches"] = 0
        expected_pop = 3
    
    # 3. Popularity Alignment
    sample_popularity = sample_row["popularity"]
    features["pop_alignment"] = 1.0 - abs(sample_popularity - expected_pop) / 4.0
    features["pop_exact_match"] = 1.0 if sample_popularity == expected_pop else 0.0
    
    # 4. Motivation Alignment
    motivation_probs = motivation_model.predict_proba(X)[0]
    features["motivation_confidence"] = max(motivation_probs)
    
    # 5. Context Features
    age_probs = age_model.predict_proba(X)[0]
    features["age_confidence"] = max(age_probs)
    
    intensity_probs = intensity_model.predict_proba(X)[0]
    features["intensity_confidence"] = max(intensity_probs)
    
    personality_probs = personality_model.predict_proba(X)[0]
    features["personality_confidence"] = max(personality_probs)
    
    # 6. Text Features
    features["text_length"] = len(text)
    features["word_count"] = len(text.split())
    features["has_contextual"] = sample_row.get("has_context", 0)
    
    # 7. Career Similarity (is it similar to top predictions?)
    top_tfidf_career = career_labels[np.argmax(career_probs)]
    if top_tfidf_career != true_career:
        similar_careers = dict(get_similar_careers(top_tfidf_career, top_k=10))
        features["career_similarity"] = similar_careers.get(true_career, 0.0)
    else:
        features["career_similarity"] = 1.0
    
    # 8. Compound Features
    features["tfidf_semantic_product"] = features["tfidf_career"] * features["semantic_score"]
    features["confidence_product"] = (
        features["tfidf_career"] * 
        features["semantic_score"] * 
        features["pop_alignment"]
    )
    
    return features

# -----------------------------
# Data Sampling Strategy
# -----------------------------
print("\n📊 Loading and sampling data...")

# Load full dataset sample
df_full = pd.read_csv(DATA_FILE, nrows=SAMPLE_SIZE * 2)

print(f"Loaded {len(df_full):,} rows")

# -----------------------------
# Generate Training Data
# -----------------------------
print("\n🔧 Generating training features...")

X_features = []
y_labels = []

# POSITIVE SAMPLES (correct career matches)
pos_sample_size = int(SAMPLE_SIZE * POSITIVE_RATIO)
df_positive = df_full.sample(n=pos_sample_size, random_state=RANDOM_STATE)

print(f"Processing {pos_sample_size:,} positive samples...")
for idx, row in df_positive.iterrows():
    text = row["text"]
    true_career = row["career_primary"]
    
    features = extract_advanced_features(text, true_career, row)
    X_features.append(list(features.values()))
    y_labels.append(1)
    
    if (len(y_labels) % 10000) == 0:
        print(f"  Positive samples: {len(y_labels):,}")

# NEGATIVE SAMPLES (wrong career matches)
neg_sample_size = int(SAMPLE_SIZE * (1 - POSITIVE_RATIO))
df_negative = df_full.sample(n=neg_sample_size, random_state=RANDOM_STATE + 1)

print(f"Processing {neg_sample_size:,} negative samples...")
all_careers = career_model.classes_

for idx, row in df_negative.iterrows():
    text = row["text"]
    true_career = row["career_primary"]
    
    # Select a wrong career (not the true one)
    wrong_career = np.random.choice(
        [c for c in all_careers if c != true_career]
    )
    
    features = extract_advanced_features(text, wrong_career, row)
    X_features.append(list(features.values()))
    y_labels.append(0)
    
    if (len(y_labels) % 10000) == 0:
        print(f"  Total samples: {len(y_labels):,}")

# Convert to arrays
X = np.array(X_features)
y = np.array(y_labels)

feature_names = list(extract_advanced_features(
    df_full.iloc[0]["text"], 
    df_full.iloc[0]["career_primary"],
    df_full.iloc[0]
).keys())

print(f"\n✅ Feature matrix: {X.shape}")
print(f"   Features: {len(feature_names)}")
print(f"   Positive samples: {sum(y)}")
print(f"   Negative samples: {len(y) - sum(y)}")

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain size: {len(X_train):,}")
print(f"Test size:  {len(X_test):,}")

# -----------------------------
# Train Fusion Model
# -----------------------------
print("\n🚀 Training Gradient Boosting fusion model...")

fusion_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=RANDOM_STATE,
    verbose=1
)

fusion_model.fit(X_train, y_train)

print("✅ Training complete")

# -----------------------------
# Evaluation
# -----------------------------
print("\n📈 Model Evaluation:")

y_pred = fusion_model.predict(X_test)
y_pred_proba = fusion_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {auc:.4f}")

# Feature importance
feature_importance = sorted(
    zip(feature_names, fusion_model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)

print("\n🔍 Top 10 Feature Importances:")
for feat, imp in feature_importance[:10]:
    print(f"  {feat:30s} {imp:.4f}")

# -----------------------------
# Save Model
# -----------------------------
print("\n💾 Saving fusion model...")

joblib.dump(fusion_model, MODEL_DIR / "fusion_model_v3.pkl")

# Save metadata
metadata = {
    "sample_size": SAMPLE_SIZE,
    "num_features": len(feature_names),
    "feature_names": feature_names,
    "feature_importance": {feat: float(imp) for feat, imp in feature_importance},
    "test_auc": float(auc),
    "positive_ratio": POSITIVE_RATIO
}

with open(MODEL_DIR / "fusion_metadata_v3.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Fusion model v3 saved")
print(f"   Location: {MODEL_DIR / 'fusion_model_v3.pkl'}")
print(f"   Test AUC: {auc:.4f}")
print("\n" + "=" * 70)