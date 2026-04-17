import joblib
import numpy as np
from pathlib import Path

# semantic v2
from step18_semantic_embeddings_v2 import semantic_scores

# -----------------------------
# Paths & models
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
career_model = joblib.load(MODEL_DIR / "career_model.pkl")

# IMPORTANT: new fusion model
fusion_model = joblib.load(MODEL_DIR / "fusion_model_v2.pkl")

# -----------------------------
# Hybrid prediction (v2)
# -----------------------------
def hybrid_predict(text: str, top_k_semantic: int = 5):
    # ----- TF-IDF prediction -----
    X = vectorizer.transform([text])
    tfidf_probs = career_model.predict_proba(X)[0]
    tfidf_labels = career_model.classes_

    tfidf_scores = dict(zip(tfidf_labels, tfidf_probs))

    # ----- Semantic scores (v2) -----
    semantic_data = semantic_scores(text)

    # rank by semantic similarity
    semantic_ranked = sorted(
        semantic_data.items(),
        key=lambda x: x[1]["semantic_score"],
        reverse=True
    )

    # semantic gating
    allowed_careers = {
        career for career, _ in semantic_ranked[:top_k_semantic]
    }

    # ----- Learned fusion (3 features) -----
    final_results = []

    for career in allowed_careers:
        tfidf_score = tfidf_scores.get(career, 0.0)

        semantic_score = semantic_data[career]["semantic_score"]
        expected_popularity = semantic_data[career]["expected_popularity"]

        # popularity alignment (same formula used in training)
        # predicted popularity is career expectation
        pop_alignment = 1.0  # neutral baseline
        # (we do NOT have user preference here yet, so alignment = 1)

        features = np.array([[tfidf_score, semantic_score, pop_alignment]])
        fusion_score = fusion_model.predict_proba(features)[0][1]

        final_results.append({
            "career": career,
            "fusion_score": float(fusion_score),
            "tfidf_score": round(tfidf_score, 4),
            "semantic_score": round(semantic_score, 4),
            "expected_popularity": expected_popularity
        })

    # rank by fusion score
    final_results.sort(
        key=lambda x: x["fusion_score"],
        reverse=True
    )

    return final_results


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    test_text = (
        "I loved performing for people but also dreamed of building something big "
        "and leading others"
    )

    result = hybrid_predict(test_text)

    print("\n🧠 HYBRID AI OUTPUT (v2 - FUSION MODEL v2)\n")
    for r in result[:5]:
        print(
            r["career"],
            "| fusion:", round(r["fusion_score"], 3),
            "| tfidf:", r["tfidf_score"],
            "| semantic:", r["semantic_score"],
            "| popularity:", r["expected_popularity"]
        )
