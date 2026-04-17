import joblib
import numpy as np
from pathlib import Path
from  step12_hybrid_embeddings import sementic_scores

#classical model
BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_DIR=BASE_DIR/"models"

vectorizer=joblib.load(MODEL_DIR/"vectorizer.pkl")
career_model=joblib.load(MODEL_DIR/"career_model.pkl")
fusion_model=joblib.load(MODEL_DIR/"fusion_model.pkl")

def hybrid_predict(text: str, top_k_semantic=3):
    # ----- TF-IDF prediction -----
    X = vectorizer.transform([text])
    tfidf_probs = career_model.predict_proba(X)[0]
    tfidf_labels = career_model.classes_

    tfidf_scores = dict(zip(tfidf_labels, tfidf_probs))

    # ----- Semantic scores -----
    embed_scores = sementic_scores(text)

    # ----- STEP 1: Semantic gating -----
    semantic_ranked = sorted(
        embed_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    allowed_careers = {c for c, _ in semantic_ranked[:top_k_semantic]}

    #learned fusion
    final_scores={}
    for career in allowed_careers:
        features=np.array([[
            tfidf_scores.get(career,0.0),
            embed_scores.get(career,0.0)
        ]])
        final_scores[career]=fusion_model.predict_proba(features)[0][1]

    # ----- STEP 3: Return ranked -----
    return sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


if __name__=="__main__":
    test = "I loved performing for people but also dreamed of building something big and leading others"

    result = hybrid_predict(test)

    print("\nHYBRID AI OUTPUT\n")
    for career,score in result[:3]:
        print(career, round(score, 3))