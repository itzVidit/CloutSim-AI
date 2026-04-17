"""
Enhanced Hybrid Prediction System v3
Features:
- Multi-model fusion with confidence scoring
- Context-aware predictions
- Explainable AI components
- Career pathway recommendations
- Ambiguity detection
"""

import joblib
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

from step18b_semantic_embeddings_v3 import semantic_scores, get_similar_careers, CAREER_PROTOTYPES

# -----------------------------
# Load All Models
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

print("📦 Loading models...")

vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
career_model = joblib.load(MODEL_DIR / "career_model.pkl")
popularity_model = joblib.load(MODEL_DIR / "popularity_model.pkl")
motivation_model = joblib.load(MODEL_DIR / "motivation_model.pkl")
age_model = joblib.load(MODEL_DIR / "age_model.pkl")
intensity_model = joblib.load(MODEL_DIR / "intensity_model.pkl")
personality_model = joblib.load(MODEL_DIR / "personality_model.pkl")
fusion_model = joblib.load(MODEL_DIR / "fusion_model_v3.pkl")

# Load metadata
with open(MODEL_DIR / "fusion_metadata_v3.json", "r") as f:
    fusion_metadata = json.load(f)
    feature_names = fusion_metadata["feature_names"]

print("✅ All models loaded")

# -----------------------------
# Feature Extraction for Prediction
# -----------------------------
def extract_prediction_features(text: str, career: str):
    """Extract features for a given text-career pair"""
    features = {}
    
    # TF-IDF
    X = vectorizer.transform([text])
    
    career_probs = career_model.predict_proba(X)[0]
    career_labels = career_model.classes_
    career_scores = dict(zip(career_labels, career_probs))
    
    features["tfidf_career"] = career_scores.get(career, 0.0)
    features["tfidf_max"] = max(career_probs)
    features["tfidf_entropy"] = -np.sum(career_probs * np.log(career_probs + 1e-10))
    
    pop_probs = popularity_model.predict_proba(X)[0]
    features["tfidf_popularity"] = max(pop_probs)
    
    # Semantic
    sem_data = semantic_scores(text, include_metadata=True)
    
    if career in sem_data:
        features["semantic_score"] = sem_data[career]["semantic_score"]
        features["keyword_matches"] = sem_data[career]["keyword_matches"]
        expected_pop = sem_data[career]["expected_popularity"]
    else:
        features["semantic_score"] = 0.0
        features["keyword_matches"] = 0
        expected_pop = 3
    
    # Popularity alignment (assume mid-range popularity for prediction)
    assumed_pop = 3
    features["pop_alignment"] = 1.0 - abs(assumed_pop - expected_pop) / 4.0
    features["pop_exact_match"] = 1.0 if assumed_pop == expected_pop else 0.0
    
    # Other model confidences
    features["motivation_confidence"] = max(motivation_model.predict_proba(X)[0])
    features["age_confidence"] = max(age_model.predict_proba(X)[0])
    features["intensity_confidence"] = max(intensity_model.predict_proba(X)[0])
    features["personality_confidence"] = max(personality_model.predict_proba(X)[0])
    
    # Text features
    features["text_length"] = len(text)
    features["word_count"] = len(text.split())
    features["has_contextual"] = 1 if len(text.split()) > 15 else 0
    
    # Career similarity
    top_tfidf_career = career_labels[np.argmax(career_probs)]
    if top_tfidf_career != career:
        similar_careers = dict(get_similar_careers(top_tfidf_career, top_k=10))
        features["career_similarity"] = similar_careers.get(career, 0.0)
    else:
        features["career_similarity"] = 1.0
    
    # Compound features
    features["tfidf_semantic_product"] = features["tfidf_career"] * features["semantic_score"]
    features["confidence_product"] = (
        features["tfidf_career"] * 
        features["semantic_score"] * 
        features["pop_alignment"]
    )
    
    return features

# -----------------------------
# Main Prediction Function
# -----------------------------
def hybrid_predict(
    text: str, 
    top_k: int = 10,
    semantic_threshold: float = 0.3,
    confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Advanced hybrid prediction with multiple signals
    
    Args:
        text: Input text describing childhood dream
        top_k: Number of top predictions to return
        semantic_threshold: Minimum semantic score to consider
        confidence_threshold: Minimum fusion score for high confidence
    
    Returns:
        List of career predictions with scores and metadata
    """
    
    # Get semantic scores for all careers
    sem_data = semantic_scores(text, include_metadata=True)
    
    # Filter by semantic threshold
    candidate_careers = [
        career for career, data in sem_data.items()
        if data["semantic_score"] >= semantic_threshold
    ]
    
    # If no candidates, take top 15 by semantic score
    if not candidate_careers:
        candidate_careers = sorted(
            sem_data.keys(),
            key=lambda c: sem_data[c]["semantic_score"],
            reverse=True
        )[:15]
    
    # Score each candidate with fusion model
    results = []
    
    for career in candidate_careers:
        features = extract_prediction_features(text, career)
        
        # Ensure feature order matches training
        feature_vector = np.array([[features[fname] for fname in feature_names]])
        
        # Fusion score
        fusion_proba = fusion_model.predict_proba(feature_vector)[0]
        fusion_score = fusion_proba[1]  # Probability of being correct match
        
        # Compile result
        career_data = sem_data[career]
        
        results.append({
            "career": career,
            "fusion_score": round(float(fusion_score), 4),
            "confidence_level": "high" if fusion_score >= confidence_threshold else "medium",
            "tfidf_score": round(features["tfidf_career"], 4),
            "semantic_score": round(features["semantic_score"], 4),
            "expected_popularity": career_data["expected_popularity"],
            "motivations": career_data["motivations"],
            "traits": career_data["traits"],
            "keyword_matches": career_data["keyword_matches"]
        })
    
    # Sort by fusion score
    results.sort(key=lambda x: x["fusion_score"], reverse=True)
    
    return results[:top_k]

# -----------------------------
# Context Prediction
# -----------------------------
def predict_context(text: str) -> Dict:
    """Predict contextual metadata about the dream"""
    X = vectorizer.transform([text])
    
    # Age context
    age_probs = age_model.predict_proba(X)[0]
    age_labels = age_model.classes_
    age_pred = age_labels[np.argmax(age_probs)]
    age_conf = max(age_probs)
    
    # Intensity
    intensity_probs = intensity_model.predict_proba(X)[0]
    intensity_labels = intensity_model.classes_
    intensity_pred = intensity_labels[np.argmax(intensity_probs)]
    intensity_conf = max(intensity_probs)
    
    # Personality
    personality_probs = personality_model.predict_proba(X)[0]
    personality_labels = personality_model.classes_
    personality_pred = personality_labels[np.argmax(personality_probs)]
    personality_conf = max(personality_probs)
    
    # Motivation
    motivation_probs = motivation_model.predict_proba(X)[0]
    motivation_labels = motivation_model.classes_
    motivation_pred = motivation_labels[np.argmax(motivation_probs)]
    motivation_conf = max(motivation_probs)
    
    return {
        "age_context": {
            "value": age_pred,
            "confidence": round(float(age_conf), 3)
        },
        "intensity": {
            "value": intensity_pred,
            "confidence": round(float(intensity_conf), 3)
        },
        "personality": {
            "value": personality_pred,
            "confidence": round(float(personality_conf), 3)
        },
        "primary_motivation": {
            "value": motivation_pred,
            "confidence": round(float(motivation_conf), 3)
        }
    }

# -----------------------------
# Ambiguity Detection
# -----------------------------
def detect_ambiguity(predictions: List[Dict]) -> Dict:
    """Detect if the input text is ambiguous between multiple careers"""
    
    if len(predictions) < 2:
        return {"is_ambiguous": False, "ambiguity_level": "clear"}
    
    top_score = predictions[0]["fusion_score"]
    second_score = predictions[1]["fusion_score"]
    
    score_gap = top_score - second_score
    
    if score_gap < 0.1:
        return {
            "is_ambiguous": True,
            "ambiguity_level": "high",
            "competing_careers": [predictions[0]["career"], predictions[1]["career"]],
            "explanation": "Very close scores suggest multiple career interpretations"
        }
    elif score_gap < 0.2:
        return {
            "is_ambiguous": True,
            "ambiguity_level": "moderate",
            "competing_careers": [predictions[0]["career"], predictions[1]["career"]],
            "explanation": "Moderate score difference suggests some ambiguity"
        }
    else:
        return {
            "is_ambiguous": False,
            "ambiguity_level": "clear",
            "explanation": "Clear dominant career prediction"
        }

# -----------------------------
# Complete Prediction Pipeline
# -----------------------------
def complete_prediction(text: str, top_k: int = 5) -> Dict:
    """
    Complete prediction with all metadata
    """
    
    # Career predictions
    predictions = hybrid_predict(text, top_k=top_k)
    
    # Context prediction
    context = predict_context(text)
    
    # Ambiguity detection
    ambiguity = detect_ambiguity(predictions)
    
    # Career pathway (similar careers to top prediction)
    if predictions:
        top_career = predictions[0]["career"]
        similar = get_similar_careers(top_career, top_k=5)
        career_pathway = [
            {"career": c, "similarity": round(s, 3)} 
            for c, s in similar
        ]
    else:
        career_pathway = []
    
    return {
        "input_text": text,
        "predictions": predictions,
        "context": context,
        "ambiguity": ambiguity,
        "career_pathway": career_pathway
    }

# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    test_cases = [
        "I loved performing for people but also dreamed of building something big and leading others",
        "Ever since I was young, I wanted to help sick people and make them feel better",
        "I dreamed of scoring goals in front of thousands of fans",
        "I wanted to explore space and discover new worlds"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST CASE {i}")
        print(f"{'=' * 70}")
        print(f"Input: {text}\n")
        
        result = complete_prediction(text, top_k=3)
        
        print("🎯 TOP PREDICTIONS:")
        for pred in result["predictions"]:
            print(f"\n  {pred['career'].upper()}")
            print(f"    Fusion Score: {pred['fusion_score']:.3f} ({pred['confidence_level']})")
            print(f"    Expected Popularity: {pred['expected_popularity']}/5")
            print(f"    Motivations: {', '.join(pred['motivations'])}")
            print(f"    Traits: {', '.join(pred['traits'])}")
        
        print(f"\n📊 CONTEXT:")
        print(f"  Age: {result['context']['age_context']['value']} "
              f"(conf: {result['context']['age_context']['confidence']:.3f})")
        print(f"  Intensity: {result['context']['intensity']['value']} "
              f"(conf: {result['context']['intensity']['confidence']:.3f})")
        print(f"  Personality: {result['context']['personality']['value']}")
        
        print(f"\n🔍 AMBIGUITY:")
        print(f"  Level: {result['ambiguity']['ambiguity_level']}")
        if result['ambiguity']['is_ambiguous']:
            print(f"  Competing: {', '.join(result['ambiguity']['competing_careers'])}")