# Text → Vectorizer → Model → Single Label + High Confidence

# to
# Text
#  ↓
# Motivation Detection
#  ↓
# Model Prediction
#  ↓
# Ambiguity Logic
#  ↓
# Adjusted Confidence + Ranked Output

import joblib
import numpy as np
from pathlib import Path

#paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

#load models
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
career_model = joblib.load(MODEL_DIR / "career_model.pkl")
popularity_model = joblib.load(MODEL_DIR / "popularity_model.pkl")

#motivation analizer
def analyze_motivation(text:str):
    text=text.lower()
    motivations=[]

    if any(w in text for w in ["perform", "acting", "stage", "entertain"]):
        motivations.append("Creative")

    if any(w in text for w in ["build", "company", "startup", "lead", "business"]):
        motivations.append("Leadership")

    if any(w in text for w in ["help", "heal", "serve", "care"]):
        motivations.append("Service")

    if any(w in text for w in ["famous", "known", "crowds", "millions"]):
        motivations.append("Fame")

    return motivations

#ambiguity detection
def detect_ambiguity(motivations):
    return len(set(motivations))>=2

#reasoned inference
def reasoned_inference(text:str):
    X=vectorizer.transform([text])

    #career probabilities
    career_probs=career_model.predict_proba(X)[0]
    career_labels=career_model.classes_

    #sort careers by prob
    ranked=sorted(
        zip(career_labels,career_probs),
        key=lambda x:x[1],
        reverse=True
    )

    # top_career,top_conf=ranked[0]
    second_career,second_conf=ranked[1]

    #lets filter out only thr non-zero probabilities
    filtered=[(c,p) for c,p in ranked if p>0.05]
    top_career,top_conf=filtered[0]

    career_ranked=[{"career":top_career,"confidence":round(top_conf,2)}]

    if len(filtered)>1:
        second_career,second_conf=filtered[1]
        career_ranked.append({
            "career":second_career,
            "confidence":round(second_conf,2)
        })

    motivations=analyze_motivation(text)
    ambiguous=detect_ambiguity(motivations)

    #confidence correction logic
    if ambiguous and filtered[0][1] > 0.90:
        corrected = []
    
        for i, (c, p) in enumerate(filtered):
            if i == 0:
                corrected.append((c, round(p * 0.6, 2)))
            elif i == 1:
                corrected.append((c, round(p * 1.4, 2)))

        filtered = corrected


    #final ranked output
    career_ranked=[]
    for c,p in filtered[:2]:
        career_ranked.append({
            "career":c,
            "confidence":p
        })
    
    #popularity
    pop_probs=popularity_model.predict_proba(X)[0]
    pop_labels=popularity_model.classes_
    pop_idx=np.argmax(pop_probs)

    return {
    "input": text,
    "motivations": motivations,
    "ambiguous": ambiguous,
    "career_ranked": career_ranked,
    "popularity": {
        "level": pop_labels[pop_idx],
        "confidence": round(float(pop_probs[pop_idx]), 2)
    }
}


#test case
if __name__ == "__main__":
    test_text = (
        "I loved performing for people but also dreamed of building something big and leading others"
    )

    result = reasoned_inference(test_text)

    print("\n🧠 REASONED AI OUTPUT\n")
    print("Input:")
    print(result["input"])

    print("\nDetected Motivations:", result["motivations"])
    print("Ambiguous Intent:", result["ambiguous"])

    print("\nCareer Possibilities:")
    for c in result["career_ranked"]:
        print(f"  {c['career']} (confidence: {c['confidence']})")

    print("\nPopularity:")
    print(result["popularity"])