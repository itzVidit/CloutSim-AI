import joblib
from pathlib import Path
import numpy as np

#paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

#load models
vectorizer=joblib.load(MODEL_DIR/"vectorizer.pkl")
career_model=joblib.load(MODEL_DIR/"career_model.pkl")
popularity_model=joblib.load(MODEL_DIR/"popularity_model.pkl")

#test input
user_text = [
    "I loved performing for people but also dreamed of building something big and leading others"
]

#vectorize
X=vectorizer.transform(user_text)

#prediction
career_probs=career_model.predict_proba(X)[0]
career_classes=career_model.classes_

career_idx=np.argmax(career_probs)
career_pred=career_classes[career_idx]
career_conf=career_probs[career_idx]

#popularity prediction
pop_probs=popularity_model.predict_proba(X)[0]
pop_classes=popularity_model.classes_

pop_idx=np.argmax(pop_probs)
pop_pred=pop_classes[pop_idx]
pop_conf=pop_probs[pop_idx]

#output
print("\nAI Inference result:\n")
print("Input: ")
print(user_text[0])

print("\nPredicted Career: ",career_pred)
print("\nCareer Confidence: ",round(float(career_conf),3))

print("\n\nPredicted popularity level: ",pop_pred)
print("\nPopularity confidence: ",round(float(pop_conf),3))