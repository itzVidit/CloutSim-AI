import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#motivaiton
def analyze_motivation(text):
    text=text.lower()
    motivations=[]

    if any(w in text for w in ["help", "heal", "care", "serve"]):
        motivations.append("Helping Others")

    if any(w in text for w in ["famous", "known", "millions", "popular"]):
        motivations.append("Fame")

    if any(w in text for w in ["stage", "perform", "acting", "expression"]):
        motivations.append("Creative Expression")

    if any(w in text for w in ["lead", "company", "startup", "business"]):
        motivations.append("Leadership")

    if not motivations:
        motivations.append("Unclear")

    return motivations

#dataset
BASE_DIR=Path(__file__).resolve().parent.parent
csv_path=BASE_DIR/"data"/"raw"/"dreams.csv"

df=pd.read_csv(csv_path)

X_text=df["text"]
Y_career=df["career"]
Y_popularity=df["popularity"]

#vectorization
vectorizer=TfidfVectorizer(stop_words="english")
X=vectorizer.fit_transform(X_text)

#train
career_model = MultinomialNB()
career_model.fit(X, Y_career)

popularity_model = MultinomialNB()
popularity_model.fit(X, Y_popularity)

#test
user_text = (
    # "I dreamed of being known by millions and performing on big stages"
    # "I dreamed of expressing emotions on stage and acting for audiences"
    # "I wanted to build a company that becomes famous worldwide"
    "I imagined helping patients and saving lives quietly"
)

test_vector = vectorizer.transform([user_text])

career_probs = career_model.predict_proba(test_vector)[0]
career_labels = career_model.classes_

career_prediction = career_labels[career_probs.argmax()]
career_confidence = career_probs.max()

popularity_prediction = popularity_model.predict(test_vector)[0]
popularity_confidence = popularity_model.predict_proba(test_vector).max()

motivations = analyze_motivation(user_text)

#decision adjustment
if career_prediction=="Entrepreneur" and "Creative Expression" in motivations:
    final_career="Actor"
    reason="Creative expression detected, overriding buisness bias"
else:
    final_career=career_prediction
    reason="Statistical match with training model"

#final report
print("\n🧠 DREAM → DESTINY ANALYSIS\n")

print("Input:")
print(user_text)

print("\nDetected Motivations:")
print(", ".join(motivations))

print("\nAI Career Prediction:")
print(career_prediction, f"(confidence: {career_confidence:.2f})")

print("\nFinal Career Decision:")
print(final_career)
print("Reason:", reason)

print("\nPopularity Desire Level:")
print(popularity_prediction, f"(confidence: {popularity_confidence:.2f})")