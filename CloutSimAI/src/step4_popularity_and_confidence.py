import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#load dataset
BASE_DIR=Path(__file__).resolve().parent.parent
csv_path=BASE_DIR/"data"/"raw"/"dreams.csv"

df=pd.read_csv(csv_path)

X_text=df["text"]
Y_career=df["career"]
Y_popularity=df["popularity"]

#vectorisaiton
vectorizer=TfidfVectorizer(stop_words="english")
X=vectorizer.fit_transform(X_text)

#train model
career_model=MultinomialNB()
career_model.fit(X,Y_career)

popularity_model=MultinomialNB()
popularity_model.fit(X,Y_popularity)

#test input
test_text = [
    "I dreamed of being known by millions and performing on big stages"
]

test_vector = vectorizer.transform(test_text)

#output
career_pred=career_model.predict(test_vector)[0]
career_prop=career_model.predict_proba(test_vector).max()

popularity_pred=popularity_model.predict(test_vector)[0]
popularity_prob=popularity_model.predict_proba(test_vector).max()

print("Input:")
print(test_text[0])

print("\nPredicted Career:", career_pred)
print("Career Confidence:", round(career_prop, 2))

print("\nPredicted Popularity Level:", popularity_pred)
print("Popularity Confidence:", round(popularity_prob, 2))