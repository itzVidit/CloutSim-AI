import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#load dataset
BASE_DIR=Path(__file__).resolve().parent.parent
csv_path=BASE_DIR/"data"/"raw"/"dreams.csv"

df=pd.read_csv(csv_path)

#features and labels
X_text=df["text"]
y_career=df["career"]

#text->numbers
vectorizer=TfidfVectorizer(stop_words="english")
X=vectorizer.fit_transform(X_text)

#train ml model
model=MultinomialNB()
model.fit(X,y_career)

#new Input
test_text=[
    "I imagined myself being on stage"
]

test_vector=vectorizer.transform(test_text)
prediction=model.predict(test_vector)

print("Input text: ")
print(test_text[0])

print("\nPredicted Career: ")
print(prediction[0])