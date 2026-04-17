import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "proceed" / "dreams_augmented.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

#parameters
CHUNK_SIZE=50_000
MAX_FEATURES=50_000

#initialise vectorizer
vectorizer=TfidfVectorizer(
    stop_words="english",
    max_features=MAX_FEATURES
)

#first pass(fit vectorizer)
print("Fitting TF-IDF vectorizer")

text_iter=(
    text
    for chunk in pd.read_csv(DATA_FILE,chunksize=CHUNK_SIZE)
    for text in chunk["text"].astype(str)
)

vectorizer.fit(text_iter)
print("Vocabulary size: ",len(vectorizer.vocabulary_))

#initialize models
career_model=MultinomialNB()
popularity_model=MultinomialNB()

career_classes=None
popularity_classes=None

#second pass(train models)
print("Training models...")

for i,chunk in enumerate(pd.read_csv(DATA_FILE,chunksize=CHUNK_SIZE)):
    X=vectorizer.transform(chunk["text"])
    Y_career=chunk["career"]
    Y_popularity=chunk["popularity"]

    if i==0:
        career_classes=Y_career.unique()
        popularity_classes=Y_popularity.unique()

        career_model.partial_fit(X,Y_career,classes=career_classes)
        popularity_model.partial_fit(X,Y_popularity,classes=popularity_classes)

    else:
        career_model.partial_fit(X,Y_career)
        popularity_model.partial_fit(X,Y_popularity)
    
    print("Trained on {(i+1)*CHUNK_SIZE:,} rows")


#save models
joblib.dump(vectorizer,MODEL_DIR/"vectorizer.pkl")
joblib.dump(career_model,MODEL_DIR/"career_model.pkl")
joblib.dump(popularity_model,MODEL_DIR/"popularity_model.pkl")

print("\nTraining Complete")
print("\nModels saved to",MODEL_DIR)