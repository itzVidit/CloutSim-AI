import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from step12_hybrid_embeddings import sementic_scores

#path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "proceed" / "dreams_validated.csv"
MODEL_DIR = BASE_DIR / "models"

vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
career_model = joblib.load(MODEL_DIR / "career_model.pkl")

#prepare training data
df=pd.read_csv(DATA_FILE,nrows=30000)
X_meta=[]
Y_meta=[]
for _,row in df.iterrows():
    text=row["text"]
    true_career=row["career"]

    #tf-idf score
    X = vectorizer.transform([text])
    tfidf_probs = career_model.predict_proba(X)[0]
    tfidf_labels = career_model.classes_
    tfidf_score = dict(zip(tfidf_labels, tfidf_probs)).get(true_career, 0.0)

    #sementic score
    sem_score = sementic_scores(text).get(true_career, 0.0)

    X_meta.append([tfidf_score,sem_score])
    Y_meta.append(1)

#add negative samples(wrong class)
for _,row in df.sample(30000).iterrows():
    text=row["text"]
    wrong_career = np.random.choice(career_model.classes_)

    X = vectorizer.transform([text])
    tfidf_probs = career_model.predict_proba(X)[0]
    tfidf_labels = career_model.classes_
    tfidf_score = dict(zip(tfidf_labels, tfidf_probs)).get(wrong_career, 0.0)

    sem_score = sementic_scores(text).get(wrong_career, 0.0)

    X_meta.append([tfidf_score, sem_score])
    Y_meta.append(0)

X_meta=np.array(X_meta)
Y_meta=np.array(Y_meta)

#final weighted fusion model
fusion_model=LogisticRegression()
fusion_model.fit(X_meta,Y_meta)

joblib.dump(fusion_model,MODEL_DIR/"fusion_model.pkl")

print("✅ Fusion weights learned")
print("TF-IDF weight:", fusion_model.coef_[0][0])
print("Semantic weight:", fusion_model.coef_[0][1])