import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report

#paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "proceed" / "dreams_validated.csv"
MODEL_DIR = BASE_DIR / "models"

#load models
vectorizer=joblib.load(MODEL_DIR/"vectorizer.pkl")
career_model=joblib.load(MODEL_DIR/"career_model.pkl")

#evaluation data
df=pd.read_csv(DATA_FILE,nrows=50_000) #small slice

X=vectorizer.transform(df["text"])
Y_true=df["career"]

Y_pred=career_model.predict(X)

#report
print("\nCareer model evaluation")
print(classification_report(Y_true,Y_pred))