from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from step21_explanation_layer_v2 import explain_prediction

app = FastAPI(title="CloutSimAI")

# --- CORS setup ---
origins = [
    "http://localhost:5173",   # Vite default
    "http://127.0.0.1:5173",
    "http://localhost:3000",   # if you use this anywhere
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] during dev
    allow_credentials=True,
    allow_methods=["*"],        # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)
# --- end CORS setup ---


class DreamInput(BaseModel):
    text: str
    desired_popularity: int | None = None


@app.post("/predict")
def predict_career(data: DreamInput):
    result = explain_prediction(data.text)
    if data.desired_popularity is not None:
        result["user_desired_popularity"] = data.desired_popularity
    return result


@app.get("/")
def root():
    return {
        "status": "CloutSimAI is running",
        "model": "hybrid_fusion_v2",
        "features": [
            "tfidf",
            "semantic",
            "popularity_alignment",
            "explainability"
        ]
    }

# Swagger UI:
# http://127.0.0.1:8000/docs
# python3 -m uvicorn api:app --reload