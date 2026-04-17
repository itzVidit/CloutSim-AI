from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import your existing API (NO CHANGES inside it)
from api2 import api

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="CloutSimAI",
    description="Career Prediction & Explainability API",
    version="1.0.0"
)

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request Models
# -----------------------------
class PredictRequest(BaseModel):
    text: str
    top_k: int = 5
    explain: bool = False


class BatchPredictRequest(BaseModel):
    texts: List[str]
    top_k: int = 5


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health_check():
    return {
        "status": "CloutSimAI is running",
        "model": "hybrid_fusion_v3",
        "features": [
            "tfidf",
            "semantic",
            "popularity_alignment",
            "explainability",
            "batch_prediction",
            "caching"
        ]
    }


@app.post("/predict")
def predict_career(data: PredictRequest):
    """
    Single prediction endpoint
    """
    result = api.predict(
        data.text,
        top_k=data.top_k,
        include_explanation=data.explain
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/predict/batch")
def predict_batch(data: BatchPredictRequest):
    """
    Batch prediction endpoint
    """
    if not data.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")

    return api.predict_batch(data.texts, top_k=data.top_k)


@app.get("/careers")
def list_careers():
    """
    List all supported careers
    """
    return {
        "count": len(api.list_careers()),
        "careers": api.list_careers()
    }


@app.get("/careers/{career_name}")
def career_details(career_name: str):
    """
    Get details of a specific career
    """
    result = api.get_career_info(career_name)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@app.post("/cache/clear")
def clear_cache():
    """
    Clear prediction cache
    """
    api.clear_cache()
    return {"status": "Cache cleared successfully"}


# http://127.0.0.1:8000/docs
# python3 -m uvicorn final:app --reload