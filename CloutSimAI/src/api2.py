"""
Production API Interface
Features:
- Simple prediction API
- Batch prediction support
- Caching layer
- Error handling
- Logging
"""

import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache
import time
import logging

from step19b_hybrid_fusion_v3 import complete_prediction, hybrid_predict
from step21b_explanation_layer_v3 import explain_prediction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# API Class
# -----------------------------
class CareerPredictionAPI:
    """
    Production-ready API for career prediction
    """
    
    def __init__(self):
        """Initialize models on startup"""
        logger.info("Initializing Career Prediction API...")
        
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.MODEL_DIR = self.BASE_DIR / "models"
        
        # Check if models exist
        required_models = [
            "vectorizer.pkl",
            "career_model.pkl",
            "popularity_model.pkl",
            "fusion_model_v3.pkl"
        ]
        
        for model_file in required_models:
            if not (self.MODEL_DIR / model_file).exists():
                raise FileNotFoundError(
                    f"Required model not found: {model_file}. "
                    f"Please train models first."
                )
        
        logger.info("✅ All required models found")
        logger.info("API ready for predictions")
    
    @lru_cache(maxsize=1000)
    def predict(
        self, 
        text: str, 
        top_k: int = 5,
        include_explanation: bool = False
    ) -> Dict:
        """
        Predict career from childhood dream text
        
        Args:
            text: Input text describing childhood dream
            top_k: Number of top predictions to return
            include_explanation: Include natural language explanation
        
        Returns:
            Dictionary with predictions and metadata
        """
        
        start_time = time.time()
        
        try:
            # Input validation
            if not text or not isinstance(text, str):
                return {
                    "error": "Invalid input. Text must be a non-empty string.",
                    "predictions": []
                }
            
            if len(text) < 10:
                return {
                    "error": "Input text too short. Please provide more detail.",
                    "predictions": []
                }
            
            # Get predictions
            if include_explanation:
                result = explain_prediction(text)
                response = {
                    "input": text,
                    "explanation": result["explanation"],
                    "predictions": result["predictions"][:top_k],
                    "context": result["context"],
                    "ambiguity": result["ambiguity"]
                }
            else:
                result = complete_prediction(text, top_k=top_k)
                response = {
                    "input": text,
                    "predictions": result["predictions"],
                    "context": result["context"],
                    "ambiguity": result["ambiguity"]
                }
            
            # Add metadata
            response["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
            response["top_k"] = top_k
            
            logger.info(f"Prediction completed in {response['processing_time_ms']}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "error": f"Prediction failed: {str(e)}",
                "predictions": []
            }
    
    def predict_batch(
        self, 
        texts: List[str], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Batch prediction for multiple texts
        
        Args:
            texts: List of input texts
            top_k: Number of top predictions per text
        
        Returns:
            List of prediction dictionaries
        """
        
        logger.info(f"Processing batch of {len(texts)} texts")
        start_time = time.time()
        
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.predict(text, top_k=top_k, include_explanation=False)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(texts)
        
        logger.info(
            f"Batch completed in {total_time:.2f}s "
            f"(avg: {avg_time*1000:.2f}ms per text)"
        )
        
        return results
    
    def get_career_info(self, career: str) -> Dict:
        """
        Get detailed information about a specific career
        
        Args:
            career: Career name (e.g., "actor", "engineer")
        
        Returns:
            Career information dictionary
        """
        
        from step18b_semantic_embeddings_v3 import CAREER_PROTOTYPES, get_similar_careers
        
        if career not in CAREER_PROTOTYPES:
            return {
                "error": f"Career '{career}' not found in database",
                "available_careers": list(CAREER_PROTOTYPES.keys())
            }
        
        career_data = CAREER_PROTOTYPES[career]
        similar = get_similar_careers(career, top_k=5)
        
        return {
            "career": career,
            "display_name": career.replace("_", " ").title(),
            "popularity": career_data["popularity"],
            "motivations": career_data["motivations"],
            "traits": career_data["traits"],
            "keywords": career_data["keywords"],
            "similar_careers": [
                {"career": c, "similarity": round(s, 3)}
                for c, s in similar
            ]
        }
    
    def list_careers(self) -> List[str]:
        """
        Get list of all available careers
        
        Returns:
            List of career names
        """
        from step18b_semantic_embeddings_v3 import CAREER_PROTOTYPES
        return sorted(list(CAREER_PROTOTYPES.keys()))
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.predict.cache_clear()
        logger.info("Prediction cache cleared")

# -----------------------------
# API Instance
# -----------------------------
api = CareerPredictionAPI()

# -----------------------------
# Convenience Functions
# -----------------------------
def predict_career(text: str, top_k: int = 5, explain: bool = False) -> Dict:
    """
    Simple prediction function
    
    Usage:
        result = predict_career("I wanted to be an actor", top_k=3, explain=True)
    """
    return api.predict(text, top_k=top_k, include_explanation=explain)

def predict_careers_batch(texts: List[str], top_k: int = 5) -> List[Dict]:
    """
    Batch prediction function
    
    Usage:
        texts = ["I wanted to be an actor", "I dreamed of being a scientist"]
        results = predict_careers_batch(texts)
    """
    return api.predict_batch(texts, top_k=top_k)

def get_career_details(career: str) -> Dict:
    """
    Get career information
    
    Usage:
        info = get_career_details("actor")
    """
    return api.get_career_info(career)

def list_all_careers() -> List[str]:
    """
    List all available careers
    
    Usage:
        careers = list_all_careers()
    """
    return api.list_careers()

# -----------------------------
# Example Usage & Testing
# -----------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("CAREER PREDICTION API - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Simple prediction
    print("\n📌 TEST 1: Simple Prediction")
    print("-" * 80)
    result = predict_career(
        "I loved performing on stage and entertaining people",
        top_k=3
    )
    
    print(f"\nInput: {result['input']}")
    print(f"\nTop {len(result['predictions'])} Predictions:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"\n  {i}. {pred['career'].replace('_', ' ').title()}")
        print(f"     Score: {pred['fusion_score']:.3f} ({pred['confidence_level']})")
        print(f"     Popularity: {pred['expected_popularity']}/5")
    
    print(f"\nProcessing time: {result['processing_time_ms']}ms")
    
    # Test 2: With explanation
    print("\n\n📌 TEST 2: Prediction with Explanation")
    print("-" * 80)
    result = predict_career(
        "Ever since I was 5, I wanted to help sick people get better",
        top_k=1,
        explain=True
    )
    
    print(f"\n{result['explanation']}")
    
    # Test 3: Batch prediction
    print("\n\n📌 TEST 3: Batch Prediction")
    print("-" * 80)
    texts = [
        "I dreamed of scoring goals",
        "I wanted to build robots",
        "I loved drawing and painting"
    ]
    
    results = predict_careers_batch(texts, top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\n  Text {i}: {result['input']}")
        top_pred = result['predictions'][0]
        print(f"  → {top_pred['career'].replace('_', ' ').title()} "
              f"(score: {top_pred['fusion_score']:.3f})")
    
    # Test 4: Career info
    print("\n\n📌 TEST 4: Career Information")
    print("-" * 80)
    info = get_career_details("engineer")
    
    print(f"\nCareer: {info['display_name']}")
    print(f"Popularity: {info['popularity']}/5")
    print(f"Motivations: {', '.join(info['motivations'])}")
    print(f"Traits: {', '.join(info['traits'])}")
    print(f"\nSimilar careers:")
    for sim in info['similar_careers'][:3]:
        print(f"  - {sim['career'].replace('_', ' ').title()} "
              f"(similarity: {sim['similarity']:.3f})")
    
    # Test 5: List careers
    print("\n\n📌 TEST 5: Available Careers")
    print("-" * 80)
    careers = list_all_careers()
    print(f"\nTotal careers: {len(careers)}")
    print(f"Sample: {', '.join([c.replace('_', ' ').title() for c in careers[:10]])}, ...")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)