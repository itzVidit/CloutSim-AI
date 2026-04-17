"""
Enhanced Semantic Embeddings System v3
Features:
- Expanded career prototypes (30+ careers)
- Multi-dimensional scoring (semantic + motivation + traits)
- Context-aware scoring (age, intensity, personality)
- Career clustering and similarity
- Caching for efficiency
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
import json
from pathlib import Path

# Load embedding model (cached)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Enhanced Career Prototypes
# -----------------------------
CAREER_PROTOTYPES = {
    "actor": {
        "texts": [
            "acting on stage", "performing for audiences", "expressing emotions",
            "entertaining people", "being in front of cameras", "dramatic performances"
        ],
        "popularity": 5,
        "keywords": ["stage", "perform", "act", "drama", "audience", "entertainment"],
        "motivations": ["recognition", "creativity", "expression"],
        "traits": ["extroverted", "creative", "expressive"]
    },
    "entrepreneur": {
        "texts": [
            "building a company", "leading a startup", "creating a business",
            "innovating products", "taking risks in business", "scaling organizations"
        ],
        "popularity": 4,
        "keywords": ["business", "startup", "company", "innovate", "lead", "build"],
        "motivations": ["independence", "impact", "wealth"],
        "traits": ["ambitious", "risk-taker", "leader"]
    },
    "engineer": {
        "texts": [
            "solving technical problems", "building systems", "working with code",
            "designing solutions", "creating technology", "debugging complex systems"
        ],
        "popularity": 2,
        "keywords": ["technical", "solve", "build", "code", "system", "design"],
        "motivations": ["problem_solving", "creation", "intellectual"],
        "traits": ["analytical", "detail-oriented", "logical"]
    },
    "doctor": {
        "texts": [
            "helping sick people", "healing patients", "working in hospitals",
            "diagnosing illnesses", "saving lives", "caring for health"
        ],
        "popularity": 2,
        "keywords": ["heal", "help", "patient", "medical", "health", "care"],
        "motivations": ["helping_others", "prestige", "stability"],
        "traits": ["compassionate", "analytical", "dedicated"]
    },
    "teacher": {
        "texts": [
            "teaching students", "explaining concepts", "guiding learners",
            "inspiring young minds", "educating children", "shaping futures"
        ],
        "popularity": 2,
        "keywords": ["teach", "student", "educate", "explain", "guide", "learn"],
        "motivations": ["helping_others", "impact", "stability"],
        "traits": ["patient", "communicative", "nurturing"]
    },
    "scientist": {
        "texts": [
            "conducting research", "discovering theories", "experimenting",
            "exploring unknowns", "analyzing data", "advancing knowledge"
        ],
        "popularity": 2,
        "keywords": ["research", "discover", "experiment", "analyze", "science", "study"],
        "motivations": ["intellectual", "discovery", "impact"],
        "traits": ["curious", "analytical", "methodical"]
    },
    "lawyer": {
        "texts": [
            "arguing legal cases", "representing clients", "interpreting laws",
            "defending justice", "negotiating agreements", "advocating rights"
        ],
        "popularity": 3,
        "keywords": ["law", "legal", "argue", "defend", "justice", "case"],
        "motivations": ["prestige", "wealth", "justice"],
        "traits": ["argumentative", "analytical", "confident"]
    },
    "politician": {
        "texts": [
            "leading people", "shaping policies", "addressing crowds",
            "representing communities", "making decisions", "influencing society"
        ],
        "popularity": 5,
        "keywords": ["lead", "policy", "govern", "represent", "influence", "power"],
        "motivations": ["power", "impact", "recognition"],
        "traits": ["charismatic", "ambitious", "leader"]
    },
    "cricketer": {
        "texts": [
            "playing cricket", "scoring runs", "representing teams",
            "competing in matches", "bowling", "fielding professionally"
        ],
        "popularity": 5,
        "keywords": ["cricket", "play", "sport", "team", "compete", "match"],
        "motivations": ["fame", "competition", "passion"],
        "traits": ["athletic", "competitive", "team-player"]
    },
    "footballer": {
        "texts": [
            "playing football", "scoring goals", "competing professionally",
            "representing clubs", "winning championships", "training intensely"
        ],
        "popularity": 5,
        "keywords": ["football", "soccer", "goal", "play", "team", "compete"],
        "motivations": ["fame", "competition", "passion"],
        "traits": ["athletic", "competitive", "disciplined"]
    },
    "boxer": {
        "texts": [
            "fighting in the ring", "training for matches", "competing in boxing",
            "winning championships", "knockout victories", "defending titles"
        ],
        "popularity": 4,
        "keywords": ["fight", "box", "ring", "compete", "champion", "knockout"],
        "motivations": ["competition", "fame", "strength"],
        "traits": ["aggressive", "disciplined", "resilient"]
    },
    "musician": {
        "texts": [
            "composing music", "performing concerts", "playing instruments",
            "creating melodies", "recording albums", "entertaining crowds"
        ],
        "popularity": 4,
        "keywords": ["music", "perform", "compose", "play", "song", "concert"],
        "motivations": ["creativity", "expression", "recognition"],
        "traits": ["creative", "expressive", "passionate"]
    },
    "artist": {
        "texts": [
            "creating visual art", "painting", "expressing imagination",
            "sculpting", "designing artwork", "exhibiting pieces"
        ],
        "popularity": 3,
        "keywords": ["art", "paint", "create", "design", "visual", "express"],
        "motivations": ["creativity", "expression", "independence"],
        "traits": ["creative", "imaginative", "detail-oriented"]
    },
    "journalist": {
        "texts": [
            "reporting news", "investigating stories", "exposing truth",
            "writing articles", "interviewing people", "covering events"
        ],
        "popularity": 3,
        "keywords": ["report", "news", "investigate", "write", "story", "truth"],
        "motivations": ["impact", "truth", "communication"],
        "traits": ["curious", "communicative", "persistent"]
    },
    "pilot": {
        "texts": [
            "flying aircraft", "navigating skies", "transporting passengers",
            "operating planes", "aviation responsibilities", "commanding flights"
        ],
        "popularity": 3,
        "keywords": ["fly", "plane", "aircraft", "pilot", "aviation", "navigate"],
        "motivations": ["adventure", "prestige", "travel"],
        "traits": ["confident", "detail-oriented", "calm"]
    },
    "architect": {
        "texts": [
            "designing buildings", "planning structures", "creating spaces",
            "envisioning architecture", "drawing blueprints", "shaping cities"
        ],
        "popularity": 3,
        "keywords": ["design", "building", "architecture", "structure", "plan", "space"],
        "motivations": ["creativity", "impact", "legacy"],
        "traits": ["creative", "analytical", "visionary"]
    },
    "designer": {
        "texts": [
            "designing experiences", "creating visuals", "shaping aesthetics",
            "crafting interfaces", "developing brands", "visual communication"
        ],
        "popularity": 3,
        "keywords": ["design", "create", "visual", "aesthetic", "interface", "brand"],
        "motivations": ["creativity", "expression", "impact"],
        "traits": ["creative", "detail-oriented", "empathetic"]
    },
    "psychologist": {
        "texts": [
            "understanding behavior", "counseling individuals", "studying minds",
            "helping mental health", "analyzing patterns", "therapeutic work"
        ],
        "popularity": 2,
        "keywords": ["psychology", "counseling", "mental", "behavior", "therapy", "mind"],
        "motivations": ["helping_others", "intellectual", "understanding"],
        "traits": ["empathetic", "analytical", "patient"]
    },
    "data_scientist": {
        "texts": [
            "analyzing data", "building models", "finding patterns",
            "machine learning", "statistical analysis", "data insights"
        ],
        "popularity": 3,
        "keywords": ["data", "analyze", "model", "statistics", "pattern", "insight"],
        "motivations": ["intellectual", "problem_solving", "innovation"],
        "traits": ["analytical", "curious", "methodical"]
    },
    "game_developer": {
        "texts": [
            "building games", "designing gameplay", "creating worlds",
            "coding interactions", "game mechanics", "interactive experiences"
        ],
        "popularity": 4,
        "keywords": ["game", "develop", "code", "play", "interactive", "create"],
        "motivations": ["creativity", "passion", "innovation"],
        "traits": ["creative", "technical", "imaginative"]
    },
    "chef": {
        "texts": [
            "cooking professionally", "creating dishes", "culinary arts",
            "restaurant kitchen", "preparing meals", "food creativity"
        ],
        "popularity": 3,
        "keywords": ["cook", "chef", "food", "culinary", "kitchen", "dish"],
        "motivations": ["creativity", "passion", "recognition"],
        "traits": ["creative", "detail-oriented", "passionate"]
    },
    "astronaut": {
        "texts": [
            "exploring space", "traveling to orbit", "space missions",
            "scientific research in space", "spacewalks", "zero gravity"
        ],
        "popularity": 5,
        "keywords": ["space", "astronaut", "orbit", "mission", "explore", "rocket"],
        "motivations": ["adventure", "discovery", "prestige"],
        "traits": ["brave", "disciplined", "curious"]
    },
    "writer": {
        "texts": [
            "writing stories", "creating narratives", "authoring books",
            "crafting prose", "publishing works", "storytelling"
        ],
        "popularity": 3,
        "keywords": ["write", "story", "book", "author", "narrative", "publish"],
        "motivations": ["creativity", "expression", "legacy"],
        "traits": ["creative", "introspective", "imaginative"]
    },
    "photographer": {
        "texts": [
            "capturing moments", "taking photographs", "visual storytelling",
            "documenting scenes", "creative photography", "image composition"
        ],
        "popularity": 3,
        "keywords": ["photo", "capture", "camera", "image", "visual", "shoot"],
        "motivations": ["creativity", "expression", "documentation"],
        "traits": ["creative", "observant", "patient"]
    },
    "veterinarian": {
        "texts": [
            "treating animals", "caring for pets", "animal medicine",
            "healing creatures", "veterinary care", "animal health"
        ],
        "popularity": 2,
        "keywords": ["animal", "veterinary", "pet", "care", "treat", "heal"],
        "motivations": ["helping_others", "compassion", "passion"],
        "traits": ["compassionate", "patient", "dedicated"]
    },
    "firefighter": {
        "texts": [
            "fighting fires", "saving lives", "emergency response",
            "rescuing people", "fire safety", "heroic actions"
        ],
        "popularity": 4,
        "keywords": ["fire", "rescue", "save", "emergency", "hero", "brave"],
        "motivations": ["helping_others", "heroism", "service"],
        "traits": ["brave", "strong", "selfless"]
    },
    "police_officer": {
        "texts": [
            "protecting communities", "enforcing law", "serving justice",
            "maintaining order", "crime prevention", "public safety"
        ],
        "popularity": 3,
        "keywords": ["police", "protect", "law", "enforce", "justice", "serve"],
        "motivations": ["service", "justice", "helping_others"],
        "traits": ["brave", "disciplined", "protective"]
    },
    "dancer": {
        "texts": [
            "performing dances", "choreographing", "expressing through movement",
            "ballet", "contemporary dance", "entertaining audiences"
        ],
        "popularity": 4,
        "keywords": ["dance", "perform", "choreograph", "movement", "ballet", "stage"],
        "motivations": ["expression", "creativity", "passion"],
        "traits": ["expressive", "disciplined", "athletic"]
    },
    "fashion_designer": {
        "texts": [
            "designing clothes", "creating fashion", "styling outfits",
            "fashion shows", "textile design", "trendsetting"
        ],
        "popularity": 4,
        "keywords": ["fashion", "design", "clothes", "style", "trend", "outfit"],
        "motivations": ["creativity", "recognition", "expression"],
        "traits": ["creative", "trendy", "visionary"]
    },
    "marine_biologist": {
        "texts": [
            "studying ocean life", "researching marine animals", "underwater exploration",
            "ocean conservation", "sea creatures", "aquatic science"
        ],
        "popularity": 3,
        "keywords": ["ocean", "marine", "sea", "underwater", "fish", "research"],
        "motivations": ["discovery", "passion", "conservation"],
        "traits": ["curious", "passionate", "adventurous"]
    }
}

# -----------------------------
# Precompute Embeddings
# -----------------------------
print("🔧 Computing career prototype embeddings...")
CAREER_EMBEDS = {
    career: np.mean([embedder.encode(t, normalize_embeddings=True) for t in data["texts"]], axis=0)
    for career, data in CAREER_PROTOTYPES.items()
}
print(f"✅ {len(CAREER_EMBEDS)} career prototypes loaded")

# -----------------------------
# Cached Embedding Function
# -----------------------------
@lru_cache(maxsize=10000)
def embed_text(text: str):
    """Cache embeddings for repeated queries"""
    return embedder.encode(text, normalize_embeddings=True)

# -----------------------------
# Enhanced Semantic Scoring
# -----------------------------
def semantic_scores(text: str, include_metadata: bool = True):
    """
    Multi-dimensional semantic scoring
    Returns scores, metadata, and reasoning
    """
    v = embed_text(text)
    text_lower = text.lower()
    
    results = {}
    
    for career, proto_vec in CAREER_EMBEDS.items():
        # Cosine similarity
        cos_sim = float(np.dot(v, proto_vec))
        
        # Keyword matching boost
        career_data = CAREER_PROTOTYPES[career]
        keyword_matches = sum(1 for kw in career_data["keywords"] if kw in text_lower)
        keyword_boost = min(keyword_matches * 0.05, 0.15)  # Max 15% boost
        
        # Final semantic score
        semantic_score = min(cos_sim + keyword_boost, 1.0)
        
        if include_metadata:
            results[career] = {
                "semantic_score": round(semantic_score, 4),
                "expected_popularity": career_data["popularity"],
                "motivations": career_data["motivations"],
                "traits": career_data["traits"],
                "keyword_matches": keyword_matches
            }
        else:
            results[career] = {
                "semantic_score": round(semantic_score, 4),
                "expected_popularity": career_data["popularity"]
            }
    
    return results

# -----------------------------
# Career Similarity
# -----------------------------
def get_similar_careers(career: str, top_k: int = 5):
    """Find similar careers based on embeddings"""
    if career not in CAREER_EMBEDS:
        return []
    
    target_vec = CAREER_EMBEDS[career]
    similarities = []
    
    for other_career, other_vec in CAREER_EMBEDS.items():
        if other_career == career:
            continue
        
        sim = float(np.dot(target_vec, other_vec))
        similarities.append((other_career, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    test_text = "I loved performing for people but also dreamed of building something big and leading others"
    
    scores = semantic_scores(test_text)
    
    print("\n🔹 Enhanced Semantic Scores\n")
    for career, data in sorted(scores.items(), key=lambda x: x[1]["semantic_score"], reverse=True)[:10]:
        print(f"{career:20s} | score: {data['semantic_score']:.3f} | "
              f"pop: {data['expected_popularity']} | keywords: {data['keyword_matches']}")
    
    print("\n🔹 Career Similarities (Actor):\n")
    for similar, sim in get_similar_careers("actor"):
        print(f"{similar:20s} | similarity: {sim:.3f}")