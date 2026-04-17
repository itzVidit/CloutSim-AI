from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# Load embedding model
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str):
    return embedder.encode(text, normalize_embeddings=True)

# -----------------------------
# Career prototypes (20+)
# -----------------------------
CAREER_PROTOTYPES = {
    "actor": {
        "texts": [
            "acting on stage",
            "performing for audiences",
            "expressing emotions in front of people"
        ],
        "popularity": 5
    },
    "entrepreneur": {
        "texts": [
            "building a company",
            "leading a startup",
            "creating a business organization"
        ],
        "popularity": 4
    },
    "engineer": {
        "texts": [
            "solving technical problems",
            "building systems",
            "working with logic and code"
        ],
        "popularity": 2
    },
    "doctor": {
        "texts": [
            "helping sick people",
            "working in hospitals",
            "healing patients"
        ],
        "popularity": 2
    },
    "teacher": {
        "texts": [
            "teaching students",
            "explaining concepts",
            "guiding learners"
        ],
        "popularity": 2
    },
    "scientist": {
        "texts": [
            "conducting research",
            "discovering new theories",
            "experimenting in laboratories"
        ],
        "popularity": 2
    },
    "lawyer": {
        "texts": [
            "arguing legal cases",
            "representing clients",
            "interpreting laws"
        ],
        "popularity": 3
    },
    "politician": {
        "texts": [
            "leading people",
            "addressing large crowds",
            "shaping public policies"
        ],
        "popularity": 5
    },
    "cricketer": {
        "texts": [
            "playing professional cricket",
            "scoring runs",
            "representing a team"
        ],
        "popularity": 5
    },
    "footballer": {
        "texts": [
            "playing professional football",
            "scoring goals",
            "competing in football matches"
        ],
        "popularity": 5
    },
    "boxer": {
        "texts": [
            "training for boxing matches",
            "fighting in the ring",
            "competing in championships"
        ],
        "popularity": 4
    },
    "musician": {
        "texts": [
            "composing music",
            "performing live concerts",
            "expressing emotions through music"
        ],
        "popularity": 4
    },
    "artist": {
        "texts": [
            "creating visual art",
            "painting artwork",
            "expressing imagination visually"
        ],
        "popularity": 3
    },
    "journalist": {
        "texts": [
            "reporting news",
            "investigating stories",
            "exposing truth"
        ],
        "popularity": 3
    },
    "pilot": {
        "texts": [
            "flying aircraft",
            "navigating the skies",
            "handling aviation responsibilities"
        ],
        "popularity": 3
    },
    "architect": {
        "texts": [
            "designing buildings",
            "planning structures",
            "creating functional spaces"
        ],
        "popularity": 3
    },
    "designer": {
        "texts": [
            "designing user experiences",
            "creating visual designs",
            "shaping aesthetics"
        ],
        "popularity": 3
    },
    "psychologist": {
        "texts": [
            "understanding human behavior",
            "counseling individuals",
            "studying mental patterns"
        ],
        "popularity": 2
    },
    "data_scientist": {
        "texts": [
            "analyzing data",
            "building predictive models",
            "finding patterns in datasets"
        ],
        "popularity": 3
    },
    "game_developer": {
        "texts": [
            "building games",
            "designing gameplay mechanics",
            "creating interactive worlds"
        ],
        "popularity": 4
    }
}

# -----------------------------
# Precompute prototype embeddings
# -----------------------------
CAREER_EMBEDS = {
    career: np.mean([embed(t) for t in data["texts"]], axis=0)
    for career, data in CAREER_PROTOTYPES.items()
}

# -----------------------------
# Semantic scoring (normalized)
# -----------------------------
def semantic_scores(text: str):
    v = embed(text)

    raw_scores = {
        career: float(np.dot(v, proto_vec))
        for career, proto_vec in CAREER_EMBEDS.items()
    }

    min_s = min(raw_scores.values())
    max_s = max(raw_scores.values())

    normalized = {
        career: (score - min_s) / (max_s - min_s + 1e-8)
        for career, score in raw_scores.items()
    }

    # attach popularity expectation
    final_scores = {
        career: {
            "semantic_score": round(normalized[career], 3),
            "expected_popularity": CAREER_PROTOTYPES[career]["popularity"]
        }
        for career in normalized
    }

    return final_scores

# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    test_text = "I love performing for people but also dreamed of leading others"

    scores = semantic_scores(test_text)

    print("\n🔹 Semantic + Popularity Scores\n")
    for career, data in sorted(
        scores.items(),
        key=lambda x: x[1]["semantic_score"],
        reverse=True
    ):
        print(
            career,
            "semantic:", data["semantic_score"],
            "| popularity:", data["expected_popularity"]
        )