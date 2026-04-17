"""
Explainable AI System v3
Features:
- Natural language explanations
- Evidence-based reasoning
- Alternative career suggestions
- Confidence explanations
- Personality trait insights
"""

from step19b_hybrid_fusion_v3 import complete_prediction, hybrid_predict
from step18b_semantic_embeddings_v3 import CAREER_PROTOTYPES, get_similar_careers
from typing import Dict, List

# -----------------------------
# Confidence Level Descriptions
# -----------------------------
CONFIDENCE_DESCRIPTIONS = {
    "high": "very confident",
    "medium": "moderately confident",
    "low": "less confident"
}

# -----------------------------
# Generate Human-Readable Explanation
# -----------------------------
def generate_explanation(result: Dict) -> str:
    """
    Generate a natural language explanation of the prediction
    """
    
    text = result["input_text"]
    predictions = result["predictions"]
    context = result["context"]
    ambiguity = result["ambiguity"]
    
    if not predictions:
        return "I couldn't identify a clear career match from your description. Could you provide more details?"
    
    # Top prediction
    top = predictions[0]
    career = top["career"].replace("_", " ").title()
    fusion_score = top["fusion_score"]
    confidence = top["confidence_level"]
    popularity = top["expected_popularity"]
    motivations = top["motivations"]
    traits = top["traits"]
    
    # Build explanation
    explanation = []
    
    # Opening statement
    explanation.append(
        f"Based on your description, I'm {CONFIDENCE_DESCRIPTIONS[confidence]} "
        f"that you dreamed of becoming a **{career}**."
    )
    
    # Why this career?
    reasoning = []
    
    # Semantic reasoning
    if top["semantic_score"] > 0.6:
        reasoning.append(
            f"Your description strongly aligns with typical {career.lower()} aspirations"
        )
    elif top["semantic_score"] > 0.4:
        reasoning.append(
            f"Your description has clear connections to the {career.lower()} profession"
        )
    
    # Keyword evidence
    if top["keyword_matches"] >= 2:
        reasoning.append(
            f"you mentioned key concepts associated with this career"
        )
    
    # TF-IDF evidence
    if top["tfidf_score"] > 0.3:
        reasoning.append(
            f"your phrasing matches common {career.lower()} aspirations"
        )
    
    if reasoning:
        explanation.append(
            f"\n**Why {career}?** " + ", and ".join(reasoning).capitalize() + "."
        )
    
    # Popularity insight
    popularity_text = {
        1: "very specialized and less commonly pursued",
        2: "stable and respectable",
        3: "moderately popular",
        4: "quite popular and aspirational",
        5: "extremely popular and highly aspirational"
    }
    
    explanation.append(
        f"\n**Popularity:** This career is {popularity_text.get(popularity, 'moderately popular')} "
        f"among childhood dreams (level {popularity}/5)."
    )
    
    # Motivations
    motivation_descriptions = {
        "helping_others": "helping and caring for others",
        "recognition": "achieving recognition and fame",
        "creativity": "expressing creativity and imagination",
        "intellectual": "intellectual curiosity and learning",
        "power": "leadership and influence",
        "wealth": "financial success",
        "adventure": "excitement and adventure",
        "independence": "independence and autonomy",
        "competition": "competitive achievement",
        "expression": "self-expression",
        "justice": "justice and fairness",
        "discovery": "discovery and exploration",
        "problem_solving": "solving complex problems",
        "impact": "making a meaningful impact",
        "passion": "following a deep passion",
        "stability": "job security and stability",
        "prestige": "social prestige",
        "service": "serving the community",
        "heroism": "being heroic",
        "innovation": "innovation and creation",
        "truth": "seeking truth",
        "communication": "communication and storytelling",
        "travel": "travel and exploration",
        "legacy": "leaving a legacy",
        "understanding": "understanding people",
        "compassion": "compassion for others",
        "conservation": "conservation and protection",
        "documentation": "documenting reality"
    }
    
    motivation_text = [
        motivation_descriptions.get(m, m.replace("_", " "))
        for m in motivations[:2]
    ]
    
    explanation.append(
        f"\n**Likely motivations:** Your dream suggests you were driven by "
        f"{' and '.join(motivation_text)}."
    )
    
    # Personality traits
    trait_descriptions = {
        "extroverted": "outgoing and social",
        "creative": "imaginative and artistic",
        "analytical": "logical and systematic",
        "compassionate": "empathetic and caring",
        "ambitious": "goal-oriented and driven",
        "risk-taker": "comfortable with uncertainty",
        "leader": "natural leadership qualities",
        "expressive": "emotionally expressive",
        "detail-oriented": "attentive to details",
        "athletic": "physically active",
        "competitive": "thrives on competition",
        "disciplined": "self-controlled and focused",
        "curious": "intellectually curious",
        "methodical": "organized and systematic",
        "patient": "calm and persistent",
        "communicative": "skilled communicator",
        "nurturing": "supportive of others",
        "confident": "self-assured",
        "argumentative": "enjoys debate",
        "charismatic": "naturally influential",
        "introspective": "reflective and thoughtful",
        "observant": "highly perceptive",
        "brave": "courageous",
        "strong": "physically strong",
        "selfless": "puts others first",
        "protective": "watches over others",
        "trendy": "fashion-conscious",
        "visionary": "forward-thinking",
        "empathetic": "understands others' feelings",
        "passionate": "deeply enthusiastic",
        "adventurous": "seeks new experiences",
        "resilient": "bounces back from setbacks",
        "persistent": "doesn't give up easily",
        "calm": "composed under pressure",
        "team-player": "works well with others",
        "imaginative": "highly creative",
        "technical": "technically skilled",
        "dedicated": "deeply committed"
    }
    
    trait_text = [
        trait_descriptions.get(t, t.replace("_", " "))
        for t in traits[:3]
    ]
    
    explanation.append(
        f"\n**Personality fit:** This career typically suits people who are "
        f"{', '.join(trait_text[:-1])}, and {trait_text[-1]}."
    )
    
    # Context insights
    age_value = context["age_context"]["value"]
    intensity_value = context["intensity"]["value"]
    
    age_text = {
        "very_young": "very early childhood (3-6 years)",
        "young": "young childhood (6-10 years)",
        "preteen": "pre-teen years (10-13 years)",
        "teen": "teenage years (13-18 years)"
    }
    
    intensity_text = {
        "strong": "very passionate",
        "moderate": "interested",
        "mild": "casually interested"
    }
    
    if context["age_context"]["confidence"] > 0.4:
        explanation.append(
            f"\n**Age context:** Your dream likely emerged during {age_text.get(age_value, 'childhood')}."
        )
    
    if context["intensity"]["confidence"] > 0.4:
        explanation.append(
            f"\n**Intensity:** You seemed {intensity_text.get(intensity_value, 'interested')} "
            f"about this aspiration."
        )
    
    # Ambiguity note
    if ambiguity["is_ambiguous"]:
        alt_career = predictions[1]["career"].replace("_", " ").title()
        explanation.append(
            f"\n⚠️ **Note:** Your description could also suggest aspirations toward becoming a "
            f"**{alt_career}**. The ambiguity is {ambiguity['ambiguity_level']}."
        )
    
    # Alternative careers
    if len(predictions) > 1:
        alternatives = [
            p["career"].replace("_", " ").title()
            for p in predictions[1:3]
        ]
        explanation.append(
            f"\n**Other possibilities:** {', '.join(alternatives)}."
        )
    
    return "\n".join(explanation)

# -----------------------------
# Career Comparison
# -----------------------------
def compare_careers(career1: str, career2: str) -> str:
    """
    Compare two careers and explain similarities/differences
    """
    
    if career1 not in CAREER_PROTOTYPES or career2 not in CAREER_PROTOTYPES:
        return "One or both careers not found in database."
    
    c1_data = CAREER_PROTOTYPES[career1]
    c2_data = CAREER_PROTOTYPES[career2]
    
    # Find common motivations
    common_motivations = set(c1_data["motivations"]) & set(c2_data["motivations"])
    
    # Find common traits
    common_traits = set(c1_data["traits"]) & set(c2_data["traits"])
    
    # Popularity comparison
    pop_diff = abs(c1_data["popularity"] - c2_data["popularity"])
    
    comparison = []
    
    comparison.append(
        f"**Comparing {career1.replace('_', ' ').title()} vs "
        f"{career2.replace('_', ' ').title()}:**\n"
    )
    
    if common_motivations:
        comparison.append(
            f"**Shared motivations:** Both careers involve "
            f"{', '.join(common_motivations)}."
        )
    
    if common_traits:
        comparison.append(
            f"\n**Shared traits:** Both suit people who are "
            f"{', '.join(common_traits)}."
        )
    
    if pop_diff <= 1:
        comparison.append(
            f"\n**Popularity:** Both are similarly popular as childhood dreams."
        )
    else:
        more_popular = career1 if c1_data["popularity"] > c2_data["popularity"] else career2
        comparison.append(
            f"\n**Popularity:** {more_popular.replace('_', ' ').title()} is more commonly "
            f"dreamed about."
        )
    
    # Compute similarity
    similar_dict = dict(get_similar_careers(career1, top_k=20))
    similarity = similar_dict.get(career2, 0.0)
    
    if similarity > 0.7:
        comparison.append(
            f"\n**Overall:** These careers are very similar (similarity: {similarity:.2f})."
        )
    elif similarity > 0.5:
        comparison.append(
            f"\n**Overall:** These careers have moderate similarity (similarity: {similarity:.2f})."
        )
    else:
        comparison.append(
            f"\n**Overall:** These careers are quite different (similarity: {similarity:.2f})."
        )
    
    return "\n".join(comparison)

# -----------------------------
# Main Explanation Function
# -----------------------------
def explain_prediction(text: str, include_comparison: bool = False) -> Dict:
    """
    Generate complete explanation for a prediction
    """
    
    # Get full prediction
    result = complete_prediction(text, top_k=5)
    
    # Generate explanation
    explanation = generate_explanation(result)
    
    response = {
        "input": text,
        "explanation": explanation,
        "predictions": result["predictions"],
        "context": result["context"],
        "ambiguity": result["ambiguity"]
    }
    
    # Optional career comparison
    if include_comparison and len(result["predictions"]) >= 2:
        career1 = result["predictions"][0]["career"]
        career2 = result["predictions"][1]["career"]
        response["comparison"] = compare_careers(career1, career2)
    
    return response

# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    test_cases = [
        "I loved performing for people but also dreamed of building something big and leading others",
        "Ever since I was 5, I wanted to help sick people get better",
        "I dreamed of scoring the winning goal in the World Cup final",
        "I wanted to discover new planets and explore the universe"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}")
        print(f"{'=' * 80}\n")
        
        result = explain_prediction(text, include_comparison=True)
        
        print(f"INPUT: {result['input']}\n")
        print(result['explanation'])
        
        if "comparison" in result:
            print(f"\n{result['comparison']}")
        
        print(f"\n{'-' * 80}\n")

        # 🔧 Computing career prototype embeddings...
# ✅ 30 career prototypes loaded
# 📦 Loading base models...
# ✅ Base models loaded

# 📊 Loading and sampling data...
# Loaded 200,000 rows

# 🔧 Generating training features...
# Processing 50,000 positive samples...
#   Positive samples: 10,000
#   Positive samples: 20,000
#   Positive samples: 30,000
#   Positive samples: 40,000
#   Positive samples: 50,000
# Processing 50,000 negative samples...
#   Total samples: 60,000
#   Total samples: 70,000
#   Total samples: 80,000
#   Total samples: 90,000
#   Total samples: 100,000

# ✅ Feature matrix: (100000, 18)
#    Features: 18
#    Positive samples: 50000
#    Negative samples: 50000

# Train size: 80,000
# Test size:  20,000

# 🚀 Training Gradient Boosting fusion model...
#       Iter       Train Loss      OOB Improve   Remaining Time 
#          1           1.2116           0.1749            1.51m
#          2           1.0688           0.1439            1.63m
#          3           0.9495           0.1179            1.66m
#          4           0.8494           0.1014            1.65m
#          5           0.7633           0.0832            1.66m
#          6           0.6899           0.0738            1.66m
#          7           0.6268           0.0648            1.66m
#          8           0.5712           0.0532            1.66m
#          9           0.5242           0.0522            1.65m
#         10           0.4811           0.0378            1.65m
#         20           0.2554           0.0119            1.57m
#         30           0.1824           0.0031            1.50m
#         40           0.1568          -0.0116            1.41m
#         50           0.1475          -0.0057            1.32m
#         60           0.1445          -0.0065            1.23m
#         70           0.1413          -0.0009            1.15m
#         80           0.1419           0.0153            1.06m
#         90           0.1346          -0.0036           58.19s
#        100           0.1358          -0.0018           52.90s
#        200           0.1190          -0.0028            0.00s
# ✅ Training complete

# 📈 Model Evaluation:

# Classification Report:
#               precision    recall  f1-score   support

#            0       1.00      0.96      0.98     10000
#            1       0.96      1.00      0.98     10000

#     accuracy                           0.98     20000
#    macro avg       0.98      0.98      0.98     20000
# weighted avg       0.98      0.98      0.98     20000


# ROC AUC Score: 0.9923

# 🔍 Top 10 Feature Importances:
#   tfidf_career                   0.9647
#   pop_alignment                  0.0194
#   confidence_product             0.0025
#   personality_confidence         0.0021
#   tfidf_entropy                  0.0020
#   tfidf_semantic_product         0.0019
#   tfidf_popularity               0.0013
#   tfidf_max                      0.0011
#   motivation_confidence          0.0010
#   semantic_score                 0.0010

# 💾 Saving fusion model...
# ✅ Fusion model v3 saved
#    Location: /Users/ritikbansal/Desktop/CloutSimAI/models/fusion_model_v3.pkl
#    Test AUC: 0.9923