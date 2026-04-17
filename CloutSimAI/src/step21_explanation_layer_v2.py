from step19_hybrid_fusion_v2 import hybrid_predict
from step18_semantic_embeddings_v2 import semantic_scores

# -----------------------------
# Thresholds (product tuning)
# -----------------------------
SEMANTIC_HIGH = 0.6
SEMANTIC_MEDIUM = 0.4

# -----------------------------
# Explanation logic (v2)
# -----------------------------
def explain_prediction(text: str):
    # Hybrid prediction (fusion v2)
    hybrid_results = hybrid_predict(text)

    if not hybrid_results:
        return {
            "input": text,
            "final_decision": None,
            "related_careers": []
        }

    # Top decision
    top = hybrid_results[0]
    final_career = top["career"]
    final_conf = round(top["fusion_score"], 3)
    final_popularity = top["expected_popularity"]

    # Semantic scores (v2)
    sem_data = semantic_scores(text)

    # Related careers explanation
    related = []
    for career, data in sorted(
        sem_data.items(),
        key=lambda x: x[1]["semantic_score"],
        reverse=True
    ):
        if career == final_career:
            continue

        score = data["semantic_score"]

        if score >= SEMANTIC_HIGH:
            level = "high"
        elif score >= SEMANTIC_MEDIUM:
            level = "medium"
        else:
            continue

        related.append({
            "career": career,
            "semantic_relevance": level,
            "expected_popularity": data["expected_popularity"]
        })

        if len(related) == 3:
            break

    return {
        "input": text,
        "final_decision": {
            "career": final_career,
            "confidence": final_conf,
            "expected_popularity": final_popularity
        },
        "related_careers": related
    }


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    test_text = (
        "I loved performing for people but also dreamed of building something big "
        "and leading others"
    )

    result = explain_prediction(test_text)

    print("\n🧠 AI CAREER EXPLANATION (v2 - FINAL)\n")
    print("Input:")
    print(result["input"])

    if result["final_decision"]:
        print("\nFinal Career Decision:")
        print(
            f"  {result['final_decision']['career']} "
            f"(confidence: {result['final_decision']['confidence']}, "
            f"expected popularity: {result['final_decision']['expected_popularity']})"
        )
    else:
        print("\nNo confident career decision.")

    if result["related_careers"]:
        print("\nAlso Related Careers:")
        for r in result["related_careers"]:
            print(
                f"  {r['career']} "
                f"(semantic relevance: {r['semantic_relevance']}, "
                f"expected popularity: {r['expected_popularity']})"
            )
    else:
        print("\nNo strong alternative semantic matches.")
