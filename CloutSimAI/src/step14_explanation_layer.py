import joblib
import numpy as np
from pathlib import Path
from step12_hybrid_embeddings import sementic_scores
from step12_hybrid_fusion import hybrid_predict

#thresholds(product tuning)
SEMANTIC_HIGH = 0.6
SEMANTIC_MEDIUM = 0.4

def explain_prediction(text:str):
    final_ranked=hybrid_predict(text)
    final_career,final_conf=final_ranked[0]

    #sementic explanation
    sem_scores=sementic_scores(text)

    related=[]
    for career,score in sorted(sem_scores.items(),key=lambda x:x[1],reverse=True):
        if career==final_career:
            continue

        if score>=SEMANTIC_HIGH:
            related.append((career,"high"))
        elif score>=SEMANTIC_MEDIUM:
            related.append((career,"medium"))

    return {
        "input":text,
        "final_decision":{
            "career":final_career,
            "confidence":round(final_conf,3)
        },
        "related_careers":related[:3]
    }

#test
if __name__ == "__main__":
    text = (
        "I loved performing for people but also dreamed of building something big and leading others"
    )

    result = explain_prediction(text)

    print("\n🧠 AI CAREER EXPLANATION\n")
    print("Input:")
    print(result["input"])

    print("\nFinal Career Decision:")
    print(
        f"  {result['final_decision']['career']} "
        f"(confidence: {result['final_decision']['confidence']})"
    )

    if result["related_careers"]:
        print("\nAlso Related Careers:")
        for c, lvl in result["related_careers"]:
            print(f"  {c} (semantic relevance: {lvl})")
    else:
        print("\nNo strong alternative semantic matches.")