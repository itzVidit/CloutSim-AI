from sentence_transformers import SentenceTransformer
import numpy as np

embedder=SentenceTransformer("all-MiniLM-L6-v2")

#prototypes
CAREER_PROTOTYPES = {
    "actor": [
        "acting on stage",
        "performing for audiences",
        "expressing emotions in front of people"
    ],
    "entrepreneur": [
        "building a company",
        "leading a startup",
        "creating a business organization"
    ],
    "engineer": [
        "solving technical problems",
        "building systems",
        "working with logic and code"
    ],
    "doctor": [
        "helping sick people",
        "working in hospitals",
        "healing patients"
    ],
    "teacher": [
        "teaching students",
        "explaining concepts",
        "guiding learners"
    ],
    "farmer": [
        "growing crops",
        "working in agriculture",
        "cultivating land"
    ],
    "electrician": [
        "fixing electrical wiring",
        "installing electrical systems",
        "working with circuits"
    ],
    "f1 driver": [
        "racing cars at high speed",
        "competing in motorsports",
        "driving in professional races"
    ],
    "boxer": [
        "training for boxing matches",
        "fighting in the ring",
        "competing in boxing championships"
    ],
    "cricketer": [
        "playing cricket professionally",
        "scoring runs and taking wickets",
        "representing a team in cricket"
    ],
    "footballer": [
        "playing football professionally",
        "scoring goals",
        "competing in football matches"
    ]
}

def embed(text:str):
    return embedder.encode(text,normalize_embeddings=True)

#precompute prototype embeddings
CAREER_EMBEDS={
    career: np.mean([embed(t) for t in texts],axis=0)
    for career,texts in CAREER_PROTOTYPES.items()
}

# normalisation needed to make comparable to tfidf scores
def sementic_scores(text:str):
    v=embed(text)

    # scores={}
    # for career,proto_vec in CAREER_EMBEDS.items():
    #     scores[career]=float(np.dot(v,proto_vec))
    # return scores
    raw_scores={
        career:float(np.dot(v,proto_vec))
        for career,proto_vec in CAREER_EMBEDS.items()
    }

    min_s=min(raw_scores.values())
    max_s=max(raw_scores.values())

    norm_scores={
        k:(v-min_s)/(max_s-min_s+1e-8)
        for k,v in raw_scores.items()
    }

    return norm_scores

if __name__=="__main__":
    test="I love performing for people but also dreamed of leading others"
    scores=sementic_scores(test)

    print("\nSementic similarity scores: ")
    for k,v in sorted(scores.items(),key=lambda x:x[1],reverse=True):
        print(k,round(v,3))