from sentence_transformers import SentenceTransformer
import numpy as np

#load pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text:str)->np.ndarray:
    """
    Converts text into sementic embedding vector
    """
    embedding=model.encode(text,normalize_embeddings=True)
    return embedding

if __name__=="__main__":
    t1 = "I loved performing for people"
    t2 = "I enjoy being on stage and entertaining crowds"
    t3 = "I wanted to build a company and lead others"

    e1=get_embedding(t1)
    e2=get_embedding(t2)
    e3=get_embedding(t3)

    #cosine similarity
    def cosine(a,b):
        return np.dot(a,b)
    
    print("Similarity(perform vs stage)",cosine(e1,e2))
    print("Similarity(perform vs build)",cosine(e1,e3))