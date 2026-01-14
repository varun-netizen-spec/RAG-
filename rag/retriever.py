import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("market_index.faiss")

with open("data/context.txt") as f:
    CONTEXT = f.readlines()

def retrieve_context(query, k=5):
    query_vector = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vector, k)
    return [CONTEXT[i].strip() for i in indices[0]]
