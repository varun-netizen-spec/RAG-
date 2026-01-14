import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/market_prices.csv")

texts = [
    f"{row['product']} price was {row['price']} rupees in {row['location']} on {row['date']}"
    for _, row in df.iterrows()
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "market_index.faiss")

with open("data/seasonal.txt", "w") as f:
    for t in texts:
        f.write(t + "\n")

print("✅ FAISS index and context created")
