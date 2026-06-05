import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"

os.makedirs("embeddings", exist_ok=True)

df = pd.read_parquet(INPUT_FILE)

texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

print(f"Texts to embed: {len(texts)}")
print(f"Loading model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"Embeddings shape: {embeddings.shape}")
print(f"First embedding norm: {np.linalg.norm(embeddings[0]):.4f}")

np.save(OUTPUT_FILE, embeddings)
print(f"Saved embeddings to {OUTPUT_FILE}")