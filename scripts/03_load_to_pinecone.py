import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"

INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

api_key = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=api_key)

existing_indexes = [index["name"] for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("Waiting for index...")
        time.sleep(5)

index = pc.Index(INDEX_NAME)

df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

print(f"Documents: {len(df)}")
print(f"Embeddings: {embeddings.shape}")

for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Uploading to Pinecone"):
    end = min(start + BATCH_SIZE, len(df))

    batch_df = df.iloc[start:end]
    batch_embeddings = embeddings[start:end]

    vectors = []

    for i, row in batch_df.iterrows():
        vector_id = f"paper_{i}"

        metadata = {
            "arxiv_id": str(row["id"]),
            "title": str(row["title"])[:500],
            "abstract": str(row["abstract"])[:500],
            "authors": str(row["authors"])[:200],
            "year": int(row["year"]),
            "category": str(row["category"]),
        }

        vectors.append({
            "id": vector_id,
            "values": batch_embeddings[i - start].tolist(),
            "metadata": metadata
        })

    index.upsert(vectors=vectors)

stats = index.describe_index_stats()

print("\nIndex stats:")
print(stats)
print(f"\nTotal vectors in index '{INDEX_NAME}': {stats['total_vector_count']}")