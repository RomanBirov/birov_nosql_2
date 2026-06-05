import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from datetime import datetime

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")
embeddings = np.load("embeddings/embeddings.npy")


def encode_query(query: str):
    return model.encode(query, normalize_embeddings=True).tolist()


def search(query: str, top_k: int = TOP_K, filter_query=None):
    query_vector = encode_query(query)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_query
    )

    return results


def print_results(title: str, results):
    print(f"\n=== {title} ===")

    if not results["matches"]:
        print("No results found.")
        return

    for i, match in enumerate(results["matches"], start=1):
        metadata = match["metadata"]

        print(f"\n#{i}")
        print(f"Score: {match['score']:.4f}")
        print(f"Title: {metadata.get('title')}")
        print(f"Category: {metadata.get('category')}")
        print(f"Year: {metadata.get('year')}")
        print(f"Authors: {metadata.get('authors')}")
        print(f"Abstract: {metadata.get('abstract')[:300]}...")


def compare_local_metrics(query: str, top_k: int = TOP_K):
    query_embedding = np.array(encode_query(query))

    rows = []

    for i, emb in enumerate(embeddings):
        cosine = float(np.dot(query_embedding, emb))
        dot_product = float(np.dot(query_embedding, emb))
        l2_distance = float(np.linalg.norm(query_embedding - emb))

        rows.append({
            "index": i,
            "title": df.iloc[i]["title"],
            "cosine": cosine,
            "dot_product": dot_product,
            "l2_distance": l2_distance
        })

    metric_results = pd.DataFrame(rows)

    print("\n=== Local metric comparison ===")

    print("\nTop by cosine:")
    print(metric_results.sort_values("cosine", ascending=False).head(top_k)[["title", "cosine"]])

    print("\nTop by dot product:")
    print(metric_results.sort_values("dot_product", ascending=False).head(top_k)[["title", "dot_product"]])

    print("\nTop by L2 distance:")
    print(metric_results.sort_values("l2_distance", ascending=True).head(top_k)[["title", "l2_distance"]])


query_1 = "teaching machines to recognize objects in pictures"
results_1 = search(query_1)
print_results("Semantic search", results_1)

current_year = datetime.now().year

filter_a = {
    "$and": [
        {"category": {"$eq": "cs.LG"}},
        {"year": {"$gte": current_year - 5}}
    ]
}

results_2 = search("reinforcement learning", filter_query=filter_a)
print_results("Filtered search: reinforcement learning, cs.LG, last 5 years", results_2)

filter_b = {
    "year": {"$lte": 2015}
}

results_3 = search("machine learning", filter_query=filter_b)
print_results("Filtered search: older papers before 2015", results_3)

compare_local_metrics(query_1)