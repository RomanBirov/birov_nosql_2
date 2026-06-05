import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


def tokenize(text: str):
    return text.lower().split()


def build_bm25_index(df):
    corpus = (df["title"] + " " + df["abstract"]).tolist()
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def search_bm25(bm25, query: str, top_k: int = TOP_K):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indexes = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "id": f"paper_{idx}",
            "title": df.iloc[idx]["title"],
            "score": float(scores[idx])
        }
        for idx in top_indexes
    ]


def search_vector(query: str, top_k: int = TOP_K):
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return [
        {
            "id": match["id"],
            "title": match["metadata"]["title"],
            "score": float(match["score"])
        }
        for match in results["matches"]
    ]


def reciprocal_rank_fusion(bm25_results, vector_results, k: int = 60):
    scores = {}

    for rank, item in enumerate(bm25_results, start=1):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    for rank, item in enumerate(vector_results, start=1):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_docs[:TOP_K]


def print_results(title: str, results):
    print(f"\n=== {title} ===")

    for i, item in enumerate(results, start=1):
        print(f"{i}. {item['title']} | score={item['score']:.4f}")


bm25 = build_bm25_index(df)

test_queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text"
]

for query in test_queries:
    print(f"\n\n==============================")
    print(f"Query: {query}")
    print(f"==============================")

    bm25_results = search_bm25(bm25, query)
    vector_results = search_vector(query)
    hybrid_ids = reciprocal_rank_fusion(bm25_results, vector_results, k=60)

    print_results("BM25 results", bm25_results[:5])
    print_results("Vector search results", vector_results[:5])

    print("\n=== Hybrid search with RRF ===")
    for rank, (doc_id, rrf_score) in enumerate(hybrid_ids[:5], start=1):
        index_number = int(doc_id.replace("paper_", ""))
        title = df.iloc[index_number]["title"]

        print(f"{rank}. {title} | RRF score={rrf_score:.4f}")