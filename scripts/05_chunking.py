import os
import re
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

FIXED_INDEX = "arxiv-chunks-fixed"
SEMANTIC_INDEX = "arxiv-chunks-semantic"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)

df = pd.read_parquet("data/arxiv_subset.parquet")


def create_index_if_needed(index_name: str):
    existing_indexes = [index["name"] for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating index: {index_name}")

        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        while not pc.describe_index(index_name).status["ready"]:
            print(f"Waiting for {index_name}...")
            time.sleep(5)

    return pc.Index(index_name)


def split_fixed(text: str, chunk_size: int = 120, overlap: int = 20):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))

    return chunks


def split_semantic(text: str, max_chunk_size: int = 180):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def prepare_chunk_vectors(df_subset, split_function, chunk_type: str):
    vectors = []

    for row_id, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"Preparing {chunk_type} chunks"):
        abstract = str(row["abstract"])
        chunks = split_function(abstract)

        for chunk_id, chunk_text in enumerate(chunks):
            embedding = model.encode(chunk_text, normalize_embeddings=True).tolist()

            vector_id = f"{row_id}_{chunk_type}_{chunk_id}"

            metadata = {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"])[:500],
                "chunk_text": chunk_text[:1000],
                "chunk_number": int(chunk_id),
                "year": int(row["year"]),
                "category": str(row["category"]),
            }

            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

    return vectors


def upload_vectors(index, vectors, batch_size: int = 100):
    for start in tqdm(range(0, len(vectors), batch_size), desc="Uploading chunks"):
        end = min(start + batch_size, len(vectors))
        index.upsert(vectors=vectors[start:end])


def search_chunks(index, query: str, top_k: int = 5):
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results


def print_results(title: str, results):
    print(f"\n=== {title} ===")

    for i, match in enumerate(results["matches"], start=1):
        metadata = match["metadata"]

        print(f"\n#{i}")
        print(f"Score: {match['score']:.4f}")
        print(f"Title: {metadata.get('title')}")
        print(f"Chunk text: {metadata.get('chunk_text')[:500]}...")


df_subset = (
    df.assign(abstract_length=df["abstract"].astype(str).str.len())
    .sort_values("abstract_length", ascending=False)
    .head(30)
)

fixed_index = create_index_if_needed(FIXED_INDEX)
semantic_index = create_index_if_needed(SEMANTIC_INDEX)

fixed_vectors = prepare_chunk_vectors(df_subset, split_fixed, "fixed")
semantic_vectors = prepare_chunk_vectors(df_subset, split_semantic, "semantic")

upload_vectors(fixed_index, fixed_vectors)
upload_vectors(semantic_index, semantic_vectors)

queries = [
    "teaching machines to recognize objects in pictures",
    "deep learning for natural language processing",
    "transformer models for computer vision"
]

for query in queries:
    fixed_results = search_chunks(fixed_index, query)
    print_results(f"Fixed chunks: {query}", fixed_results)

    semantic_results = search_chunks(semantic_index, query)
    print_results(f"Semantic chunks: {query}", semantic_results)