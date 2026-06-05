import json
import os
import pandas as pd
from tqdm import tqdm

INPUT_FILE = "data/arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "data/arxiv_subset.parquet"
MAX_RECORDS = 5000

os.makedirs("data", exist_ok=True)


def extract_year(paper: dict) -> int:
    try:
        versions = paper.get("versions", [])
        if versions:
            created = versions[0].get("created", "")
            return int(created.split()[-2])
    except Exception:
        pass

    update_date = paper.get("update_date", "2000-01-01")
    return int(update_date[:4])


def format_authors(paper: dict) -> str:
    parsed = paper.get("authors_parsed", [])

    if parsed:
        authors = []
        for author in parsed[:10]:
            last = author[0].strip() if len(author) > 0 else ""
            first = author[1].strip() if len(author) > 1 else ""

            if last:
                authors.append(f"{last} {first}".strip())

        return ", ".join(authors)

    return paper.get("authors", "").replace("\n", " ")


records = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in tqdm(file, desc="Reading dataset"):
        if len(records) >= MAX_RECORDS:
            break

        line = line.strip()
        if not line:
            continue

        paper = json.loads(line)

        title = paper.get("title", "").replace("\n", " ").strip()
        abstract = paper.get("abstract", "").replace("\n", " ").strip()

        if not title or not abstract:
            continue

        categories_raw = paper.get("categories", "unknown")
        primary_category = categories_raw.split()[0]

        records.append({
            "id": paper.get("id"),
            "title": title,
            "abstract": abstract,
            "authors": format_authors(paper),
            "year": extract_year(paper),
            "category": primary_category,
        })


df = pd.DataFrame(records)

print(f"\nPrepared records: {len(df)}")
print("\nTop categories:")
print(df["category"].value_counts().head(10))

print("\nYears:")
print(df["year"].value_counts().sort_index().tail(10))

print("\nSample record:")
print(df.iloc[0].to_dict())

df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")