from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings.npy"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embedding_metadata.json"
)

SECTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sections.jsonl"
)

DEFAULT_TOP_K = 5


# ============================================================
# LOAD DATA
# ============================================================

def load_embeddings(
    path: Path,
) -> np.ndarray:
    """Load the document embedding matrix."""

    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {path}"
        )

    embeddings = np.load(path)

    if embeddings.ndim != 2:
        raise ValueError(
            "Embedding matrix must be 2-dimensional."
        )

    return embeddings


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_sections(
    path: Path,
) -> dict[str, dict[str, Any]]:
    """
    Load the original structured records and index them by
    record_id.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Sections file not found: {path}"
        )

    records: dict[str, dict[str, Any]] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}"
                ) from exc

            record_id = record.get(
                "record_id"
            )

            if not record_id:
                raise ValueError(
                    f"Record at line "
                    f"{line_number} has no record_id."
                )

            records[record_id] = record

    return records


# ============================================================
# MODEL
# ============================================================

def load_embedding_model() -> SentenceTransformer:
    """Load the local embedding model."""

    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    return SentenceTransformer(
        MODEL_NAME
    )


# ============================================================
# QUERY REPRESENTATION
# ============================================================

def embed_query(
    model: SentenceTransformer,
    query: str,
) -> np.ndarray:
    """
    Convert the user query into the same 384-dimensional
    representation used for document records.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    query_vector = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32,
    )

    if query_vector.ndim != 1:
        query_vector = query_vector.reshape(
            -1
        )

    if query_vector.shape[0] != 384:
        raise ValueError(
            "Unexpected query embedding "
            f"dimension: {query_vector.shape[0]}"
        )

    return query_vector


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarities(
    query_vector: np.ndarray,
    document_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between the query vector
    and every document vector.

    Because both query and document vectors are normalized,
    cosine similarity is equivalent to their dot product.
    """

    if query_vector.ndim != 1:
        raise ValueError(
            "Query vector must be 1-dimensional."
        )

    if document_embeddings.ndim != 2:
        raise ValueError(
            "Document embeddings must be 2-dimensional."
        )

    if query_vector.shape[0] != document_embeddings.shape[1]:
        raise ValueError(
            "Query/document vector dimensions do not match."
        )

    similarities = (
        document_embeddings @ query_vector
    )

    return similarities


# ============================================================
# SEARCH
# ============================================================

def search(
    query: str,
    model: SentenceTransformer,
    document_embeddings: np.ndarray,
    metadata: dict[str, Any],
    sections: dict[str, dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Search the local document embedding matrix and return
    the top-k most semantically similar records.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    query_vector = embed_query(
        model=model,
        query=query,
    )

    similarities = cosine_similarities(
        query_vector=query_vector,
        document_embeddings=document_embeddings,
    )

    # Get indices sorted by similarity, descending.
    sorted_indices = np.argsort(
        similarities
    )[::-1]

    top_indices = sorted_indices[
        :top_k
    ]

    metadata_records = metadata.get(
        "records",
        [],
    )

    results: list[dict[str, Any]] = []

    for index in top_indices:

        index = int(index)

        if index >= len(metadata_records):
            raise ValueError(
                f"Vector index {index} "
                f"has no metadata."
            )

        metadata_record = (
            metadata_records[index]
        )

        record_id = metadata_record[
            "record_id"
        ]

        original_record = sections.get(
            record_id
        )

        if original_record is None:
            raise ValueError(
                f"No structured record found "
                f"for {record_id}"
            )

        results.append(
            {
                "rank": len(results) + 1,
                "score": float(
                    similarities[index]
                ),
                "vector_index": index,
                "record_id": record_id,
                "document_id": original_record[
                    "document_id"
                ],
                "document_title": original_record[
                    "document_title"
                ],
                "source_file": original_record[
                    "source_file"
                ],
                "section": original_record.get(
                    "section"
                ),
                "subsection": original_record.get(
                    "subsection"
                ),
                "text": original_record[
                    "text"
                ],
            }
        )

    return results


# ============================================================
# DISPLAY
# ============================================================

def print_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    """Print retrieval results in a readable format."""

    print("\n" + "=" * 80)
    print("SEMANTIC RETRIEVAL RESULTS")
    print("=" * 80)

    print(
        f"Query: {query}"
    )

    print(
        f"Results: {len(results)}"
    )

    for result in results:

        print("\n" + "-" * 80)

        print(
            f"Rank       : {result['rank']}"
        )

        print(
            f"Similarity : "
            f"{result['score']:.6f}"
        )

        print(
            f"Record ID  : "
            f"{result['record_id']}"
        )

        print(
            f"Document   : "
            f"{result['document_title']}"
        )

        print(
            f"Section    : "
            f"{result['section']}"
        )

        print(
            f"Subsection : "
            f"{result['subsection']}"
        )

        print(
            f"Source     : "
            f"{result['source_file']}"
        )

        print("\nText:")
        print(result["text"])


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("PHASE 3C - LOCAL SEMANTIC RETRIEVAL")
    print("=" * 80)

    # --------------------------------------------------------
    # Load document embeddings
    # --------------------------------------------------------

    embeddings = load_embeddings(
        EMBEDDINGS_PATH
    )

    print(
        f"Embedding matrix: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = load_json(
        METADATA_PATH
    )

    sections = load_sections(
        SECTIONS_PATH
    )

    metadata_count = metadata.get(
        "record_count"
    )

    if metadata_count != len(sections):
        raise ValueError(
            "Metadata/section record count mismatch."
        )

    if embeddings.shape[0] != len(sections):
        raise ValueError(
            "Embedding count does not match "
            "section record count."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_embedding_model()

    # --------------------------------------------------------
    # Interactive search
    # --------------------------------------------------------

    print("\nEnter a query.")
    print(
        "Type 'exit' or 'quit' to stop."
    )

    while True:

        try:
            query = input(
                "\nQuery: "
            ).strip()

        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if query.lower() in {
            "exit",
            "quit",
        }:
            print("Exiting.")
            break

        if not query:
            print(
                "Please enter a non-empty query."
            )
            continue

        try:
            results = search(
                query=query,
                model=model,
                document_embeddings=embeddings,
                metadata=metadata,
                sections=sections,
                top_k=DEFAULT_TOP_K,
            )

            print_results(
                query=query,
                results=results,
            )

        except Exception as exc:
            print(
                f"\nSearch error: {exc}"
            )


if __name__ == "__main__":
    main()