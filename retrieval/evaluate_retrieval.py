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

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "qa_dataset"
    / "processed"
    / "test.jsonl"
)

SECTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sections.jsonl"
)

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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_evaluation.json"
)


# ============================================================
# LOADERS
# ============================================================

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Load JSONL records."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    records: list[dict[str, Any]] = []

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
                    f"{line_number} in {path}"
                ) from exc

            records.append(record)

    return records


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# GOLD RECORD IDENTIFICATION
# ============================================================

def find_gold_record(
    test_record: dict[str, Any],
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Find the structured source record for a QA example.

    The QA context may represent only part of the original
    structured record, so exact equality is too strict.

    We first restrict candidates to the same document_id and
    then look for records whose text contains the QA context.

    If multiple records match, prefer the shortest matching
    structured record because it is the most specific source.
    """

    qa_context = test_record["context"]
    qa_document_id = test_record["document_id"]

    candidates = [
        record
        for record in sections
        if record.get("document_id")
        == qa_document_id
        and isinstance(
            record.get("text"),
            str,
        )
        and qa_context in record["text"]
    ]

    if not candidates:
        raise ValueError(
            "Could not find structured source record "
            f"for QA ID {test_record['id']}.\n"
            f"Document ID: {qa_document_id}"
        )

    # Prefer the most specific/shortest matching record.
    candidates.sort(
        key=lambda record: len(
            record["text"]
        )
    )

    if len(candidates) > 1:
        # We don't silently ignore ambiguity.
        # If multiple records contain exactly the same context,
        # the first/shortest candidate is selected, but we report it.
        selected = candidates[0]

        print(
            f"\nWARNING: Multiple source records matched "
            f"{test_record['id']}"
        )

        print(
            "Selected:",
            selected["record_id"],
        )

        print(
            "Other matches:",
            [
                candidate["record_id"]
                for candidate in candidates[1:]
            ],
        )

        return selected

    return candidates[0]


def build_record_lookup(
    sections: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index structured records by record_id."""

    return {
        record["record_id"]: record
        for record in sections
    }


# ============================================================
# EMBEDDING / SIMILARITY
# ============================================================

def embed_queries(
    model: SentenceTransformer,
    queries: list[str],
) -> np.ndarray:
    """Generate normalized query embeddings."""

    embeddings = model.encode(
        queries,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


def cosine_similarity(
    query_vector: np.ndarray,
    document_vector: np.ndarray,
) -> float:
    """
    Calculate cosine similarity.

    Both vectors are normalized, so this is simply
    their dot product.
    """

    return float(
        np.dot(
            query_vector,
            document_vector,
        )
    )


def retrieve_top_k(
    query_vector: np.ndarray,
    document_embeddings: np.ndarray,
    k: int,
) -> list[int]:
    """Return top-k document vector indices."""

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    scores = (
        document_embeddings
        @ query_vector
    )

    sorted_indices = np.argsort(
        scores
    )[::-1]

    return [
        int(index)
        for index in sorted_indices[:k]
    ]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PHASE 3D - AUTOMATED RETRIEVAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test QA data
    # --------------------------------------------------------

    test_records = load_jsonl(
        TEST_PATH
    )

    # --------------------------------------------------------
    # Load structured source records
    # --------------------------------------------------------

    sections = load_jsonl(
        SECTIONS_PATH
    )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    # --------------------------------------------------------
    # Load embedding metadata
    # --------------------------------------------------------

    embedding_metadata = load_json(
        METADATA_PATH
    )

    print(
        f"Test questions : "
        f"{len(test_records)}"
    )

    print(
        f"Document records: "
        f"{len(sections)}"
    )

    print(
        f"Embedding matrix: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # Validate basic dimensions
    # --------------------------------------------------------

    if embeddings.ndim != 2:
        raise ValueError(
            "Embedding matrix must be 2-dimensional."
        )

    if embeddings.shape[0] != len(sections):
        raise ValueError(
            "Number of embedding vectors does not "
            "match number of structured records."
        )

    if embedding_metadata.get(
        "record_count"
    ) != len(sections):

        raise ValueError(
            "Embedding metadata record count does "
            "not match structured record count."
        )

    metadata_records = embedding_metadata[
        "records"
    ]

    if len(metadata_records) != len(
        sections
    ):
        raise ValueError(
            "Embedding metadata records do not "
            "match structured record count."
        )

    # --------------------------------------------------------
    # Build record lookup
    # --------------------------------------------------------

    record_lookup = build_record_lookup(
        sections
    )

    # --------------------------------------------------------
    # Find gold source record for each QA example
    # --------------------------------------------------------

    print(
        "\nIdentifying gold source records..."
    )

    gold_records: list[
        dict[str, Any]
    ] = []

    for test_record in test_records:

        gold = find_gold_record(
            test_record=test_record,
            sections=sections,
        )

        gold_records.append(gold)

        print(
            f"  {test_record['id']}"
            f" -> "
            f"{gold['record_id']}"
        )

    # --------------------------------------------------------
    # Load BGE
    # --------------------------------------------------------

    print(
        f"\nLoading model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Embed queries
    # --------------------------------------------------------

    queries = [
        record["question"]
        for record in test_records
    ]

    print(
        "\nEmbedding test questions..."
    )

    query_embeddings = embed_queries(
        model=model,
        queries=queries,
    )

    if query_embeddings.shape[0] != len(
        test_records
    ):
        raise ValueError(
            "Number of query embeddings does not "
            "match number of test questions."
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    result_records: list[
        dict[str, Any]
    ] = []

    recall_at_1_hits = 0
    recall_at_3_hits = 0
    recall_at_5_hits = 0

    total = len(test_records)

    for index, test_record in enumerate(
        test_records
    ):

        query_vector = query_embeddings[
            index
        ]

        gold_record = gold_records[
            index
        ]

        gold_record_id = gold_record[
            "record_id"
        ]

        # --------------------------------------------
        # Retrieve top 5
        # --------------------------------------------

        top5_indices = retrieve_top_k(
            query_vector=query_vector,
            document_embeddings=embeddings,
            k=5,
        )

        top5_results: list[
            dict[str, Any]
        ] = []

        for rank, vector_index in enumerate(
            top5_indices,
            start=1,
        ):

            if vector_index >= len(
                metadata_records
            ):
                raise ValueError(
                    f"Invalid vector index: "
                    f"{vector_index}"
                )

            metadata_record = (
                metadata_records[
                    vector_index
                ]
            )

            retrieved_record_id = (
                metadata_record["record_id"]
            )

            if retrieved_record_id not in (
                record_lookup
            ):
                raise ValueError(
                    f"Retrieved record ID "
                    f"{retrieved_record_id} "
                    f"not found in sections.jsonl"
                )

            retrieved_record = record_lookup[
                retrieved_record_id
            ]

            score = cosine_similarity(
                query_vector,
                embeddings[vector_index],
            )

            top5_results.append(
                {
                    "rank": rank,
                    "record_id": retrieved_record_id,
                    "document_id": retrieved_record[
                        "document_id"
                    ],
                    "document_title": retrieved_record[
                        "document_title"
                    ],
                    "section": retrieved_record.get(
                        "section"
                    ),
                    "subsection": retrieved_record.get(
                        "subsection"
                    ),
                    "score": score,
                }
            )

        # --------------------------------------------
        # Recall@K
        # --------------------------------------------

        top1_ids = {
            item["record_id"]
            for item in top5_results[:1]
        }

        top3_ids = {
            item["record_id"]
            for item in top5_results[:3]
        }

        top5_ids = {
            item["record_id"]
            for item in top5_results[:5]
        }

        hit1 = (
            gold_record_id
            in top1_ids
        )

        hit3 = (
            gold_record_id
            in top3_ids
        )

        hit5 = (
            gold_record_id
            in top5_ids
        )

        recall_at_1_hits += int(hit1)
        recall_at_3_hits += int(hit3)
        recall_at_5_hits += int(hit5)

        result_records.append(
            {
                "id": test_record["id"],
                "question": test_record[
                    "question"
                ],
                "gold_record_id": (
                    gold_record_id
                ),
                "gold_document_id": (
                    gold_record["document_id"]
                ),
                "gold_section": (
                    gold_record.get("section")
                ),
                "recall_at_1": hit1,
                "recall_at_3": hit3,
                "recall_at_5": hit5,
                "top_5": top5_results,
            }
        )

        print(
            f"[{index + 1:02d}/{total:02d}] "
            f"R@1={int(hit1)} "
            f"R@3={int(hit3)} "
            f"R@5={int(hit5)} "
            f"Gold={gold_record_id}"
        )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    if total == 0:
        raise ValueError(
            "No test examples found."
        )

    recall_at_1 = (
        recall_at_1_hits / total
    )

    recall_at_3 = (
        recall_at_3_hits / total
    )

    recall_at_5 = (
        recall_at_5_hits / total
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "model": MODEL_NAME,
        "dataset": "test.jsonl",
        "test_examples": total,
        "document_records": len(sections),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "hits": {
            "recall_at_1": recall_at_1_hits,
            "recall_at_3": recall_at_3_hits,
            "recall_at_5": recall_at_5_hits,
        },
        "results": result_records,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"Recall@1 : "
        f"{recall_at_1 * 100:.2f}% "
        f"({recall_at_1_hits}/{total})"
    )

    print(
        f"Recall@3 : "
        f"{recall_at_3 * 100:.2f}% "
        f"({recall_at_3_hits}/{total})"
    )

    print(
        f"Recall@5 : "
        f"{recall_at_5 * 100:.2f}% "
        f"({recall_at_5_hits}/{total})"
    )

    print(
        f"\nDetailed results:"
        f"\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()