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

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sections.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

EMBEDDINGS_PATH = (
    OUTPUT_DIR
    / "embeddings.npy"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "embedding_metadata.json"
)


# ============================================================
# LOAD STRUCTURED RECORDS
# ============================================================

def load_records(
    path: Path,
) -> list[dict[str, Any]]:
    """Load structured records from JSONL."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
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
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


# ============================================================
# BUILD EMBEDDING TEXT
# ============================================================

def build_embedding_text(
    record: dict[str, Any],
) -> str:
    """
    Construct the text that will be embedded.

    We include the section hierarchy as context before the
    actual text. The source text itself is not modified.
    """

    hierarchy_parts = [
        record.get("section"),
        record.get("subsection"),
        record.get("subsubsection"),
    ]

    hierarchy = " > ".join(
        part
        for part in hierarchy_parts
        if isinstance(part, str)
        and part.strip()
    )

    text = record["text"].strip()

    if hierarchy:
        return (
            f"Section: {hierarchy}\n"
            f"Content: {text}"
        )

    return text


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(
    model: SentenceTransformer,
    records: list[dict[str, Any]],
) -> np.ndarray:
    """
    Generate one embedding vector per record.
    """

    texts = [
        build_embedding_text(record)
        for record in records
    ]

    print(
        f"\nGenerating embeddings for "
        f"{len(texts)} records..."
    )

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if not isinstance(
        embeddings,
        np.ndarray,
    ):
        embeddings = np.asarray(
            embeddings
        )

    return embeddings


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    path: Path,
) -> None:
    """
    Save enough metadata to map each vector back to its
    original structured record.
    """

    metadata_records: list[dict[str, Any]] = []

    for index, record in enumerate(records):

        metadata_records.append(
            {
                "vector_index": index,
                "record_id": record["record_id"],
                "document_id": record["document_id"],
                "document_title": record[
                    "document_title"
                ],
                "source_file": record[
                    "source_file"
                ],
                "section": record.get(
                    "section"
                ),
                "subsection": record.get(
                    "subsection"
                ),
                "subsubsection": record.get(
                    "subsubsection"
                ),
            }
        )

    metadata = {
        "embedding_model": MODEL_NAME,
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "record_count": len(records),
        "normalized": True,
        "records": metadata_records,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# VALIDATE OUTPUT
# ============================================================

def validate_embeddings(
    embeddings: np.ndarray,
    records: list[dict[str, Any]],
) -> None:
    """Validate embedding shape and numerical values."""

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D matrix."
        )

    record_count, dimension = (
        embeddings.shape
    )

    if record_count != len(records):
        raise ValueError(
            f"Embedding count mismatch: "
            f"{record_count} vectors for "
            f"{len(records)} records."
        )

    if dimension != 384:
        raise ValueError(
            f"Expected 384-dimensional "
            f"embeddings, got {dimension}."
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise ValueError(
            "Embeddings contain NaN or "
            "infinite values."
        )

    # Because normalize_embeddings=True,
    # vectors should have approximately unit length.
    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Embeddings are not normalized "
            "to approximately unit length."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PHASE 3B - LOCAL DOCUMENT EMBEDDING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    records = load_records(
        INPUT_PATH
    )

    print(
        f"Input records: {len(records)}"
    )

    if not records:
        raise ValueError(
            "No records found in input dataset."
        )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        f"\nLoading model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = generate_embeddings(
        model=model,
        records=records,
    )

    print(
        f"\nEmbedding matrix shape: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_embeddings(
        embeddings=embeddings,
        records=records,
    )

    print(
        "Embedding validation: PASSED"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    save_metadata(
        records=records,
        embeddings=embeddings,
        path=METADATA_PATH,
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    file_size_mb = (
        EMBEDDINGS_PATH.stat().st_size
        / (1024 * 1024)
    )

    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Records              : "
        f"{len(records)}"
    )

    print(
        f"Vector dimension     : "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Embedding matrix     : "
        f"{embeddings.shape[0]} x "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Normalized vectors   : Yes"
    )

    print(
        f"Embeddings file      : "
        f"{EMBEDDINGS_PATH}"
    )

    print(
        f"Metadata file        : "
        f"{METADATA_PATH}"
    )

    print(
        f"Embedding file size  : "
        f"{file_size_mb:.2f} MB"
    )


if __name__ == "__main__":
    main()