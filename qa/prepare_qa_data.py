from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "qa_dataset"
    / "processed"
)

SPLITS = {
    "train": DATASET_DIR / "train.jsonl",
    "validation": DATASET_DIR / "validation.jsonl",
    "test": DATASET_DIR / "test.jsonl",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
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
                    f"Invalid JSON at {path}, "
                    f"line {line_number}"
                ) from exc

            records.append(record)

    return records


def validate_record(
    record: dict[str, Any],
    split_name: str,
) -> None:
    """Validate one extractive QA example."""

    required_fields = {
        "id",
        "document_id",
        "context",
        "question",
        "answers",
    }

    missing = required_fields - record.keys()

    if missing:
        raise ValueError(
            f"{split_name}: {record.get('id')} "
            f"missing fields: {sorted(missing)}"
        )

    context = record["context"]
    question = record["question"]
    answers = record["answers"]

    if not isinstance(context, str) or not context.strip():
        raise ValueError(
            f"{split_name}: {record['id']} "
            f"has empty context"
        )

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"{split_name}: {record['id']} "
            f"has empty question"
        )

    answer_texts = answers.get("text", [])
    answer_starts = answers.get("answer_start", [])

    if len(answer_texts) != 1:
        raise ValueError(
            f"{split_name}: {record['id']} "
            f"must contain exactly one answer text"
        )

    if len(answer_starts) != 1:
        raise ValueError(
            f"{split_name}: {record['id']} "
            f"must contain exactly one answer_start"
        )

    answer_text = answer_texts[0]
    answer_start = answer_starts[0]

    if not isinstance(answer_text, str):
        raise ValueError(
            f"{split_name}: {record['id']} "
            f"answer text is not a string"
        )

    if not isinstance(answer_start, int):
        raise ValueError(
            f"{split_name}: {record['id']} "
            f"answer_start is not an integer"
        )

    if answer_start < 0:
        raise ValueError(
            f"{split_name}: {record['id']} "
            f"negative answer_start"
        )

    answer_end = (
        answer_start
        + len(answer_text)
    )

    if answer_end > len(context):
        raise ValueError(
            f"{split_name}: {record['id']} "
            f"answer extends beyond context"
        )

    actual_span = context[
        answer_start:answer_end
    ]

    if actual_span != answer_text:
        raise ValueError(
            f"{split_name}: {record['id']} "
            f"answer span mismatch\n"
            f"Expected: {answer_text!r}\n"
            f"Found:    {actual_span!r}"
        )


def main() -> None:
    print("=" * 70)
    print("PHASE 5B-1 - QA DATA PREPARATION")
    print("=" * 70)

    total = 0

    for split_name, path in SPLITS.items():

        records = load_jsonl(path)

        print(
            f"\n{split_name.upper()}: "
            f"{len(records)} examples"
        )

        seen_ids: set[str] = set()

        for record in records:

            record_id = record["id"]

            if record_id in seen_ids:
                raise ValueError(
                    f"Duplicate ID in "
                    f"{split_name}: {record_id}"
                )

            seen_ids.add(record_id)

            validate_record(
                record,
                split_name,
            )

        total += len(records)

        print(
            f"  Answer spans: VALID"
        )

    print("\n" + "=" * 70)
    print(
        f"TOTAL EXAMPLES: {total}"
    )
    print("ALL CHARACTER-LEVEL ANSWER SPANS ARE VALID")
    print("=" * 70)


if __name__ == "__main__":
    main()