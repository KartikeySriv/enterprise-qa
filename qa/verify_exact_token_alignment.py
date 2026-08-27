from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "qa_dataset"
    / "processed"
)

TOKENIZED_DIR = DATASET_DIR / "tokenized"

SPLITS = {
    "train": DATASET_DIR / "train.jsonl",
    "validation": DATASET_DIR / "validation.jsonl",
    "test": DATASET_DIR / "test.jsonl",
}


def load_jsonl(path: Path) -> dict[str, dict]:
    """
    Load original QA records and index them by ID.
    """
    records = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            records[record["id"]] = record

    return records


def verify_split(split_name: str) -> tuple[int, int]:
    """
    Verify that token start/end positions map back exactly
    to the original character-level answer span.
    """

    original_records = load_jsonl(
        SPLITS[split_name]
    )

    tokenized_path = TOKENIZED_DIR / split_name

    dataset = load_from_disk(
        str(tokenized_path)
    )

    checked = 0
    failures = 0

    print("\n" + "=" * 70)
    print(f"EXACT ALIGNMENT CHECK - {split_name.upper()}")
    print("=" * 70)

    for index in range(len(dataset)):

        feature = dataset[index]

        # Only check features for which this window
        # contains the answer.
        if not feature["has_answer"]:
            continue

        example_id = feature["example_id"]

        if example_id not in original_records:
            failures += 1
            print(
                f"FAIL: {example_id} "
                f"does not exist in original dataset."
            )
            continue

        original = original_records[
            example_id
        ]

        context = original["context"]

        expected_answer = (
            original["answers"]["text"][0]
        )

        expected_start = (
            original["answers"]["answer_start"][0]
        )

        expected_end = (
            expected_start
            + len(expected_answer)
        )

        start_token = feature[
            "start_positions"
        ]

        end_token = feature[
            "end_positions"
        ]

        offsets = feature[
            "offset_mapping"
        ]

        # Token offset for start and end.
        start_char = offsets[
            start_token
        ][0]

        end_char = offsets[
            end_token
        ][1]

        reconstructed_answer = context[
            start_char:end_char
        ]

        checked += 1

        # ----------------------------------------------------
        # Exact validation
        # ----------------------------------------------------

        if (
            reconstructed_answer
            != expected_answer
            or start_char != expected_start
            or end_char != expected_end
        ):
            failures += 1

            print(
                f"\nFAIL: {example_id}"
            )

            print(
                f"  Expected answer : "
                f"{expected_answer!r}"
            )

            print(
                f"  Reconstructed   : "
                f"{reconstructed_answer!r}"
            )

            print(
                f"  Expected chars  : "
                f"{expected_start} - {expected_end}"
            )

            print(
                f"  Token chars     : "
                f"{start_char} - {end_char}"
            )

            print(
                f"  Start token     : "
                f"{start_token}"
            )

            print(
                f"  End token       : "
                f"{end_token}"
            )

    print(
        f"\nChecked: {checked}"
    )

    print(
        f"Failures: {failures}"
    )

    return checked, failures


def main() -> None:

    total_checked = 0
    total_failures = 0

    for split_name in (
        "train",
        "validation",
        "test",
    ):

        checked, failures = verify_split(
            split_name
        )

        total_checked += checked
        total_failures += failures

    print("\n" + "=" * 70)
    print("EXACT CHARACTER ↔ TOKEN ALIGNMENT")
    print("=" * 70)

    print(
        f"Total checked : {total_checked}"
    )

    print(
        f"Total failures: {total_failures}"
    )

    if total_failures == 0:
        print(
            "\nEXACT ALIGNMENT PASSED"
        )
    else:
        print(
            "\nEXACT ALIGNMENT FAILED"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()