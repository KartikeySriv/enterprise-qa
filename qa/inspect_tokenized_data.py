from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZED_DIR = (
    PROJECT_ROOT
    / "data"
    / "qa_dataset"
    / "processed"
    / "tokenized"
)

MODEL_NAME = "deepset/tinyroberta-squad2"


def inspect_split(
    split_name: str,
    tokenizer,
) -> None:

    path = TOKENIZED_DIR / split_name

    dataset = load_from_disk(
        str(path)
    )

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} TOKENIZED DATA")
    print("=" * 70)

    print(
        f"Features: {len(dataset)}"
    )

    print(
        "Columns:"
    )

    for column in dataset.column_names:
        print(
            f"  - {column}"
        )

    answer_count = sum(
        dataset["has_answer"]
    )

    print(
        f"\nFeatures containing answer: "
        f"{answer_count}"
    )

    print(
        f"Features without answer: "
        f"{len(dataset) - answer_count}"
    )

    # Show first feature with an answer.
    example_index = None

    for index, has_answer in enumerate(
        dataset["has_answer"]
    ):
        if has_answer:
            example_index = index
            break

    if example_index is None:
        print(
            "\nWARNING: no answer-containing "
            "feature found."
        )
        return

    feature = dataset[
        example_index
    ]

    input_ids = feature[
        "input_ids"
    ]

    start = feature[
        "start_positions"
    ]

    end = feature[
        "end_positions"
    ]

    example_id = feature[
        "example_id"
    ]

    predicted_answer = tokenizer.decode(
        input_ids[start:end + 1],
        skip_special_tokens=True,
    )

    print(
        "\nExample feature:"
    )

    print(
        f"  Original example ID : "
        f"{example_id}"
    )

    print(
        f"  Start token         : "
        f"{start}"
    )

    print(
        f"  End token           : "
        f"{end}"
    )

    print(
        f"  Decoded answer      : "
        f"{predicted_answer!r}"
    )

    print(
        "\nThis should correspond "
        "to the answer recorded in "
        "the original JSONL example."
    )


def main() -> None:

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        inspect_split(
            split_name,
            tokenizer,
        )


if __name__ == "__main__":
    main()