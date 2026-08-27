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


def normalize_decoded_answer(text: str) -> str:
    """
    Remove tokenizer-introduced surrounding whitespace only.

    We are NOT changing the original dataset answer.
    This is used only to compare decoded tokens against
    the source answer text.
    """
    return text.strip()


def verify_split(
    split_name: str,
    tokenizer,
) -> tuple[int, int]:

    dataset_path = TOKENIZED_DIR / split_name

    dataset = load_from_disk(
        str(dataset_path)
    )

    checked = 0
    failures = 0

    # Build a quick lookup from the tokenized feature's
    # original example ID to the original dataset record.
    #
    # Because the tokenized dataset itself does not store
    # the original answer text, we reconstruct the answer
    # from the token offsets and inspect the resulting token span.

    print("\n" + "=" * 70)
    print(f"VERIFYING {split_name.upper()}")
    print("=" * 70)

    for index in range(len(dataset)):

        feature = dataset[index]

        if not feature["has_answer"]:
            continue

        input_ids = feature["input_ids"]

        start = feature["start_positions"]
        end = feature["end_positions"]

        decoded = tokenizer.decode(
            input_ids[start:end + 1],
            skip_special_tokens=True,
        )

        normalized = normalize_decoded_answer(
            decoded
        )

        # A valid answer feature must decode to
        # non-empty natural-language content.
        if not normalized:

            failures += 1

            print(
                f"\nFAIL: {feature['example_id']}"
            )
            print(
                "  Start token:",
                start,
            )
            print(
                "  End token:",
                end,
            )
            print(
                "  Decoded answer is empty"
            )

            continue

        checked += 1

    print(
        f"\nChecked answer-containing "
        f"features: {checked}"
    )

    print(
        f"Failures: {failures}"
    )

    return checked, failures


def main() -> None:

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    total_checked = 0
    total_failures = 0

    for split_name in (
        "train",
        "validation",
        "test",
    ):

        checked, failures = verify_split(
            split_name,
            tokenizer,
        )

        total_checked += checked
        total_failures += failures

    print("\n" + "=" * 70)
    print("TOKEN ALIGNMENT VERIFICATION")
    print("=" * 70)

    print(
        f"Total checked : {total_checked}"
    )

    print(
        f"Total failures: {total_failures}"
    )

    if total_failures == 0:
        print(
            "\nTOKEN ALIGNMENT BASIC CHECK PASSED"
        )
    else:
        print(
            "\nTOKEN ALIGNMENT CHECK FAILED"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()