from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "qa_dataset"
    / "processed"
)

TOKENIZED_DIR = (
    DATASET_DIR
    / "tokenized"
)

MODEL_NAME = "deepset/tinyroberta-squad2"

MAX_LENGTH = 512
DOC_STRIDE = 128


SPLITS = {
    "train": DATASET_DIR / "train.jsonl",
    "validation": DATASET_DIR / "validation.jsonl",
    "test": DATASET_DIR / "test.jsonl",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL into a list of records."""

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def tokenize_split(
    records: list[dict[str, Any]],
    tokenizer,
    split_name: str,
) -> Dataset:
    """
    Tokenize one split and map character-level answer spans
    to exact token-level start/end positions.

    Long contexts are handled using overlapping windows.

    The mapping uses token offset overlap with the exact
    character interval [answer_start, answer_end), preventing
    punctuation immediately after the answer from being included.
    """

    dataset = Dataset.from_list(records)

    def prepare_features(
        examples: dict[str, list[Any]],
    ) -> dict[str, list[Any]]:

        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop(
            "overflow_to_sample_mapping"
        )

        offset_mapping = tokenized["offset_mapping"]

        start_positions: list[int] = []
        end_positions: list[int] = []

        feature_example_ids: list[str] = []
        has_answer: list[bool] = []

        for feature_index, offsets in enumerate(
            offset_mapping
        ):

            sample_index = sample_mapping[
                feature_index
            ]

            record_id = examples["id"][
                sample_index
            ]

            feature_example_ids.append(record_id)

            answer_text = examples["answers"][
                sample_index
            ]["text"][0]

            answer_start = examples["answers"][
                sample_index
            ]["answer_start"][0]

            answer_end = (
                answer_start
                + len(answer_text)
            )

            sequence_ids = tokenized.sequence_ids(
                feature_index
            )

            # ------------------------------------------------
            # Locate the context token range.
            # sequence_id == 0 → question
            # sequence_id == 1 → context
            # None → special tokens
            # ------------------------------------------------

            context_token_start = None
            context_token_end = None

            for token_index, sequence_id in enumerate(
                sequence_ids
            ):
                if sequence_id == 1:
                    if context_token_start is None:
                        context_token_start = token_index

                    context_token_end = token_index

            if (
                context_token_start is None
                or context_token_end is None
            ):
                # No context tokens in this feature.
                start_positions.append(0)
                end_positions.append(0)
                has_answer.append(False)
                continue

            # ------------------------------------------------
            # Check whether this window contains the answer.
            #
            # We require the entire answer interval to be
            # covered by this context window.
            # ------------------------------------------------

            window_start_char = offsets[
                context_token_start
            ][0]

            window_end_char = offsets[
                context_token_end
            ][1]

            if (
                answer_start < window_start_char
                or answer_end > window_end_char
            ):
                # Answer is not fully contained in
                # this overflow window.
                start_positions.append(0)
                end_positions.append(0)
                has_answer.append(False)
                continue

            # ------------------------------------------------
            # Find the token whose span contains the first
            # answer character.
            #
            # Condition:
            # token_start <= answer_start < token_end
            # ------------------------------------------------

            token_start = None

            for token_index in range(
                context_token_start,
                context_token_end + 1,
            ):
                token_char_start, token_char_end = (
                    offsets[token_index]
                )

                if (
                    token_char_start
                    <= answer_start
                    < token_char_end
                ):
                    token_start = token_index
                    break

            # ------------------------------------------------
            # Find the token containing the final answer
            # character.
            #
            # Condition:
            # token_start < answer_end <= token_end
            # ------------------------------------------------

            token_end = None

            last_answer_character = answer_end - 1

            for token_index in range(
                context_token_start,
                context_token_end + 1,
            ):
                token_char_start, token_char_end = (
                    offsets[token_index]
                )

                if (
                    token_char_start
                    <= last_answer_character
                    < token_char_end
                ):
                    token_end = token_index
                    break

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if (
                token_start is None
                or token_end is None
            ):
                start_positions.append(0)
                end_positions.append(0)
                has_answer.append(False)
                continue

            # The answer must never produce an inverted span.
            if token_end < token_start:
                start_positions.append(0)
                end_positions.append(0)
                has_answer.append(False)
                continue

            start_positions.append(token_start)
            end_positions.append(token_end)
            has_answer.append(True)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        tokenized["example_id"] = feature_example_ids
        tokenized["has_answer"] = has_answer

        return tokenized

    tokenized_dataset = dataset.map(
        prepare_features,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split_name}",
    )

    return tokenized_dataset


def main() -> None:
    print("=" * 70)
    print("PHASE 5B-2 - QA TOKENIZATION")
    print("=" * 70)

    TOKENIZED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading tokenizer: {MODEL_NAME}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required "
            "for offset_mapping."
        )

    print(
        f"Tokenizer maximum length: "
        f"{tokenizer.model_max_length}"
    )

    print(
        f"Using MAX_LENGTH: {MAX_LENGTH}"
    )

    print(
        f"Using DOC_STRIDE: {DOC_STRIDE}"
    )

    for split_name, path in SPLITS.items():

        print(
            f"\nProcessing {split_name}..."
        )

        records = load_jsonl(path)

        tokenized_dataset = tokenize_split(
            records=records,
            tokenizer=tokenizer,
            split_name=split_name,
        )

        output_path = (
            TOKENIZED_DIR
            / split_name
        )

        tokenized_dataset.save_to_disk(
            str(output_path)
        )

        answer_windows = sum(
            tokenized_dataset[
                "has_answer"
            ]
        )

        total_windows = len(
            tokenized_dataset
        )

        print(
            f"Original examples : "
            f"{len(records)}"
        )

        print(
            f"Tokenized windows  : "
            f"{total_windows}"
        )

        print(
            f"Windows containing "
            f"answer             : "
            f"{answer_windows}"
        )

        print(
            f"Windows without "
            f"answer             : "
            f"{total_windows - answer_windows}"
        )

        print(
            f"Saved to           : "
            f"{output_path}"
        )

    print("\n" + "=" * 70)
    print("TOKENIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()