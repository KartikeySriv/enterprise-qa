from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "qa_dataset"
    / "processed"
)

MODEL_NAME = "deepset/tinyroberta-squad2"


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def check_record(
    record: dict,
    tokenizer,
) -> tuple[bool, str]:

    question = record["question"]
    context = record["context"]

    answer_text = record[
        "answers"
    ]["text"][0]

    answer_start = record[
        "answers"
    ]["answer_start"][0]

    answer_end = (
        answer_start
        + len(answer_text)
    )

    tokenized = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=512,
        return_offsets_mapping=True,
    )

    offsets = tokenized[
        "offset_mapping"
    ]

    sequence_ids = tokenized.sequence_ids()

    context_token_indices = [
        i
        for i, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
    ]

    answer_token_indices = []

    for i in context_token_indices:

        token_start, token_end = offsets[i]

        # Token overlaps the answer span.
        if (
            token_start < answer_end
            and token_end > answer_start
        ):
            answer_token_indices.append(i)

    if not answer_token_indices:
        return False, "no_token_overlap"

    first_token = answer_token_indices[0]
    last_token = answer_token_indices[-1]

    token_start = offsets[first_token][0]
    token_end = offsets[last_token][1]

    exact = (
        token_start == answer_start
        and token_end == answer_end
    )

    if exact:
        return True, ""

    reconstructed = context[
        token_start:token_end
    ]

    return (
        False,
        (
            f"expected={answer_text!r}, "
            f"reconstructed={reconstructed!r}, "
            f"expected_chars="
            f"{answer_start}-{answer_end}, "
            f"token_chars="
            f"{token_start}-{token_end}"
        ),
    )


def main() -> None:

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    total = 0
    aligned = 0
    misaligned = 0

    print("=" * 70)
    print("TOKEN BOUNDARY ANALYSIS")
    print("=" * 70)

    for split_name in (
        "train",
        "validation",
        "test",
    ):

        path = (
            DATASET_DIR
            / f"{split_name}.jsonl"
        )

        records = load_jsonl(path)

        print(
            f"\n{split_name.upper()}"
        )

        for record in records:

            total += 1

            ok, reason = check_record(
                record,
                tokenizer,
            )

            if ok:
                aligned += 1
            else:
                misaligned += 1

                print(
                    f"\nMISALIGNED: "
                    f"{record['id']}"
                )

                print(
                    f"  Question: "
                    f"{record['question']}"
                )

                print(
                    f"  {reason}"
                )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Total examples : {total}"
    )

    print(
        f"Token-aligned  : {aligned}"
    )

    print(
        f"Misaligned     : {misaligned}"
    )

    if misaligned == 0:
        print(
            "\nALL ANSWERS ARE TOKEN-ALIGNED"
        )
    else:
        print(
            f"\n{misaligned} examples "
            f"need special handling."
        )


if __name__ == "__main__":
    main()