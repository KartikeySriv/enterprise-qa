from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "qa_dataset"
    / "processed"
)

INPUT_PATH = DATASET_DIR / "all_examples.jsonl"

BACKUP_PATH = (
    DATASET_DIR
    / "all_examples.before_correction.jsonl"
)

REPORT_PATH = (
    DATASET_DIR
    / "correction_report.json"
)


# ============================================================
# EXACT CORRECTIONS
# ============================================================

CORRECTIONS: dict[str, dict[str, Any]] = {

    "02_home_loan_product_manual__qa_008": {
        "old_answer": "0.85%",
        "new_answer": "0.85%,",
        "answer_start": 179,
        "reason": (
            "The answer ends immediately before a comma "
            "that is part of the tokenizer-aligned span."
        ),
    },

    "09_financial_compliance_regulation__qa_004": {
        "old_answer": "100%",
        "new_answer": "100%,",
        "answer_start": 75,
        "reason": (
            "The answer ends immediately before a comma "
            "that is part of the tokenizer-aligned span."
        ),
    },

    "05_customer_onboarding_policy__qa_002": {
        "old_answer": (
            "The unique tax identifier required for all customers "
            "opening an account with Aurelia Bank, used for "
            "regulatory reporting under the Financial Compliance "
            "Regulation (AUR-FC-009)"
        ),
        "new_answer": (
            "The unique tax identifier required for all customers "
            "opening an account with Aurelia Bank, used for "
            "regulatory reporting under the Financial Compliance "
            "Regulation (AUR-FC-009)."
        ),
        "answer_start": 43,
        "reason": (
            "The answer ends immediately before the sentence-final "
            "period that is part of the tokenizer-aligned span."
        ),
    },

    "08_credit_risk_eligibility_policy__qa_005": {
        "old_answer": "35%",
        "new_answer": "35%;",
        "answer_start": 101,
        "reason": (
            "The answer ends immediately before a semicolon "
            "that is part of the tokenizer-aligned span."
        ),
    },
}


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""

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
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def validate_original_record(
    record: dict[str, Any],
    correction: dict[str, Any],
) -> None:
    """
    Verify that the record is exactly in the state we expect
    BEFORE applying the correction.

    This prevents accidentally changing the wrong record.
    """

    record_id = record.get("id")

    expected_old_answer = correction["old_answer"]
    expected_start = correction["answer_start"]

    actual_answer = (
        record["answers"]["text"][0]
    )

    actual_start = (
        record["answers"]["answer_start"][0]
    )

    if actual_answer != expected_old_answer:
        raise ValueError(
            f"{record_id}: unexpected original answer.\n"
            f"Expected: {expected_old_answer!r}\n"
            f"Found:    {actual_answer!r}"
        )

    if actual_start != expected_start:
        raise ValueError(
            f"{record_id}: unexpected original answer_start.\n"
            f"Expected: {expected_start}\n"
            f"Found:    {actual_start}"
        )


def validate_corrected_record(
    record: dict[str, Any],
    correction: dict[str, Any],
) -> None:
    """
    Verify the corrected answer is an exact substring
    of the context at the expected character position.
    """

    record_id = record["id"]

    context = record["context"]

    answer = (
        record["answers"]["text"][0]
    )

    answer_start = (
        record["answers"]["answer_start"][0]
    )

    expected_answer = correction[
        "new_answer"
    ]

    expected_start = correction[
        "answer_start"
    ]

    if answer != expected_answer:
        raise ValueError(
            f"{record_id}: corrected answer mismatch."
        )

    if answer_start != expected_start:
        raise ValueError(
            f"{record_id}: corrected answer_start mismatch."
        )

    answer_end = (
        answer_start
        + len(answer)
    )

    if answer_end > len(context):
        raise ValueError(
            f"{record_id}: corrected answer "
            "extends beyond context."
        )

    actual_span = context[
        answer_start:answer_end
    ]

    if actual_span != answer:
        raise ValueError(
            f"{record_id}: corrected answer "
            "does not match context.\n"
            f"Expected: {answer!r}\n"
            f"Found:    {actual_span!r}"
        )


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write records as UTF-8 JSONL."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("ANSWER SPAN CORRECTION")
    print("=" * 70)

    records = load_jsonl(INPUT_PATH)

    print(
        f"Loaded records: {len(records)}"
    )

    record_by_id = {
        record["id"]: record
        for record in records
    }

    # --------------------------------------------------------
    # Confirm all four records exist.
    # --------------------------------------------------------

    missing_ids = [
        record_id
        for record_id in CORRECTIONS
        if record_id not in record_by_id
    ]

    if missing_ids:
        raise ValueError(
            "Expected correction IDs not found:\n"
            + "\n".join(missing_ids)
        )

    # --------------------------------------------------------
    # Validate the original records BEFORE modifying anything.
    # --------------------------------------------------------

    for record_id, correction in CORRECTIONS.items():

        validate_original_record(
            record=record_by_id[record_id],
            correction=correction,
        )

    print(
        "\nOriginal records validated: "
        f"{len(CORRECTIONS)}"
    )

    # --------------------------------------------------------
    # Create a backup.
    # --------------------------------------------------------

    if BACKUP_PATH.exists():
        print(
            "\nBackup already exists:"
            f"\n{BACKUP_PATH}"
        )
    else:
        shutil.copy2(
            INPUT_PATH,
            BACKUP_PATH,
        )

        print(
            "\nCreated backup:"
            f"\n{BACKUP_PATH}"
        )

    # --------------------------------------------------------
    # Apply corrections.
    # --------------------------------------------------------

    correction_results: list[dict[str, Any]] = []

    for record_id, correction in CORRECTIONS.items():

        record = record_by_id[record_id]

        old_answer = (
            record["answers"]["text"][0]
        )

        new_answer = correction[
            "new_answer"
        ]

        # Keep answer_start unchanged.
        answer_start = correction[
            "answer_start"
        ]

        record["answers"]["text"][0] = (
            new_answer
        )

        record["answers"][
            "answer_start"
        ][0] = answer_start

        # Verify immediately after correction.
        validate_corrected_record(
            record=record,
            correction=correction,
        )

        correction_results.append(
            {
                "id": record_id,
                "old_answer": old_answer,
                "new_answer": new_answer,
                "answer_start": answer_start,
                "reason": correction["reason"],
            }
        )

        print(
            f"\nCORRECTED: {record_id}"
        )

        print(
            f"  Old: {old_answer!r}"
        )

        print(
            f"  New: {new_answer!r}"
        )

    # --------------------------------------------------------
    # Write corrected dataset.
    # --------------------------------------------------------

    write_jsonl(
        path=INPUT_PATH,
        records=records,
    )

    # --------------------------------------------------------
    # Final validation of the entire dataset.
    # --------------------------------------------------------

    corrected_records = load_jsonl(
        INPUT_PATH
    )

    if len(corrected_records) != len(records):
        raise RuntimeError(
            "Record count changed unexpectedly."
        )

    # Validate all four corrected records again.
    for record_id, correction in CORRECTIONS.items():

        validate_corrected_record(
            record=next(
                record
                for record in corrected_records
                if record["id"] == record_id
            ),
            correction=correction,
        )

    # --------------------------------------------------------
    # Write report.
    # --------------------------------------------------------

    report = {
        "input_file": str(INPUT_PATH),
        "backup_file": str(BACKUP_PATH),
        "total_records": len(corrected_records),
        "corrections_applied": len(CORRECTIONS),
        "corrections": correction_results,
    }

    with REPORT_PATH.open(
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
    # Final output.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CORRECTION COMPLETE")
    print("=" * 70)

    print(
        f"Total records      : "
        f"{len(corrected_records)}"
    )

    print(
        f"Corrections applied: "
        f"{len(CORRECTIONS)}"
    )

    print(
        "\nAll corrected answer spans "
        "validated successfully."
    )

    print(
        f"\nBackup:"
        f"\n{BACKUP_PATH}"
    )

    print(
        f"\nCorrected dataset:"
        f"\n{INPUT_PATH}"
    )

    print(
        f"\nCorrection report:"
        f"\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()