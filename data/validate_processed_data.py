from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

REQUIRED_RECORD_FIELDS = {
    "record_id",
    "document_id",
    "document_title",
    "organization",
    "doctype",
    "source_file",
    "metadata",
    "heading_level",
    "section",
    "subsection",
    "subsubsection",
    "title_path",
    "text",
}


# ============================================================
# VALIDATION RESULT
# ============================================================

class ValidationReport:
    """Collect validation errors and warnings."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


# ============================================================
# FILE HELPERS
# ============================================================

def load_json_file(
    path: Path,
    report: ValidationReport,
) -> Any | None:
    """Load a JSON file and report parsing errors."""

    if not path.exists():
        report.error(f"Missing file: {path}")
        return None

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        report.error(
            f"Invalid JSON in {path}: {exc}"
        )
        return None

    except OSError as exc:
        report.error(
            f"Could not read {path}: {exc}"
        )
        return None


# ============================================================
# RAW DOCUMENT VALIDATION
# ============================================================

def find_raw_documents(
    raw_dir: Path,
    report: ValidationReport,
) -> list[Path]:
    """Find all source AsciiDoc documents."""

    if not raw_dir.exists():
        report.error(
            f"Missing raw directory: {raw_dir}"
        )
        return []

    files = sorted(raw_dir.glob("*.adoc"))

    if not files:
        report.error(
            f"No .adoc files found in {raw_dir}"
        )

    return files


# ============================================================
# JSONL VALIDATION
# ============================================================

def load_and_validate_jsonl(
    jsonl_path: Path,
    raw_files: list[Path],
    report: ValidationReport,
) -> list[dict]:
    """
    Load every JSONL record and validate its basic structure.
    """

    if not jsonl_path.exists():
        report.error(
            f"Missing processed JSONL file: {jsonl_path}"
        )
        return []

    records: list[dict] = []
    record_ids: set[str] = set()
    source_files = {
        path.name
        for path in raw_files
    }

    try:
        with jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, raw_line in enumerate(
                file,
                start=1,
            ):

                line = raw_line.strip()

                if not line:
                    report.warning(
                        f"Blank JSONL line at line {line_number}"
                    )
                    continue

                # ----------------------------------------
                # JSON parsing
                # ----------------------------------------

                try:
                    record = json.loads(line)

                except json.JSONDecodeError as exc:
                    report.error(
                        f"Invalid JSON at JSONL line "
                        f"{line_number}: {exc}"
                    )
                    continue

                # ----------------------------------------
                # Top-level object
                # ----------------------------------------

                if not isinstance(record, dict):
                    report.error(
                        f"JSONL line {line_number} "
                        f"is not a JSON object"
                    )
                    continue

                records.append(record)

                # ----------------------------------------
                # Required fields
                # ----------------------------------------

                missing_fields = (
                    REQUIRED_RECORD_FIELDS
                    - record.keys()
                )

                if missing_fields:
                    report.error(
                        f"Record at JSONL line "
                        f"{line_number} is missing fields: "
                        f"{sorted(missing_fields)}"
                    )

                # ----------------------------------------
                # Record ID
                # ----------------------------------------

                record_id = record.get("record_id")

                if not isinstance(record_id, str):
                    report.error(
                        f"Record at line {line_number} "
                        f"has invalid record_id"
                    )

                elif not record_id.strip():
                    report.error(
                        f"Record at line {line_number} "
                        f"has empty record_id"
                    )

                elif record_id in record_ids:
                    report.error(
                        f"Duplicate record_id: "
                        f"{record_id}"
                    )

                else:
                    record_ids.add(record_id)

                # ----------------------------------------
                # Document ID
                # ----------------------------------------

                document_id = record.get(
                    "document_id"
                )

                if not isinstance(
                    document_id,
                    str,
                ) or not document_id.strip():
                    report.error(
                        f"Record at line {line_number} "
                        f"has invalid document_id"
                    )

                # ----------------------------------------
                # Source file
                # ----------------------------------------

                source_file = record.get(
                    "source_file"
                )

                if source_file not in source_files:
                    report.error(
                        f"Record at line {line_number} "
                        f"references missing raw source: "
                        f"{source_file}"
                    )

                # ----------------------------------------
                # Text
                # ----------------------------------------

                text = record.get("text")

                if not isinstance(text, str):
                    report.error(
                        f"Record at line {line_number} "
                        f"has non-string text"
                    )

                elif not text.strip():
                    report.error(
                        f"Record at line {line_number} "
                        f"has empty text"
                    )

                # ----------------------------------------
                # Heading level
                # ----------------------------------------

                heading_level = record.get(
                    "heading_level"
                )

                if not isinstance(
                    heading_level,
                    int,
                ) or heading_level < 2:
                    report.error(
                        f"Record at line {line_number} "
                        f"has invalid heading_level: "
                        f"{heading_level}"
                    )

                # ----------------------------------------
                # Title path
                # ----------------------------------------

                title_path = record.get(
                    "title_path"
                )

                if not isinstance(
                    title_path,
                    list,
                ):
                    report.error(
                        f"Record at line {line_number} "
                        f"has invalid title_path"
                    )

                elif not title_path:
                    report.error(
                        f"Record at line {line_number} "
                        f"has empty title_path"
                    )

                # ----------------------------------------
                # Hierarchy consistency
                # ----------------------------------------

                section = record.get("section")
                subsection = record.get(
                    "subsection"
                )
                subsubsection = record.get(
                    "subsubsection"
                )

                if heading_level == 2:
                    if section is None:
                        report.error(
                            f"Level-2 record at line "
                            f"{line_number} has no section"
                        )

                if heading_level == 3:
                    if (
                        section is None
                        or subsection is None
                    ):
                        report.error(
                            f"Level-3 record at line "
                            f"{line_number} has incomplete "
                            f"hierarchy"
                        )

                if heading_level == 4:
                    if (
                        section is None
                        or subsection is None
                        or subsubsection is None
                    ):
                        report.error(
                            f"Level-4 record at line "
                            f"{line_number} has incomplete "
                            f"hierarchy"
                        )

                # ----------------------------------------
                # Metadata
                # ----------------------------------------

                metadata = record.get(
                    "metadata"
                )

                if not isinstance(
                    metadata,
                    dict,
                ):
                    report.error(
                        f"Record at line {line_number} "
                        f"has invalid metadata"
                    )

                # ----------------------------------------
                # Raw AsciiDoc heading leakage
                # ----------------------------------------

                if isinstance(text, str):

                    text_lines = text.splitlines()

                    for text_line in text_lines:

                        stripped = text_line.strip()

                        if (
                            stripped.startswith("== ")
                            or stripped.startswith("=== ")
                            or stripped.startswith("==== ")
                        ):
                            report.error(
                                f"AsciiDoc heading leaked into "
                                f"record text at line "
                                f"{line_number}: "
                                f"{stripped}"
                            )

                            break

    except OSError as exc:
        report.error(
            f"Could not read JSONL file: {exc}"
        )

    return records


# ============================================================
# CORPUS METADATA VALIDATION
# ============================================================

def validate_corpus_metadata(
    corpus: Any,
    records: list[dict],
    raw_files: list[Path],
    report: ValidationReport,
) -> None:
    """Validate corpus.json against actual processed data."""

    if not isinstance(corpus, dict):
        report.error(
            "corpus.json root must be a JSON object"
        )
        return

    # --------------------------------------------
    # Required corpus fields
    # --------------------------------------------

    required_fields = {
        "corpus_name",
        "synthetic",
        "source_format",
        "document_count",
        "documents_successfully_processed",
        "record_count",
        "records_removed",
        "documents",
    }

    missing = required_fields - corpus.keys()

    if missing:
        report.error(
            "corpus.json missing fields: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------
    # Document count
    # --------------------------------------------

    actual_document_count = len(raw_files)

    if corpus.get(
        "document_count"
    ) != actual_document_count:

        report.error(
            "Document count mismatch: "
            f"corpus.json says "
            f"{corpus.get('document_count')}, "
            f"but raw directory contains "
            f"{actual_document_count}"
        )

    # --------------------------------------------
    # Successfully processed count
    # --------------------------------------------

    documents_processed = corpus.get(
        "documents_successfully_processed"
    )

    if documents_processed != actual_document_count:
        report.error(
            "Processed document count mismatch: "
            f"{documents_processed} vs "
            f"{actual_document_count}"
        )

    # --------------------------------------------
    # Record count
    # --------------------------------------------

    actual_record_count = len(records)

    if corpus.get(
        "record_count"
    ) != actual_record_count:

        report.error(
            "Record count mismatch: "
            f"corpus.json says "
            f"{corpus.get('record_count')}, "
            f"actual JSONL contains "
            f"{actual_record_count}"
        )

    # --------------------------------------------
    # Processed documents list
    # --------------------------------------------

    documents = corpus.get(
        "documents"
    )

    if not isinstance(
        documents,
        list,
    ):
        report.error(
            "corpus.json 'documents' must be a list"
        )
        return

    if len(documents) != actual_document_count:
        report.error(
            "corpus.json document list count does not "
            "match raw document count"
        )

    # --------------------------------------------
    # Per-document record counts
    # --------------------------------------------

    actual_counts: dict[str, int] = {}

    for record in records:

        document_id = record.get(
            "document_id"
        )

        if document_id:
            actual_counts[document_id] = (
                actual_counts.get(
                    document_id,
                    0,
                )
                + 1
            )

    for document in documents:

        if not isinstance(
            document,
            dict,
        ):
            report.error(
                "Invalid document entry in corpus.json"
            )
            continue

        document_id = document.get(
            "document_id"
        )

        expected_count = document.get(
            "record_count"
        )

        actual_count = actual_counts.get(
            document_id,
            0,
        )

        if expected_count != actual_count:
            report.error(
                f"Record count mismatch for "
                f"{document_id}: "
                f"corpus.json={expected_count}, "
                f"JSONL={actual_count}"
            )


# ============================================================
# DOCUMENT COVERAGE VALIDATION
# ============================================================

def validate_document_coverage(
    records: list[dict],
    raw_files: list[Path],
    report: ValidationReport,
) -> None:
    """Check that every raw document produced records."""

    source_to_count: dict[str, int] = {}

    for record in records:

        source_file = record.get(
            "source_file"
        )

        if isinstance(
            source_file,
            str,
        ):
            source_to_count[source_file] = (
                source_to_count.get(
                    source_file,
                    0,
                )
                + 1
            )

    for raw_file in raw_files:

        if raw_file.name not in source_to_count:

            report.error(
                f"Raw document has no processed "
                f"records: {raw_file.name}"
            )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    records: list[dict],
    raw_files: list[Path],
    corpus: Any,
    report: ValidationReport,
) -> None:
    """Print a readable validation report."""

    print("=" * 80)
    print("PHASE 2G - DATA VALIDATION")
    print("=" * 80)

    print(
        f"Raw documents              : "
        f"{len(raw_files)}"
    )

    print(
        f"Processed records          : "
        f"{len(records)}"
    )

    if isinstance(
        corpus,
        dict,
    ):
        print(
            f"Corpus record count       : "
            f"{corpus.get('record_count')}"
        )

        print(
            f"Records removed            : "
            f"{corpus.get('records_removed')}"
        )

    print(
        f"Warnings                   : "
        f"{len(report.warnings)}"
    )

    print(
        f"Errors                     : "
        f"{len(report.errors)}"
    )

    print("=" * 80)

    if report.warnings:
        print("\nWARNINGS:")

        for warning in report.warnings:
            print(f"  - {warning}")

    if report.errors:
        print("\nERRORS:")

        for error in report.errors:
            print(f"  - {error}")

    if report.passed:
        print("\nVALIDATION PASSED")
    else:
        print(
            "\nVALIDATION FAILED"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Run all Phase 2 validation checks."""

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    raw_dir = (
        project_root
        / "data"
        / "raw"
    )

    processed_dir = (
        project_root
        / "data"
        / "processed"
    )

    jsonl_path = (
        processed_dir
        / "sections.jsonl"
    )

    corpus_path = (
        processed_dir
        / "corpus.json"
    )

    report = ValidationReport()

    # --------------------------------------------------------
    # Locate raw documents
    # --------------------------------------------------------

    raw_files = find_raw_documents(
        raw_dir=raw_dir,
        report=report,
    )

    # --------------------------------------------------------
    # Load corpus metadata
    # --------------------------------------------------------

    corpus = load_json_file(
        path=corpus_path,
        report=report,
    )

    # --------------------------------------------------------
    # Validate JSONL
    # --------------------------------------------------------

    records = load_and_validate_jsonl(
        jsonl_path=jsonl_path,
        raw_files=raw_files,
        report=report,
    )

    # --------------------------------------------------------
    # Validate coverage
    # --------------------------------------------------------

    validate_document_coverage(
        records=records,
        raw_files=raw_files,
        report=report,
    )

    # --------------------------------------------------------
    # Validate corpus metadata
    # --------------------------------------------------------

    validate_corpus_metadata(
        corpus=corpus,
        records=records,
        raw_files=raw_files,
        report=report,
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print_summary(
        records=records,
        raw_files=raw_files,
        corpus=corpus,
        report=report,
    )

    # Non-zero exit code when validation fails.
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())