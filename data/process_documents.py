from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Optional
import json

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class DocumentMetadata:
    """Document-level metadata extracted from the AsciiDoc header."""

    document_id: str
    document_title: Optional[str]
    organization: Optional[str]
    doctype: Optional[str]
    source_file: str

    policy_id: Optional[str] = None
    version: Optional[str] = None
    classification: Optional[str] = None


@dataclass
class SectionRecord:
    """Represents one logical block of document text."""

    heading_level: int
    section: Optional[str] = None
    subsection: Optional[str] = None
    subsubsection: Optional[str] = None
    title_path: list[str] = field(default_factory=list)
    text: str = ""

    # Added in Phase 2D
    record_id: Optional[str] = None


@dataclass
class ParsedDocument:
    """Represents one parsed AsciiDoc document."""

    metadata: DocumentMetadata
    records: list[SectionRecord]
    removed_records: int = 0


# ============================================================
# REGEX
# ============================================================

HEADING_PATTERN = re.compile(r"^(=+)\s+(.+?)\s*$")

DOCUMENT_METADATA_PATTERN = re.compile(
    r"^(Policy ID|Version|Classification)\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)

ASCIIDOC_ATTRIBUTE_PATTERN = re.compile(
    r"^:([^:]+):\s*(.*?)\s*$"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def is_heading(line: str) -> bool:
    """Return True when line is an AsciiDoc heading."""
    return HEADING_PATTERN.match(line) is not None


def parse_heading(line: str) -> tuple[int, str]:
    """Parse an AsciiDoc heading into level and title."""
    match = HEADING_PATTERN.match(line)

    if not match:
        raise ValueError(f"Invalid heading line: {line!r}")

    level = len(match.group(1))
    title = match.group(2).strip()

    return level, title


def normalize_lines(lines: list[str]) -> list[str]:
    """Normalize line endings while preserving content."""
    normalized: list[str] = []

    for line in lines:
        line = line.replace("\r\n", "\n").replace("\r", "\n")
        normalized.append(line.rstrip())

    return normalized


# ============================================================
# PHASE 2C - TEXT CLEANING
# ============================================================

def clean_metadata_value(value: Optional[str]) -> Optional[str]:
    """
    Conservatively clean a metadata value.

    Removes trailing AsciiDoc continuation markers such as '+'
    and normalizes surrounding whitespace.
    """
    if value is None:
        return None

    value = value.strip()

    # Remove trailing AsciiDoc continuation marker.
    value = re.sub(r"\s*\+\s*$", "", value)

    # Normalize repeated spaces.
    value = re.sub(r"[ \t]+", " ", value)

    return value.strip()


def clean_text(text: str) -> str:
    """
    Conservatively clean document text.

    This does not paraphrase, summarize, lowercase, stem,
    lemmatize, or otherwise change semantic content.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.rstrip() for line in text.split("\n")]

    # Remove blank lines at beginning.
    while lines and not lines[0].strip():
        lines.pop(0)

    # Remove blank lines at end.
    while lines and not lines[-1].strip():
        lines.pop()

    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        # Collapse repeated spaces/tabs.
        stripped = re.sub(r"[ \t]+", " ", stripped)

        cleaned_lines.append(stripped)

    cleaned_text = "\n".join(cleaned_lines)

    # Collapse excessive blank lines.
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def build_text(lines: list[str]) -> str:
    """
    Build a cleaned text block while preserving paragraph boundaries.
    """
    if not lines:
        return ""

    raw_text = "\n".join(lines)

    return clean_text(raw_text)


# ============================================================
# PHASE 2D - USABILITY CHECK
# ============================================================

def is_usable_record(record: SectionRecord) -> tuple[bool, str]:
    """
    Determine whether a parsed record contains meaningful content.

    We intentionally use conservative rules.

    We DO NOT remove short records simply because they are short,
    because short definitions may be valuable for QA.

    Returns:
        (True, "") when usable
        (False, reason) when unusable
    """

    text = record.text.strip()

    # Case 1: completely empty.
    if not text:
        return False, "empty_text"

    # Case 2: no alphanumeric content at all.
    if not re.search(r"[A-Za-z0-9]", text):
        return False, "formatting_only"

    # Case 3: text consists almost entirely of AsciiDoc
    # formatting symbols and contains no meaningful words.
    #
    # This is intentionally conservative.
    alphanumeric_count = sum(char.isalnum() for char in text)
    total_non_space = sum(not char.isspace() for char in text)

    if total_non_space > 0:
        alphanumeric_ratio = alphanumeric_count / total_non_space

        if alphanumeric_ratio < 0.20:
            return False, "mostly_markup"

    return True, ""


# ============================================================
# PHASE 2D - STABLE RECORD IDS
# ============================================================

def generate_record_id(document_id: str, record_number: int) -> str:
    """
    Generate a deterministic record ID.

    The same unchanged source document will produce the same IDs
    when processed in the same order.
    """
    return f"{document_id}__record_{record_number:03d}"


# ============================================================
# HIERARCHY
# ============================================================

def hierarchy_from_path(
    title_stack: dict[int, str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Convert heading hierarchy into convenient fields."""
    section = title_stack.get(2)
    subsection = title_stack.get(3)
    subsubsection = title_stack.get(4)

    return section, subsection, subsubsection


def create_record(
    heading_level: int,
    title_stack: dict[int, str],
    body_lines: list[str],
) -> Optional[SectionRecord]:
    """Create a cleaned record when body text exists."""

    text = build_text(body_lines)

    if not text:
        return None

    title_path = [
        title_stack[level]
        for level in sorted(title_stack)
        if level <= heading_level
    ]

    section, subsection, subsubsection = hierarchy_from_path(
        title_stack
    )

    return SectionRecord(
        heading_level=heading_level,
        section=section,
        subsection=subsection,
        subsubsection=subsubsection,
        title_path=title_path,
        text=text,
    )


# ============================================================
# DOCUMENT METADATA EXTRACTION
# ============================================================

def extract_document_metadata(
    path: Path,
    lines: list[str],
) -> DocumentMetadata:
    """Extract document metadata from the AsciiDoc header."""

    document_title: Optional[str] = None
    organization: Optional[str] = None
    doctype: Optional[str] = None

    policy_id: Optional[str] = None
    version: Optional[str] = None
    classification: Optional[str] = None

    header_finished = False
    title_seen = False
    organization_seen = False

    for line in lines:

        heading_match = HEADING_PATTERN.match(line)

        if heading_match:
            level = len(heading_match.group(1))

            if level >= 2:
                header_finished = True
                break

            if level == 1:
                title = heading_match.group(2).strip()

                if not title_seen:
                    document_title = title
                    title_seen = True

                continue

        if header_finished:
            break

        stripped = line.strip()

        if not stripped:
            continue

        attribute_match = ASCIIDOC_ATTRIBUTE_PATTERN.match(stripped)

        if attribute_match:
            key = attribute_match.group(1).strip().lower()
            value = clean_metadata_value(
                attribute_match.group(2)
            )

            if key == "doctype":
                doctype = value

            continue

        metadata_match = DOCUMENT_METADATA_PATTERN.match(stripped)

        if metadata_match:
            key = metadata_match.group(1).strip().lower()
            value = clean_metadata_value(
                metadata_match.group(2)
            )

            if key == "policy id":
                policy_id = value

            elif key == "version":
                version = value

            elif key == "classification":
                classification = value

            continue

        if title_seen and not organization_seen:
            organization = clean_metadata_value(stripped)
            organization_seen = True

    return DocumentMetadata(
        document_id=path.stem,
        document_title=document_title,
        organization=organization,
        doctype=doctype,
        source_file=path.name,
        policy_id=policy_id,
        version=version,
        classification=classification,
    )


# ============================================================
# PHASE 2D - FILTER AND ID RECORDS
# ============================================================

def prepare_records(
    records: list[SectionRecord],
    document_id: str,
) -> tuple[list[SectionRecord], dict[str, int]]:
    """
    Remove unusable records and assign deterministic IDs.

    IDs are assigned AFTER filtering so that valid records have
    contiguous deterministic numbering.
    """

    usable_records: list[SectionRecord] = []

    removal_counts: dict[str, int] = {}

    for record in records:

        usable, reason = is_usable_record(record)

        if not usable:
            removal_counts[reason] = (
                removal_counts.get(reason, 0) + 1
            )
            continue

        usable_records.append(record)

    # Assign deterministic IDs after filtering.
    for index, record in enumerate(
        usable_records,
        start=1,
    ):
        record.record_id = generate_record_id(
            document_id=document_id,
            record_number=index,
        )

    return usable_records, removal_counts


# ============================================================
# MAIN PARSER
# ============================================================

def parse_asciidoc_file(path: Path) -> ParsedDocument:
    """Parse and prepare one AsciiDoc document."""

    if not path.exists():
        raise FileNotFoundError(
            f"AsciiDoc file not found: {path}"
        )

    if path.suffix.lower() != ".adoc":
        raise ValueError(
            f"Expected .adoc file, got: {path}"
        )

    raw_text = path.read_text(encoding="utf-8")

    lines = normalize_lines(
        raw_text.splitlines()
    )

    # Document metadata.
    metadata = extract_document_metadata(
        path=path,
        lines=lines,
    )

    # Heading hierarchy.
    title_stack: dict[int, str] = {}

    raw_records: list[SectionRecord] = []

    current_heading_level: Optional[int] = None
    current_body: list[str] = []

    def flush_current_record() -> None:
        nonlocal current_body

        if current_heading_level is None:
            current_body = []
            return

        record = create_record(
            heading_level=current_heading_level,
            title_stack=title_stack,
            body_lines=current_body,
        )

        if record is not None:
            raw_records.append(record)

        current_body = []

    for line in lines:

        if is_heading(line):
            level, title = parse_heading(line)

            # Document title.
            if level == 1:
                continue

            # Save current section before moving to next heading.
            flush_current_record()

            # Remove deeper hierarchy levels.
            levels_to_remove = [
                existing_level
                for existing_level in title_stack
                if existing_level >= level
            ]

            for existing_level in levels_to_remove:
                del title_stack[existing_level]

            title_stack[level] = title

            current_heading_level = level
            current_body = []

        else:

            # Ignore document header content before
            # the first level-2 section.
            if current_heading_level is not None:
                current_body.append(line)

    # Save final record.
    flush_current_record()

    # Phase 2D:
    # filter unusable records + generate IDs.
    prepared_records, removal_counts = prepare_records(
        records=raw_records,
        document_id=metadata.document_id,
    )

    total_removed = sum(
        removal_counts.values()
    )

    return ParsedDocument(
        metadata=metadata,
        records=prepared_records,
        removed_records=total_removed,
    )


# ============================================================
# DEBUG OUTPUT
# ============================================================

def print_parsed_document(
    document: ParsedDocument,
) -> None:
    """Print the parsed document for manual inspection."""

    metadata = document.metadata

    print("=" * 80)
    print(f"FILE           : {metadata.source_file}")
    print(f"DOCUMENT ID    : {metadata.document_id}")
    print(f"TITLE          : {metadata.document_title}")
    print(f"ORGANIZATION   : {metadata.organization}")
    print(f"DOCTYPE        : {metadata.doctype}")
    print(f"POLICY ID      : {metadata.policy_id}")
    print(f"VERSION        : {metadata.version}")
    print(f"CLASSIFICATION : {metadata.classification}")
    print(f"RECORDS        : {len(document.records)}")
    print(f"REMOVED        : {document.removed_records}")
    print("=" * 80)

    for index, record in enumerate(
        document.records,
        start=1,
    ):
        print(f"\n[{index}]")
        print(f"  Record ID     : {record.record_id}")
        print(f"  Heading level : {record.heading_level}")
        print(f"  Section       : {record.section}")
        print(f"  Subsection    : {record.subsection}")
        print(f"  Subsubsection : {record.subsubsection}")
        print(
            f"  Title path    : "
            f"{' > '.join(record.title_path)}"
        )

        print("  Text:")
        print(f"    {record.text}")

# ============================================================
# PHASE 2E - JSONL SERIALIZATION
# ============================================================

def record_to_dict(
    record: SectionRecord,
    metadata: DocumentMetadata,
) -> dict:
    """
    Convert one SectionRecord + document metadata into the
    structured JSON representation used by the processed corpus.
    """

    return {
        "record_id": record.record_id,
        "document_id": metadata.document_id,
        "document_title": metadata.document_title,
        "organization": metadata.organization,
        "doctype": metadata.doctype,
        "source_file": metadata.source_file,
        "metadata": {
            "policy_id": metadata.policy_id,
            "version": metadata.version,
            "classification": metadata.classification,
        },
        "heading_level": record.heading_level,
        "section": record.section,
        "subsection": record.subsection,
        "subsubsection": record.subsubsection,
        "title_path": record.title_path,
        "text": record.text,
    }


def write_jsonl(
    records: list[dict],
    output_path: Path,
) -> None:
    """
    Write structured records as UTF-8 JSONL.

    One JSON object is written per line.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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
# PHASE 2F - CORPUS METADATA
# ============================================================

def write_corpus_metadata(
    corpus_path: Path,
    documents: list[ParsedDocument],
    total_input_documents: int,
) -> None:
    """
    Write corpus-level metadata and processing statistics.
    """

    document_entries: list[dict] = []

    total_records = 0
    total_removed = 0

    for document in documents:

        record_count = len(document.records)

        total_records += record_count
        total_removed += document.removed_records

        document_entries.append(
            {
                "document_id": document.metadata.document_id,
                "title": document.metadata.document_title,
                "source_file": document.metadata.source_file,
                "policy_id": document.metadata.policy_id,
                "version": document.metadata.version,
                "record_count": record_count,
                "records_removed": document.removed_records,
            }
        )

    corpus = {
        "corpus_name": "Aurelia Bank Synthetic Research Corpus",
        "synthetic": True,
        "source_format": "AsciiDoc",
        "document_count": total_input_documents,
        "documents_successfully_processed": len(documents),
        "record_count": total_records,
        "records_removed": total_removed,
        "documents": document_entries,
    }

    corpus_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with corpus_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            corpus,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# PHASE 2E + 2F - PROCESS COMPLETE CORPUS
# ============================================================

def process_corpus(
    raw_dir: Path,
    processed_dir: Path,
) -> None:
    """
    Process all AsciiDoc files in the raw corpus.

    Pipeline:

        .adoc files
            ↓
        parse
            ↓
        clean
            ↓
        filter unusable records
            ↓
        generate stable IDs
            ↓
        JSONL
            ↓
        corpus metadata
    """

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw directory not found: {raw_dir}"
        )

    adoc_files = sorted(
        raw_dir.glob("*.adoc")
    )

    if not adoc_files:
        raise FileNotFoundError(
            f"No .adoc files found in {raw_dir}"
        )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("PHASE 2E + 2F")
    print("PROCESSING ASCIIDOC CORPUS")
    print("=" * 80)

    parsed_documents: list[ParsedDocument] = []
    all_records: list[dict] = []

    total_removed = 0

    for index, file_path in enumerate(
        adoc_files,
        start=1,
    ):

        print(
            f"\n[{index}/{len(adoc_files)}] "
            f"{file_path.name}"
        )

        try:
            document = parse_asciidoc_file(
                file_path
            )

            parsed_documents.append(document)

            total_removed += document.removed_records

            print(
                f"  Title   : "
                f"{document.metadata.document_title}"
            )

            print(
                f"  Records : "
                f"{len(document.records)}"
            )

            print(
                f"  Removed : "
                f"{document.removed_records}"
            )

            for record in document.records:

                all_records.append(
                    record_to_dict(
                        record=record,
                        metadata=document.metadata,
                    )
                )

        except Exception as exc:
            print(
                f"  ERROR: {exc}"
            )
            raise

    # --------------------------------------------------------
    # Write sections.jsonl
    # --------------------------------------------------------

    jsonl_path = (
        processed_dir
        / "sections.jsonl"
    )

    write_jsonl(
        records=all_records,
        output_path=jsonl_path,
    )

    # --------------------------------------------------------
    # Write corpus.json
    # --------------------------------------------------------

    corpus_path = (
        processed_dir
        / "corpus.json"
    )

    write_corpus_metadata(
        corpus_path=corpus_path,
        documents=parsed_documents,
        total_input_documents=len(adoc_files),
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

    print(
        f"Documents found          : "
        f"{len(adoc_files)}"
    )

    print(
        f"Documents processed      : "
        f"{len(parsed_documents)}"
    )

    print(
        f"Total records generated  : "
        f"{len(all_records)}"
    )

    print(
        f"Total records removed    : "
        f"{total_removed}"
    )

    print(
        f"JSONL output             : "
        f"{jsonl_path}"
    )

    print(
        f"Corpus metadata          : "
        f"{corpus_path}"
    )


def main() -> None:
    """
    Run Phase 2E + 2F on the complete corpus.
    """

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

    process_corpus(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )


if __name__ == "__main__":
    main()