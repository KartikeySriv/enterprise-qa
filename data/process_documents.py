from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class DocumentMetadata:
    """
    Document-level metadata extracted from the AsciiDoc header.
    """

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
    """
    Represents one logical block of text associated with
    an AsciiDoc heading.
    """

    heading_level: int
    section: Optional[str] = None
    subsection: Optional[str] = None
    subsubsection: Optional[str] = None
    title_path: list[str] = field(default_factory=list)
    text: str = ""


@dataclass
class ParsedDocument:
    """
    Represents the parsed structure of one AsciiDoc document.
    """

    metadata: DocumentMetadata
    records: list[SectionRecord]


# ============================================================
# REGEX
# ============================================================

# AsciiDoc headings:
#
# = Document Title
# == Section
# === Subsection
# ==== Subsubsection
HEADING_PATTERN = re.compile(r"^(=+)\s+(.+?)\s*$")

# Domain metadata lines in the document header.
#
# Example:
# Policy ID: AUR-RB-001
# Version: 3.2
# Classification: Internal - Employee Use
DOCUMENT_METADATA_PATTERN = re.compile(
    r"^(Policy ID|Version|Classification)\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)

# AsciiDoc attributes:
#
# :doctype: article
# :toc: left
# :sectnums:
ASCIIDOC_ATTRIBUTE_PATTERN = re.compile(
    r"^:([^:]+):\s*(.*?)\s*$"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_heading(line: str) -> bool:
    """
    Return True if the line is an AsciiDoc heading.
    """
    return HEADING_PATTERN.match(line) is not None


def parse_heading(line: str) -> tuple[int, str]:
    """
    Parse an AsciiDoc heading.

    Example:
        '=== Eligibility' -> (3, 'Eligibility')
    """
    match = HEADING_PATTERN.match(line)

    if not match:
        raise ValueError(f"Invalid heading line: {line!r}")

    level = len(match.group(1))
    title = match.group(2).strip()

    return level, title


def normalize_lines(lines: list[str]) -> list[str]:
    """
    Normalize line endings while preserving meaningful text.
    """
    normalized: list[str] = []

    for line in lines:
        line = line.replace("\r\n", "\n").replace("\r", "\n")
        normalized.append(line.rstrip())

    return normalized


def clean_metadata_value(value: Optional[str]) -> Optional[str]:
    """
    Conservatively clean a document metadata value.

    Removes obvious AsciiDoc continuation artifacts and
    surrounding whitespace without changing the actual value.
    """
    if value is None:
        return None

    value = value.strip()

    # AsciiDoc line continuation artifact.
    # Example:
    #   AUR-RB-001 +
    # becomes:
    #   AUR-RB-001
    value = re.sub(r"\s*\+\s*$", "", value)

    # Collapse repeated internal whitespace.
    value = re.sub(r"[ \t]+", " ", value)

    return value.strip()


def clean_text(text: str) -> str:
    """
    Conservatively clean document text.

    This function must preserve semantic meaning and wording.
    It performs formatting cleanup only.
    """

    if not text:
        return ""

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing whitespace from each line.
    lines = [line.rstrip() for line in text.split("\n")]

    # Remove completely empty lines from the beginning/end.
    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        # Collapse repeated spaces/tabs inside a line.
        stripped = re.sub(r"[ \t]+", " ", stripped)

        cleaned_lines.append(stripped)

    # Collapse 3+ consecutive blank lines into a single blank line.
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def build_text(lines: list[str]) -> str:
    """
    Build a clean text block from accumulated source lines.

    Paragraph boundaries are preserved.
    """
    if not lines:
        return ""

    raw_text = "\n".join(lines)

    return clean_text(raw_text)


def hierarchy_from_path(
    title_stack: dict[int, str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Convert heading hierarchy into convenient fields.
    """
    section = title_stack.get(2)
    subsection = title_stack.get(3)
    subsubsection = title_stack.get(4)

    return section, subsection, subsubsection


def create_record(
    heading_level: int,
    title_stack: dict[int, str],
    body_lines: list[str],
) -> Optional[SectionRecord]:
    """
    Create a SectionRecord if meaningful body text exists.

    A section does not require a subsection.
    """
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
    """
    Extract document-level metadata from the AsciiDoc header.

    Expected structure:

        = Document Title
        Organization
        :doctype: article
        :toc: left

        Policy ID: ...
        Version: ...
        Classification: ...

    Metadata extraction stops once the first level-2 section
    is encountered.
    """

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

        # Once the first real section is reached,
        # header metadata is complete.
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

        # --------------------------------------------
        # AsciiDoc attributes
        # --------------------------------------------
        attribute_match = ASCIIDOC_ATTRIBUTE_PATTERN.match(stripped)

        if attribute_match:
            key = attribute_match.group(1).strip().lower()
            value = clean_metadata_value(attribute_match.group(2))

            if key == "doctype":
                doctype = value

            continue

        # --------------------------------------------
        # Document metadata lines
        # --------------------------------------------
        metadata_match = DOCUMENT_METADATA_PATTERN.match(stripped)

        if metadata_match:
            key = metadata_match.group(1).strip().lower()
            value = clean_metadata_value(metadata_match.group(2))

            if key == "policy id":
                policy_id = value

            elif key == "version":
                version = value

            elif key == "classification":
                classification = value

            continue

        # --------------------------------------------
        # Organization
        # --------------------------------------------
        # In these generated documents the organization
        # appears immediately after the title.
        #
        # We only capture the first ordinary non-metadata
        # line after the document title.
        if title_seen and not organization_seen:
            organization = stripped
            organization_seen = True

    # Use filename stem as deterministic document ID.
    document_id = path.stem

    return DocumentMetadata(
        document_id=document_id,
        document_title=document_title,
        organization=organization,
        doctype=doctype,
        source_file=path.name,
        policy_id=policy_id,
        version=version,
        classification=classification,
    )


# ============================================================
# MAIN PARSER
# ============================================================

def parse_asciidoc_file(path: Path) -> ParsedDocument:
    """
    Parse one AsciiDoc file.

    Handles:

        = Title
        == Section
        === Subsection
        ==== Subsubsection

    Important behavior:

    1. Direct text under a section is preserved.
    2. A section does not require a subsection.
    3. Direct section text is kept separately when
       subsections also exist.
    4. Document-level metadata is extracted from the header.
    """

    if not path.exists():
        raise FileNotFoundError(f"AsciiDoc file not found: {path}")

    if path.suffix.lower() != ".adoc":
        raise ValueError(f"Expected .adoc file, got: {path}")

    raw_text = path.read_text(encoding="utf-8")

    lines = normalize_lines(raw_text.splitlines())

    # --------------------------------------------------------
    # Phase 2B: document metadata
    # --------------------------------------------------------
    metadata = extract_document_metadata(
        path=path,
        lines=lines,
    )

    # --------------------------------------------------------
    # Section parsing
    # --------------------------------------------------------
    title_stack: dict[int, str] = {}

    records: list[SectionRecord] = []

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
            records.append(record)

        current_body = []

    for line in lines:

        if is_heading(line):
            level, title = parse_heading(line)

            # --------------------------------------------
            # Document title
            # --------------------------------------------
            if level == 1:
                continue

            # --------------------------------------------
            # New section/subsection
            # --------------------------------------------
            flush_current_record()

            # Remove deeper levels from the current hierarchy.
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
            # Ignore header content before the first section.
            if current_heading_level is not None:
                current_body.append(line)

    # Flush final section.
    flush_current_record()

    return ParsedDocument(
        metadata=metadata,
        records=records,
    )


# ============================================================
# DEBUG OUTPUT
# ============================================================

def print_parsed_document(document: ParsedDocument) -> None:
    """
    Print parsed metadata and records for inspection.
    """

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
    print("=" * 80)

    for index, record in enumerate(document.records, start=1):
        print(f"\n[{index}]")
        print(f"  Heading level : {record.heading_level}")
        print(f"  Section       : {record.section}")
        print(f"  Subsection    : {record.subsection}")
        print(f"  Subsubsection : {record.subsubsection}")
        print(f"  Title path    : {' > '.join(record.title_path)}")
        print("  Text:")
        print(f"    {record.text}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    """
    Test the parser against the first raw document.
    """

    project_root = Path(__file__).resolve().parent.parent

    sample_file = (
        project_root
        / "data"
        / "raw"
        / "01_retail_banking_policy.adoc"
    )

    document = parse_asciidoc_file(sample_file)

    print_parsed_document(document)


if __name__ == "__main__":
    main()