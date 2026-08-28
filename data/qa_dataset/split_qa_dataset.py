#!/usr/bin/env python3
"""
split_qa_dataset.py

Purpose (ONLY):
    Read the validated combined extractive QA dataset at
        data/qa_dataset/processed/all_examples.jsonl
    and split it into train / validation / test sets using a
    document-stratified split, so that every document is represented
    in every split.

This script does NOT modify all_examples.jsonl.
This script does NOT generate, paraphrase, augment, oversample, or
undersample any examples. It only partitions existing records.

Outputs:
    data/qa_dataset/processed/train.jsonl
    data/qa_dataset/processed/validation.jsonl
    data/qa_dataset/processed/test.jsonl
    data/qa_dataset/processed/split_report.json
"""

import json
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR / "processed"

INPUT_PATH = PROCESSED_DIR / "all_examples.jsonl"
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
VAL_PATH = PROCESSED_DIR / "validation.jsonl"
TEST_PATH = PROCESSED_DIR / "test.jsonl"
REPORT_PATH = PROCESSED_DIR / "split_report.json"

# Per-document target split (documents have exactly 15 examples each in the
# expected dataset). This is used as a *guide*; the actual per-document
# split falls back to proportional rounding for documents that don't have
# exactly 15 examples, so the script still works if counts differ slightly.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_examples(path):
    if not path.exists():
        print(f"ERROR: input dataset not found at {path}", file=sys.stderr)
        sys.exit(1)

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: invalid JSON on line {line_num} of {path}: {e}",
                      file=sys.stderr)
                sys.exit(1)
            examples.append(record)
    return examples


def largest_remainder_targets(total, seed):
    """
    Compute dataset-wide target counts for train/validation/test out of
    `total` examples using largest-remainder rounding on the 70/15/15
    ratios. Ties in fractional remainder are broken deterministically
    using the seed, so results are reproducible.
    """
    raw = {
        "train": total * TRAIN_RATIO,
        "validation": total * VAL_RATIO,
        "test": total * TEST_RATIO,
    }
    floors = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floors.values())

    fracs = [(k, raw[k] - floors[k]) for k in ("train", "validation", "test")]
    rng = random.Random(seed)
    # Shuffle first so ties break deterministically-but-not-always-the-same-way
    # for a given seed, then do a stable sort by fractional part descending.
    rng.shuffle(fracs)
    fracs.sort(key=lambda x: x[1], reverse=True)

    targets = dict(floors)
    for i in range(remainder):
        targets[fracs[i % len(fracs)][0]] += 1

    assert sum(targets.values()) == total
    return targets


def stratified_split(examples, seed):
    """
    Split examples into train/validation/test with document stratification:
    every document contributes to every split, and each document's own
    examples are allocated close to a 70/15/15 split (10-or-11 / 2 / 2-or-3
    out of 15). The *which* documents round up on which split is chosen so
    that the overall dataset totals land close to the requested global
    targets (e.g. 105 / 22 / 23 out of 150), not just each document
    independently rounding the same way every time.
    """
    by_doc = defaultdict(list)
    for ex in examples:
        by_doc[ex["document_id"]].append(ex)

    doc_ids = sorted(by_doc.keys())
    total = len(examples)

    rng = random.Random(seed)

    # Shuffle each document's own examples deterministically (for choosing
    # *which* examples land in which split later).
    for doc_id in doc_ids:
        doc_examples = list(by_doc[doc_id])
        doc_examples.sort(key=lambda r: r.get("id", ""))
        rng.shuffle(doc_examples)
        by_doc[doc_id] = doc_examples

    # Step 1: per-document floor allocation (e.g. 10/2/2 out of 15), plus
    # each document's leftover ("remainder") examples still needing a split.
    floor_counts = {}
    remainder_pool = []  # list of doc_ids, one entry per leftover example
    for doc_id in doc_ids:
        n = len(by_doc[doc_id])
        raw = {
            "train": n * TRAIN_RATIO,
            "validation": n * VAL_RATIO,
            "test": n * TEST_RATIO,
        }
        floors = {k: int(v) for k, v in raw.items()}
        used = sum(floors.values())
        leftover = n - used
        floor_counts[doc_id] = floors
        remainder_pool.extend([doc_id] * leftover)

    # Step 2: dataset-wide target counts (e.g. 105 / 22 / 23), and how many
    # of those are still unmet by the per-document floors.
    global_targets = largest_remainder_targets(total, seed)
    floor_totals = {
        split: sum(floor_counts[d][split] for d in doc_ids)
        for split in ("train", "validation", "test")
    }
    needed = {
        split: max(0, global_targets[split] - floor_totals[split])
        for split in ("train", "validation", "test")
    }

    # Build a list of split labels matching `needed`, one per leftover slot
    # in remainder_pool, then shuffle deterministically and hand them out.
    # If `needed` totals don't exactly match len(remainder_pool) (can happen
    # with uneven document sizes), pad/trim by cycling through splits so
    # every leftover example still gets assigned somewhere.
    label_pool = []
    for split in ("train", "validation", "test"):
        label_pool.extend([split] * needed[split])
    while len(label_pool) < len(remainder_pool):
        label_pool.append(("train", "validation", "test")[len(label_pool) % 3])
    label_pool = label_pool[: len(remainder_pool)]

    rng.shuffle(remainder_pool)
    rng.shuffle(label_pool)

    extra_counts = defaultdict(lambda: defaultdict(int))
    for doc_id, split in zip(remainder_pool, label_pool):
        extra_counts[doc_id][split] += 1

    # Step 3: final per-document counts = floor + assigned extras.
    splits = {"train": [], "validation": [], "test": []}
    doc_distribution = {}

    for doc_id in doc_ids:
        counts = dict(floor_counts[doc_id])
        for split, extra in extra_counts[doc_id].items():
            counts[split] += extra

        doc_examples = by_doc[doc_id]
        assert sum(counts.values()) == len(doc_examples)

        train_examples = doc_examples[: counts["train"]]
        val_examples = doc_examples[counts["train"]: counts["train"] + counts["validation"]]
        test_examples = doc_examples[counts["train"] + counts["validation"]:]

        splits["train"].extend(train_examples)
        splits["validation"].extend(val_examples)
        splits["test"].extend(test_examples)

        doc_distribution[doc_id] = {
            "train": len(train_examples),
            "validation": len(val_examples),
            "test": len(test_examples),
        }

    return splits, doc_distribution


def question_type_distribution(records):
    counter = Counter(r.get("question_type", "unknown") for r in records)
    return dict(sorted(counter.items()))


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def run_validations(all_examples, splits, input_path_before_hash, input_path):
    problems = []

    total = len(all_examples)
    train, val, test = splits["train"], splits["validation"], splits["test"]

    # 1. counts equal total
    if len(train) + len(val) + len(test) != total:
        problems.append(
            f"Split counts ({len(train)}+{len(val)}+{len(test)}) do not sum to total_examples ({total})."
        )

    # 2. No example ID occurs in more than one split
    id_to_splits = defaultdict(set)
    for split_name, records in splits.items():
        for r in records:
            id_to_splits[r["id"]].add(split_name)
    dup_ids = {i: s for i, s in id_to_splits.items() if len(s) > 1}
    if dup_ids:
        problems.append(f"Duplicate IDs found across splits: {dup_ids}")

    # 3. No identical question+context pair occurs in more than one split
    qc_to_splits = defaultdict(set)
    for split_name, records in splits.items():
        for r in records:
            key = (r.get("question", ""), r.get("context", ""))
            qc_to_splits[key].add(split_name)
    dup_qc = [k for k, s in qc_to_splits.items() if len(s) > 1]
    if dup_qc:
        problems.append(f"Duplicate question/context pairs found across splits: {len(dup_qc)} pairs")

    # 4-6. Every document appears in train/validation/test
    all_docs = set(r["document_id"] for r in all_examples)
    for split_name, records in splits.items():
        split_docs = set(r["document_id"] for r in records)
        missing = all_docs - split_docs
        if missing:
            problems.append(f"Documents missing from {split_name}: {sorted(missing)}")

    # 7. Global split approx 70/15/15 (allow +/- 5 percentage points)
    if total > 0:
        train_pct = 100.0 * len(train) / total
        val_pct = 100.0 * len(val) / total
        test_pct = 100.0 * len(test) / total
        if not (65 <= train_pct <= 75):
            problems.append(f"Train percentage {train_pct:.1f}% is outside expected ~70% range.")
        if not (10 <= val_pct <= 20):
            problems.append(f"Validation percentage {val_pct:.1f}% is outside expected ~15% range.")
        if not (10 <= test_pct <= 20):
            problems.append(f"Test percentage {test_pct:.1f}% is outside expected ~15% range.")

    # 8. original all_examples.jsonl unchanged (hash before vs after)
    input_path_after_hash = hash_file(input_path)
    if input_path_before_hash != input_path_after_hash:
        problems.append("all_examples.jsonl was modified during script execution!")

    # 9. All answer_start values remain valid (answer text found at that offset in context)
    bad_answer_starts = []
    for split_name, records in splits.items():
        for r in records:
            answers = r.get("answers", {})
            texts = answers.get("text", [])
            starts = answers.get("answer_start", [])
            context = r.get("context", "")
            for text, start in zip(texts, starts):
                if not (0 <= start <= len(context)) or context[start:start + len(text)] != text:
                    bad_answer_starts.append(r["id"])
    if bad_answer_starts:
        problems.append(f"Invalid answer_start values for IDs: {bad_answer_starts}")

    # 10. Every split file is valid JSONL -- checked at write/reload time in main()

    return problems


def hash_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def validate_jsonl_file(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                return f"{path} is not valid JSONL at line {line_num}: {e}"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_hash_before = hash_file(INPUT_PATH)

    all_examples = load_examples(INPUT_PATH)
    total = len(all_examples)

    # Sanity check: unique IDs in the source dataset itself
    source_ids = [r["id"] for r in all_examples]
    if len(source_ids) != len(set(source_ids)):
        dupes = [i for i, c in Counter(source_ids).items() if c > 1]
        print(f"ERROR: all_examples.jsonl itself contains duplicate IDs: {dupes}", file=sys.stderr)
        sys.exit(1)

    splits, doc_distribution = stratified_split(all_examples, SEED)

    problems = run_validations(all_examples, splits, input_hash_before, INPUT_PATH)

    # Write output files
    write_jsonl(TRAIN_PATH, splits["train"])
    write_jsonl(VAL_PATH, splits["validation"])
    write_jsonl(TEST_PATH, splits["test"])

    # Re-validate that written files are valid JSONL (check #10)
    for path in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        err = validate_jsonl_file(path)
        if err:
            problems.append(err)

    # Re-check input file unchanged after writing outputs too
    input_hash_after = hash_file(INPUT_PATH)
    if input_hash_before != input_hash_after:
        problems.append("all_examples.jsonl was modified after writing split outputs!")

    train_count = len(splits["train"])
    val_count = len(splits["validation"])
    test_count = len(splits["test"])

    train_pct = 100.0 * train_count / total if total else 0.0
    val_pct = 100.0 * val_count / total if total else 0.0
    test_pct = 100.0 * test_count / total if total else 0.0

    qtype_distribution = {
        "train": question_type_distribution(splits["train"]),
        "validation": question_type_distribution(splits["validation"]),
        "test": question_type_distribution(splits["test"]),
    }

    report = {
        "seed": SEED,
        "total_examples": total,
        "train_count": train_count,
        "validation_count": val_count,
        "test_count": test_count,
        "train_percentage": round(train_pct, 2),
        "validation_percentage": round(val_pct, 2),
        "test_percentage": round(test_pct, 2),
        "document_distribution": doc_distribution,
        "question_type_distribution": qtype_distribution,
        "validation_problems": problems,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Duplicate checks for terminal summary
    id_to_splits = defaultdict(set)
    qc_to_splits = defaultdict(set)
    for split_name, records in splits.items():
        for r in records:
            id_to_splits[r["id"]].add(split_name)
            key = (r.get("question", ""), r.get("context", ""))
            qc_to_splits[key].add(split_name)
    dup_id_count = sum(1 for s in id_to_splits.values() if len(s) > 1)
    dup_qc_count = sum(1 for s in qc_to_splits.values() if len(s) > 1)

    all_docs = set(r["document_id"] for r in all_examples)
    docs_in_train = len(set(r["document_id"] for r in splits["train"]) & all_docs)
    docs_in_val = len(set(r["document_id"] for r in splits["validation"]) & all_docs)
    docs_in_test = len(set(r["document_id"] for r in splits["test"]) & all_docs)
    n_docs = len(all_docs)

    print("=" * 40)
    print("QA DATASET SPLIT")
    print("=" * 40)
    print(f"Total examples : {total}")
    print(f"Train          : {train_count}")
    print(f"Validation     : {val_count}")
    print(f"Test           : {test_count}")
    print()
    print("Approximate ratios:")
    print(f"Train          : {train_pct:.1f}%")
    print(f"Validation     : {val_pct:.1f}%")
    print(f"Test           : {test_pct:.1f}%")
    print()
    print("Documents represented in:")
    print(f"Train          : {docs_in_train}/{n_docs}")
    print(f"Validation     : {docs_in_val}/{n_docs}")
    print(f"Test           : {docs_in_test}/{n_docs}")
    print()
    print(f"Duplicate IDs across splits: {dup_id_count}")
    print(f"Duplicate question/context pairs: {dup_qc_count}")
    print()

    if problems:
        print("VALIDATION FAILED:")
        for p in problems:
            print(f" - {p}")
        print()
        print("Output written, but please review the problems above.")
        sys.exit(1)
    else:
        print("All validations passed.")
        print()
        print("Output:")
        print(f"{TRAIN_PATH.relative_to(SCRIPT_DIR.parent.parent)}")
        print(f"{VAL_PATH.relative_to(SCRIPT_DIR.parent.parent)}")
        print(f"{TEST_PATH.relative_to(SCRIPT_DIR.parent.parent)}")
        print(f"{REPORT_PATH.relative_to(SCRIPT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
