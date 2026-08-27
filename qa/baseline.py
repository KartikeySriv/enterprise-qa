from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)


MODEL_NAME = "deepset/tinyroberta-squad2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "qa_dataset"
    / "processed"
    / "test.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "baseline_predictions.json"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""

    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {path}"
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
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


# ============================================================
# ANSWER COMPARISON
# ============================================================

def normalize_answer(text: str) -> str:
    """
    Normalize an answer only for evaluation.

    The original answer in the dataset is never modified.
    """

    return " ".join(
        text.strip().lower().split()
    )


def exact_match(
    prediction: str,
    reference: str,
) -> int:
    """Return 1 when normalized answers match exactly."""

    return int(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


def token_f1(
    prediction: str,
    reference: str,
) -> float:
    """Calculate token-level F1."""

    prediction_tokens = (
        normalize_answer(prediction).split()
    )

    reference_tokens = (
        normalize_answer(reference).split()
    )

    if not prediction_tokens or not reference_tokens:
        return float(
            prediction_tokens == reference_tokens
        )

    prediction_counts: dict[str, int] = {}

    for token in prediction_tokens:
        prediction_counts[token] = (
            prediction_counts.get(token, 0)
            + 1
        )

    reference_counts: dict[str, int] = {}

    for token in reference_tokens:
        reference_counts[token] = (
            reference_counts.get(token, 0)
            + 1
        )

    common = 0

    for token, count in prediction_counts.items():
        common += min(
            count,
            reference_counts.get(token, 0),
        )

    if common == 0:
        return 0.0

    precision = (
        common / len(prediction_tokens)
    )

    recall = (
        common / len(reference_tokens)
    )

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_tokenizer():
    """Load the pretrained tinyRoBERTa QA model."""

    print(
        f"Loading tokenizer: {MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print(
        f"Loading model: {MODEL_NAME}"
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME
    )

    # Use GPU when available, otherwise CPU.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    print(f"Device: {device}")

    return tokenizer, model, device


# ============================================================
# SINGLE QUESTION PREDICTION
# ============================================================

def answer_question(
    question: str,
    context: str,
    tokenizer,
    model,
    device,
) -> dict[str, Any]:
    """
    Run extractive QA directly through the model.

    The model predicts:
        start position
        end position

    which are then converted back into text.
    """

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Keep a CPU copy for decoding.
    input_ids = inputs["input_ids"][0]

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Best start and end positions.
    start_index = int(
        torch.argmax(start_logits).item()
    )

    end_index = int(
        torch.argmax(end_logits).item()
    )

    # Invalid span: return empty answer.
    if end_index < start_index:
        return {
            "answer": "",
            "score": 0.0,
            "start_token": start_index,
            "end_token": end_index,
        }

    answer_tokens = input_ids[
        start_index : end_index + 1
    ]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True,
    ).strip()

    # Approximate confidence.
    start_probability = torch.softmax(
        start_logits,
        dim=-1,
    )[start_index]

    end_probability = torch.softmax(
        end_logits,
        dim=-1,
    )[end_index]

    score = float(
        (
            start_probability
            * end_probability
        ).item()
    )

    return {
        "answer": answer,
        "score": score,
        "start_token": start_index,
        "end_token": end_index,
    }


# ============================================================
# MAIN BASELINE EVALUATION
# ============================================================

def main() -> None:

    print("=" * 70)
    print("TINYROBERTA SQUAD2 - BASELINE EVALUATION")
    print("=" * 70)

    records = load_jsonl(TEST_PATH)

    print(
        f"Test examples: {len(records)}"
    )

    tokenizer, model, device = (
        load_model_and_tokenizer()
    )

    predictions: list[dict[str, Any]] = []

    exact_matches: list[int] = []
    f1_scores: list[float] = []

    for index, record in enumerate(
        records,
        start=1,
    ):

        question = record["question"]
        context = record["context"]

        reference_answer = (
            record["answers"]["text"][0]
        )

        result = answer_question(
            question=question,
            context=context,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        prediction = result["answer"]

        em = exact_match(
            prediction,
            reference_answer,
        )

        f1 = token_f1(
            prediction,
            reference_answer,
        )

        exact_matches.append(em)
        f1_scores.append(f1)

        predictions.append(
            {
                "id": record["id"],
                "document_id": record["document_id"],
                "question": question,
                "reference_answer": reference_answer,
                "prediction": prediction,
                "score": result["score"],
                "start_token": result["start_token"],
                "end_token": result["end_token"],
                "exact_match": em,
                "f1": f1,
            }
        )

        print(
            f"[{index:02d}/{len(records):02d}] "
            f"EM={em} "
            f"F1={f1:.3f} "
            f"Q={question}"
        )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    average_em = (
        sum(exact_matches)
        / len(exact_matches)
        if exact_matches
        else 0.0
    )

    average_f1 = (
        sum(f1_scores)
        / len(f1_scores)
        if f1_scores
        else 0.0
    )

    report = {
        "model": MODEL_NAME,
        "dataset": "test.jsonl",
        "device": str(device),
        "examples": len(records),
        "exact_match": average_em,
        "token_f1": average_f1,
        "predictions": predictions,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(
        f"Exact Match : "
        f"{average_em:.4f} "
        f"({average_em * 100:.2f}%)"
    )

    print(
        f"Token F1    : "
        f"{average_f1:.4f} "
        f"({average_f1 * 100:.2f}%)"
    )

    print(
        "\nPredictions saved to:"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()