from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)


# ============================================================
# PATHS / MODELS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "qa_dataset"
    / "processed"
    / "test.jsonl"
)

BASE_MODEL_NAME = "deepset/tinyroberta-squad2"

FINE_TUNED_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tinyroberta-aurelia-qa"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "qa_evaluation.json"
)


# ============================================================
# DATA
# ============================================================

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Load a JSONL file."""

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
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}"
                ) from exc

    return records


# ============================================================
# METRICS
# ============================================================

def normalize_answer(
    text: str,
) -> str:
    """
    Normalize answers only for metric comparison.

    The original dataset is never modified.
    """
    return " ".join(
        text.strip().lower().split()
    )


def exact_match(
    prediction: str,
    reference: str,
) -> int:
    """Return 1 when normalized strings match."""

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
            prediction_tokens
            == reference_tokens
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
# MODEL
# ============================================================

def load_model(
    model_name: str,
):
    """Load tokenizer and QA model."""

    print(
        f"Loading tokenizer: {model_name}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name
        )
    )

    print(
        f"Loading model: {model_name}"
    )

    model = (
        AutoModelForQuestionAnswering.from_pretrained(
            model_name
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# SINGLE QUESTION
# ============================================================

def predict_answer(
    question: str,
    context: str,
    tokenizer,
    model,
    device,
) -> dict[str, Any]:
    """
    Predict an extractive answer from question + context.

    This uses the same inference logic for both models.
    """

    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
    )

    input_ids = encoded[
        "input_ids"
    ][0]

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**encoded)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    start_index = int(
        torch.argmax(start_logits).item()
    )

    end_index = int(
        torch.argmax(end_logits).item()
    )

    # Invalid span.
    if end_index < start_index:
        return {
            "answer": "",
            "score": 0.0,
            "start_token": start_index,
            "end_token": end_index,
        }

    answer_tokens = input_ids[
        start_index:end_index + 1
    ]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True,
    ).strip()

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
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate one QA model on the frozen test set.
    """

    print("\n" + "=" * 70)
    print(
        f"EVALUATING: {model_name}"
    )
    print("=" * 70)

    tokenizer, model, device = (
        load_model(model_name)
    )

    print(
        f"Device: {device}"
    )

    predictions: list[dict[str, Any]] = []

    em_scores: list[int] = []
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

        result = predict_answer(
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

        em_scores.append(em)
        f1_scores.append(f1)

        predictions.append(
            {
                "id": record["id"],
                "document_id": record[
                    "document_id"
                ],
                "question": question,
                "reference_answer": (
                    reference_answer
                ),
                "prediction": prediction,
                "score": result["score"],
                "start_token": (
                    result["start_token"]
                ),
                "end_token": (
                    result["end_token"]
                ),
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

    average_em = (
        sum(em_scores)
        / len(em_scores)
        if em_scores
        else 0.0
    )

    average_f1 = (
        sum(f1_scores)
        / len(f1_scores)
        if f1_scores
        else 0.0
    )

    return {
        "model": model_name,
        "device": str(device),
        "examples": len(records),
        "exact_match": average_em,
        "token_f1": average_f1,
        "predictions": predictions,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PHASE 6A-6C - QA MODEL COMPARISON")
    print("=" * 70)

    records = load_jsonl(TEST_PATH)

    print(
        f"Frozen test examples: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # 6B - BASE MODEL
    # --------------------------------------------------------

    baseline_results = evaluate_model(
        model_name=BASE_MODEL_NAME,
        records=records,
    )

    # --------------------------------------------------------
    # 6C - FINE-TUNED MODEL
    # --------------------------------------------------------

    fine_tuned_results = evaluate_model(
        model_name=str(
            FINE_TUNED_MODEL_PATH
        ),
        records=records,
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    baseline_em = baseline_results[
        "exact_match"
    ]

    baseline_f1 = baseline_results[
        "token_f1"
    ]

    fine_tuned_em = fine_tuned_results[
        "exact_match"
    ]

    fine_tuned_f1 = fine_tuned_results[
        "token_f1"
    ]

    comparison = {
        "baseline": {
            "model": BASE_MODEL_NAME,
            "exact_match": baseline_em,
            "token_f1": baseline_f1,
        },
        "fine_tuned": {
            "model": str(
                FINE_TUNED_MODEL_PATH
            ),
            "exact_match": fine_tuned_em,
            "token_f1": fine_tuned_f1,
        },
        "improvement": {
            "exact_match": (
                fine_tuned_em
                - baseline_em
            ),
            "token_f1": (
                fine_tuned_f1
                - baseline_f1
            ),
        },
    }

    final_report = {
        "test_dataset": str(TEST_PATH),
        "examples": len(records),
        "baseline": baseline_results,
        "fine_tuned": fine_tuned_results,
        "comparison": comparison,
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
            final_report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"Baseline Exact Match : "
        f"{baseline_em * 100:.2f}%"
    )

    print(
        f"Fine-tuned Exact Match: "
        f"{fine_tuned_em * 100:.2f}%"
    )

    print(
        f"EM improvement       : "
        f"{(fine_tuned_em - baseline_em) * 100:.2f} "
        f"percentage points"
    )

    print()

    print(
        f"Baseline Token F1    : "
        f"{baseline_f1 * 100:.2f}%"
    )

    print(
        f"Fine-tuned Token F1  : "
        f"{fine_tuned_f1 * 100:.2f}%"
    )

    print(
        f"F1 improvement       : "
        f"{(fine_tuned_f1 - baseline_f1) * 100:.2f} "
        f"percentage points"
    )

    print()

    print(
        f"Detailed results saved to:"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()