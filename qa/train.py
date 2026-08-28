from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "deepset/tinyroberta-squad2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZED_DIR = (
    PROJECT_ROOT
    / "data"
    / "qa_dataset"
    / "processed"
    / "tokenized"
)

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "tinyroberta-aurelia-qa"
)

TRAIN_DATASET_PATH = TOKENIZED_DIR / "train"
VALIDATION_DATASET_PATH = TOKENIZED_DIR / "validation"


# First controlled experiment.
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

WEIGHT_DECAY = 0.01

SEED = 42


# ============================================================
# DATA
# ============================================================

def load_datasets():
    """Load tokenized train and validation datasets."""

    train_dataset = load_from_disk(
        str(TRAIN_DATASET_PATH)
    )

    validation_dataset = load_from_disk(
        str(VALIDATION_DATASET_PATH)
    )

    return train_dataset, validation_dataset


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PHASE 5B-4 - TINYROBERTA FINE-TUNING")
    print("=" * 70)

    print(f"Model: {MODEL_NAME}")

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Load tokenized datasets
    # --------------------------------------------------------

    print("\nLoading tokenized datasets...")

    train_dataset, validation_dataset = (
        load_datasets()
    )

    print(
        f"Train examples      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation examples : "
        f"{len(validation_dataset)}"
    )

    # --------------------------------------------------------
    # Load pretrained QA model
    # --------------------------------------------------------

    print("\nLoading pretrained QA model...")

    model = (
        AutoModelForQuestionAnswering.from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------
    # Show parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Device information
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Training device      : "
        f"{device}"
    )

    if device == "cuda":
        print(
            f"GPU                  : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Data collator
    # --------------------------------------------------------

    data_collator = DefaultDataCollator()

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),

        # Training
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        weight_decay=WEIGHT_DECAY,

        # Evaluation
        eval_strategy="epoch",

        # Checkpoints
        save_strategy="epoch",
        save_total_limit=2,

        load_best_model_at_end=True,

        # Logging
        logging_strategy="epoch",

        # Reproducibility
        seed=SEED,
        data_seed=SEED,

        # Keep the experiment local.
        report_to="none",

        # Do not upload anything to Hugging Face.
        push_to_hub=False,

        # Better control over unused columns.
        remove_unused_columns=True,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    train_result = trainer.train()

    # --------------------------------------------------------
    # Save final/best model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    trainer.save_model(
        str(MODEL_OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(MODEL_OUTPUT_DIR)
    )

    # --------------------------------------------------------
    # Save training metrics
    # --------------------------------------------------------

    metrics = train_result.metrics

    trainer.log_metrics(
        "train",
        metrics,
    )

    trainer.save_metrics(
        "train",
        metrics,
    )

    trainer.save_state()

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    validation_metrics = trainer.evaluate()

    trainer.log_metrics(
        "validation",
        validation_metrics,
    )

    trainer.save_metrics(
        "validation",
        validation_metrics,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)

    print(
        f"Model saved to:\n"
        f"{MODEL_OUTPUT_DIR}"
    )

    print(
        "\nTraining metrics:"
    )

    for key, value in metrics.items():
        print(
            f"  {key}: {value}"
        )

    print(
        "\nValidation metrics:"
    )

    for key, value in validation_metrics.items():
        print(
            f"  {key}: {value}"
        )


if __name__ == "__main__":
    main()