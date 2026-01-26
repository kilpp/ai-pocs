"""Minimal fine-tuning helper for custom sentiment models.

Expected CSV format:
    text,label
    I love this!,positive
    This is bad,negative
Label values are mapped via --labels JSON, e.g. '{"negative":0,"neutral":1,"positive":2}'.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, Tuple

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def prepare_labels(labels_json: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    forward = json.loads(labels_json)
    inverse = {v: k for k, v in forward.items()}
    return forward, inverse


def compute_metrics(eval_pred, id2label):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean().item()
    return {"accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a sentiment model")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV")
    parser.add_argument("--model_name", default="distilbert-base-multilingual-cased")
    parser.add_argument("--output_dir", default="./model-out")
    parser.add_argument(
        "--labels",
        required=True,
        help='JSON mapping, e.g. {"negative":0,"neutral":1,"positive":2}',
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    label2id, id2label = prepare_labels(args.labels)

    dataset = load_dataset("csv", data_files={"train": args.train_csv})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label={k: v for k, v in id2label.items()},
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="no",
        save_strategy="no",
        learning_rate=5e-5,
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, id2label),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
