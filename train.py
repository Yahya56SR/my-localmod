import os
import csv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    model_name = "UBC-NLP/MARBERTv2"

    dataset = load_dataset("ShoRit/DarijaDataset")

    if "validation" not in dataset and "test" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = {"train": split["train"], "test": split["test"]}
    elif "test" not in dataset:
        dataset["test"] = dataset["validation"]

    def normalize_example(example):
        text = example.get("text") or example.get("sentence") or example.get("tweet") or example.get("content")
        label = example.get("label") or example.get("class") or example.get("target") or example.get("offensive")
        return {"text": str(text), "label": int(label)}

    dataset["train"] = dataset["train"].map(normalize_example)
    dataset["test"] = dataset["test"].map(normalize_example)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = dataset["train"].map(tokenize_function, batched=True)
    tokenized_test = dataset["test"].map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Training on: {device}")
    print(f"Train samples: {len(tokenized_train)}, Test samples: {len(tokenized_test)}")

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained("./darija_moderator_model")
    tokenizer.save_pretrained("./darija_moderator_model")
    print("Model saved to ./darija_moderator_model")


if __name__ == "__main__":
    main()
