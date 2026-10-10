import os
import glob
import csv
import torch
import pandas as pd
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


def prepare_dataset(df):
    text_col = None
    label_col = None

    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ('text', 'sentence', 'tweet', 'content', 'comments'):
            text_col = col
        elif col_lower in ('label', 'class', 'target', 'offensive'):
            label_col = col

    if text_col is None:
        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    if label_col is None:
        label_col = df.columns[-1]

    df = df[[text_col, label_col]].copy()
    df.columns = ['text', 'label']
    df['text'] = df['text'].astype(str)
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.9)

    os.makedirs("data", exist_ok=True)
    df.iloc[:split_idx].to_csv("data/train.csv", index=False, quoting=csv.QUOTE_ALL)
    df.iloc[split_idx:].to_csv("data/test.csv", index=False, quoting=csv.QUOTE_ALL)

    print(f"Train: {split_idx}, Test: {len(df) - split_idx}")
    print(f"Columns used: text='{text_col}', label='{label_col}'")


def find_and_prepare_csv():
    csv_files = glob.glob("data/*.csv")
    csv_files = [f for f in csv_files if f not in ("data/train.csv", "data/test.csv")]

    if not csv_files:
        raise FileNotFoundError("No CSV found in data/. Place your dataset CSV in the data/ folder.")

    filename = csv_files[0]
    print(f"Found dataset: {filename}")
    df = pd.read_csv(filename)
    prepare_dataset(df)


def main():
    model_name = "UBC-NLP/MARBERTv2"

    if not (os.path.exists("data/train.csv") and os.path.exists("data/test.csv")):
        find_and_prepare_csv()

    dataset = load_dataset("csv", data_files={"train": "data/train.csv", "test": "data/test.csv"})

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Training on: {device}")

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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained("./darija_moderator_model")
    tokenizer.save_pretrained("./darija_moderator_model")
    print("Model saved to ./darija_moderator_model")


if __name__ == "__main__":
    main()
