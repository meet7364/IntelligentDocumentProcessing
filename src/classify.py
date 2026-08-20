"""Document classification: TF-IDF+LogReg baseline vs fine-tuned DistilBERT,
both operating on OCR text (not the raw image)."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def train_tfidf_logreg(train_texts: list[str], train_labels: list[str]) -> Pipeline:
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipe.fit(train_texts, train_labels)
    return pipe


def evaluate(y_true: list[str], y_pred: list[str]) -> str:
    return classification_report(y_true, y_pred, digits=3, zero_division=0)


class DistilBertTextClassifier:
    """Thin wrapper around a fine-tuned DistilBERT sequence classifier."""

    def __init__(self, model_dir: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, texts: list[str], batch_size: int = 16) -> list[str]:
        return [label for label, _ in self.predict_with_confidence(texts, batch_size)]

    def predict_with_confidence(self, texts: list[str], batch_size: int = 16) -> list[tuple[str, float]]:
        import torch

        id2label = self.model.config.id2label
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            results.extend((id2label[int(i)], float(c)) for i, c in zip(idx.tolist(), conf.tolist()))
        return results


def fine_tune_distilbert(
    train_texts: list[str],
    train_labels: list[int],
    eval_texts: list[str],
    eval_labels: list[int],
    label_names: list[str],
    output_dir: str,
    epochs: int = 4,
):
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_names), id2label=id2label, label2id=label2id
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize, batched=True)
    eval_ds = Dataset.from_dict({"text": eval_texts, "label": eval_labels}).map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, metrics
