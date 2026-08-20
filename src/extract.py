"""Field extraction: build word-level BIO datasets from SROIE box/entities,
and fine-tune/run the bundled LayoutLM checkpoint as a token classifier.

SROIE's box/ files are line-level (one bbox per text line, not per word),
and entities/ gives the clean ground-truth field value with no position
info at all. To get LayoutLM's required (word, bbox, label) triples we:
  1. split each line proportionally by character offset into per-word boxes
  2. fuzzy-match each entity value against contiguous word spans to find
     which words it corresponds to, and tag those B-<FIELD>/I-<FIELD>
This matching is heuristic and imperfect (documented in error analysis).
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image

FIELDS = ["company", "date", "address", "total"]
LABELS = ["O"] + [f"{p}-{f.upper()}" for f in FIELDS for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().upper())


def parse_box_file(box_path: str | Path) -> list[dict]:
    """Return words in reading order with proportional per-word bboxes
    (pixel coords), derived from each line's bbox split by character width."""
    words = []
    for line in Path(box_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split(",", 8)
        if len(parts) != 9:
            continue
        x1, y1, x2, y2, x3, y3, x4, y4 = map(int, parts[:8])
        text = parts[8]
        x_left, x_right = min(x1, x4), max(x2, x3)
        y_top, y_bottom = min(y1, y2), max(y3, y4)
        width = max(x_right - x_left, 1)

        tokens = text.split()
        if not tokens:
            continue
        line_len = len(text)
        cursor = 0
        for tok in tokens:
            start = text.index(tok, cursor)
            end = start + len(tok)
            cursor = end
            wx1 = x_left + width * start / line_len
            wx2 = x_left + width * end / line_len
            words.append({"text": tok, "bbox": [round(wx1), y_top, round(wx2), y_bottom]})
    return words


def _find_best_span(words: list[dict], target: str, min_ratio: float = 0.6, max_span: int = 12):
    norm_target = _normalize(target)
    best_ratio, best_span = 0.0, None
    n = len(words)
    for start in range(n):
        concat = ""
        for end in range(start, min(start + max_span, n)):
            concat = (concat + " " + words[end]["text"]).strip()
            norm_concat = _normalize(concat)
            ratio = SequenceMatcher(None, norm_concat, norm_target).ratio()
            if ratio > best_ratio:
                best_ratio, best_span = ratio, (start, end)
            if len(norm_concat) > len(norm_target) * 2 + 10:
                break
    return best_span if best_ratio >= min_ratio else None


def _find_total_span(words: list[dict], target: str):
    try:
        target_val = float(re.sub(r"[^0-9.]", "", target))
    except ValueError:
        return _find_best_span(words, target, min_ratio=0.5)
    exact = [
        i
        for i, w in enumerate(words)
        if re.search(r"\d", w["text"]) and _amount_equal(w["text"], target_val)
    ]
    if "." in target:
        # prefer tokens formatted like the target (e.g. "6.00", not a bare "6"
        # that happens to equal it numerically, like a stray tax percentage)
        formatted = [i for i in exact if "." in words[i]["text"]]
        if formatted:
            exact = formatted
    if exact:
        return (exact[-1], exact[-1])  # grand total is usually the last matching amount
    return _find_best_span(words, target, min_ratio=0.5)


def _amount_equal(token: str, value: float) -> bool:
    try:
        return abs(float(re.sub(r"[^0-9.]", "", token)) - value) < 0.005
    except ValueError:
        return False


def label_document(box_path: str | Path, entities_path: str | Path) -> tuple[list[dict], list[str]]:
    words = parse_box_file(box_path)
    labels = ["O"] * len(words)
    entities = json.loads(Path(entities_path).read_text(encoding="utf-8", errors="ignore"))

    for field in FIELDS:
        value = entities.get(field, "")
        if not value:
            continue
        if field == "total":
            span = _find_total_span(words, value)
        else:
            span = _find_best_span(words, value, max_span=25 if field == "address" else 12)
        if span is None:
            continue
        start, end = span
        if any(labels[i] != "O" for i in range(start, end + 1)):
            continue  # already claimed by another field, skip rather than overwrite
        labels[start] = f"B-{field.upper()}"
        for i in range(start + 1, end + 1):
            labels[i] = f"I-{field.upper()}"
    return words, labels


def normalize_bbox(bbox: list[int], img_size: tuple[int, int]) -> list[int]:
    w, h = img_size
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(1000, round(1000 * x1 / w))),
        max(0, min(1000, round(1000 * y1 / h))),
        max(0, min(1000, round(1000 * x2 / w))),
        max(0, min(1000, round(1000 * y2 / h))),
    ]


def build_split(split_dir: str | Path) -> list[dict]:
    """One example per receipt: words, normalized boxes, BIO labels."""
    split_dir = Path(split_dir)
    box_dir, ent_dir, img_dir = split_dir / "box", split_dir / "entities", split_dir / "img"
    examples = []
    for box_path in sorted(box_dir.iterdir()):
        stem = box_path.stem
        ent_path = ent_dir / f"{stem}.txt"
        img_path = img_dir / f"{stem}.jpg"
        if not ent_path.exists() or not img_path.exists():
            continue
        words, labels = label_document(box_path, ent_path)
        if not words:
            continue
        img_size = Image.open(img_path).size
        boxes = [normalize_bbox(w["bbox"], img_size) for w in words]
        examples.append(
            {
                "id": stem,
                "words": [w["text"] for w in words],
                "boxes": boxes,
                "labels": labels,
            }
        )
    return examples


def _tokenize_and_align(examples: dict, tokenizer, max_length: int = 512) -> dict:
    """Word-piece tokenize with pre-split words. LayoutLM (v1) has no
    box-aware tokenizer, so bbox/labels are aligned to subword tokens by
    hand via word_ids: each subword inherits its word's bbox, special/pad
    tokens get bbox [0,0,0,0], and only a word's first subword carries the
    label (rest get -100, ignored by the loss) - the standard HF NER
    convention."""
    tokenized = tokenizer(
        examples["words"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    all_labels, all_boxes = [], []
    for i, label_ids in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        boxes = examples["boxes"][i]
        prev_word_id = None
        aligned_labels, aligned_boxes = [], []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
                aligned_boxes.append([0, 0, 0, 0])
            else:
                aligned_boxes.append(boxes[word_id])
                if word_id != prev_word_id:
                    aligned_labels.append(LABEL2ID[label_ids[word_id]])
                else:
                    aligned_labels.append(-100)
            prev_word_id = word_id
        all_labels.append(aligned_labels)
        all_boxes.append(aligned_boxes)
    tokenized["labels"] = all_labels
    tokenized["bbox"] = all_boxes
    return tokenized


def _load_layoutlm_checkpoint(checkpoint_dir: str, num_labels: int, id2label: dict, label2id: dict):
    """The bundled Kaggle SROIE2019 `layoutlm-base-uncased` checkpoint was saved
    under transformers' pre-standardization key prefix (`bert.*`, including the
    LayoutLM-specific x/y/h/w spatial position embeddings) rather than the
    `layoutlm.*` prefix current `LayoutLMModel` expects, so a plain
    `from_pretrained` silently drops every pretrained weight as UNEXPECTED/
    MISSING and trains from random init. Remap the prefix (and drop the MLM
    head, unused for token classification) before loading."""
    import torch
    from transformers import LayoutLMConfig, LayoutLMForTokenClassification

    raw_state = torch.load(f"{checkpoint_dir}/pytorch_model.bin", map_location="cpu", weights_only=True)
    remapped = {}
    for key, value in raw_state.items():
        if key.startswith("cls."):
            continue
        if key.startswith("bert."):
            key = "layoutlm." + key[len("bert.") :]
        remapped[key] = value

    config = LayoutLMConfig.from_pretrained(
        checkpoint_dir, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    model = LayoutLMForTokenClassification(config)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # only the randomly-initialized classifier/pooler heads should be missing;
    # anything else here means the remap didn't actually line up with the model.
    unexpected_missing = [m for m in missing if not m.startswith(("classifier.", "layoutlm.pooler."))]
    if unexpected_missing or unexpected:
        raise RuntimeError(f"layoutlm checkpoint remap mismatch: missing={unexpected_missing} unexpected={unexpected}")
    return model


def fine_tune_layoutlm(
    train_examples: list[dict],
    eval_examples: list[dict],
    checkpoint_dir: str,
    output_dir: str,
    epochs: int = 10,
):
    from datasets import Dataset
    from transformers import (
        EarlyStoppingCallback,
        LayoutLMTokenizerFast,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    tokenizer = LayoutLMTokenizerFast.from_pretrained(checkpoint_dir)
    model = _load_layoutlm_checkpoint(checkpoint_dir, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    def to_hf(examples: list[dict]) -> Dataset:
        return Dataset.from_dict(
            {
                "words": [e["words"] for e in examples],
                "boxes": [e["boxes"] for e in examples],
                "labels": [e["labels"] for e in examples],
            }
        )

    remove_cols = ["words", "boxes"]
    train_ds = to_hf(train_examples).map(
        lambda ex: _tokenize_and_align(ex, tokenizer), batched=True, remove_columns=remove_cols
    )
    eval_ds = to_hf(eval_examples).map(
        lambda ex: _tokenize_and_align(ex, tokenizer), batched=True, remove_columns=remove_cols
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import f1_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_preds = preds[labels != -100]
        true_labels = labels[labels != -100]
        return {"micro_f1": f1_score(true_labels, true_preds, average="micro")}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer


def get_extractor(model_dir: str):
    """Load the LayoutLM tokenizer+model once. `model_dir` works transparently
    as either a local path or a Hugging Face Hub repo_id (both are accepted by
    `from_pretrained`). Callers (e.g. a Streamlit `st.cache_resource` wrapper)
    hold onto the returned tuple across requests instead of reloading it every
    call -- see CONCERNS.md's "Model loaded per-request in field extraction"."""
    from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast

    tokenizer = LayoutLMTokenizerFast.from_pretrained(model_dir)
    model = LayoutLMForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def _predict_word_labels(
    words: list[str],
    boxes: list[list[int]],
    model_dir: str | None = None,
    extractor: tuple | None = None,
) -> list[str]:
    """Run token classification, return one BIO label per input word.

    Pass a pre-loaded `extractor=(tokenizer, model)` (e.g. from a cached
    loader) to skip `get_extractor` entirely and avoid reloading weights on
    every call; omit it to load fresh from `model_dir` as before."""
    import torch

    tokenizer, model = extractor if extractor is not None else get_extractor(model_dir)

    enc = tokenizer(words, is_split_into_words=True, truncation=True, return_tensors="pt")
    bbox = [[0, 0, 0, 0]] * len(enc["input_ids"][0])
    word_ids = enc.word_ids(batch_index=0)
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            bbox[i] = boxes[word_id]
    enc["bbox"] = torch.tensor([bbox])

    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred_ids = logits.argmax(dim=-1).tolist()

    word_labels = ["O"] * len(words)
    seen = set()
    for word_id, label_id in zip(word_ids, pred_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        word_labels[word_id] = model.config.id2label[label_id]
    return word_labels


def predict_fields(
    words: list[str],
    boxes: list[list[int]],
    model_dir: str | None = None,
    extractor: tuple | None = None,
) -> dict[str, str]:
    """Run token classification and collapse B-/I- spans back into field strings.

    Pass a pre-loaded `extractor=(tokenizer, model)` to skip a fresh
    `from_pretrained` load; omit it to load fresh from `model_dir`."""
    word_labels = _predict_word_labels(words, boxes, model_dir, extractor=extractor)
    fields: dict[str, list[str]] = {f: [] for f in FIELDS}
    for word, label in zip(words, word_labels):
        if label == "O":
            continue
        fields[label.split("-", 1)[1].lower()].append(word)
    return {f: " ".join(v) for f, v in fields.items()}


def evaluate_extraction(split_dir: str | Path, model_dir: str) -> dict:
    """Field-level token P/R/F1 (predicted vs. our heuristic BIO ground
    truth, word-index aligned) and exact-match accuracy per field
    (predicted joined string vs. the original entities/ ground truth)."""
    split_dir = Path(split_dir)
    box_dir, ent_dir, img_dir = split_dir / "box", split_dir / "entities", split_dir / "img"

    tp = {f: 0 for f in FIELDS}
    fp = {f: 0 for f in FIELDS}
    fn = {f: 0 for f in FIELDS}
    exact_match = {f: 0 for f in FIELDS}
    total = {f: 0 for f in FIELDS}

    for box_path in sorted(box_dir.iterdir()):
        stem = box_path.stem
        ent_path, img_path = ent_dir / f"{stem}.txt", img_dir / f"{stem}.jpg"
        if not ent_path.exists() or not img_path.exists():
            continue
        words_dicts = parse_box_file(box_path)
        if not words_dicts:
            continue
        img_size = Image.open(img_path).size
        words = [w["text"] for w in words_dicts]
        boxes = [normalize_bbox(w["bbox"], img_size) for w in words_dicts]

        pred_word_labels = _predict_word_labels(words, boxes, model_dir)
        entities = json.loads(ent_path.read_text(encoding="utf-8", errors="ignore"))
        _, gold_labels = label_document(box_path, ent_path)

        for field in FIELDS:
            fu = field.upper()
            for i in range(len(words)):
                gold_is_field = gold_labels[i] != "O" and gold_labels[i].split("-", 1)[1] == fu
                pred_is_field = pred_word_labels[i] != "O" and pred_word_labels[i].split("-", 1)[1] == fu
                if gold_is_field and pred_is_field:
                    tp[field] += 1
                elif pred_is_field and not gold_is_field:
                    fp[field] += 1
                elif gold_is_field and not pred_is_field:
                    fn[field] += 1

            pred_value = _normalize(" ".join(w for w, l in zip(words, pred_word_labels) if fu in l))
            gold_value = _normalize(entities.get(field, ""))
            total[field] += 1
            if gold_value and gold_value == pred_value:
                exact_match[field] += 1

    per_field = {}
    for f in FIELDS:
        precision = tp[f] / (tp[f] + fp[f]) if (tp[f] + fp[f]) else 0.0
        recall = tp[f] / (tp[f] + fn[f]) if (tp[f] + fn[f]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_field[f] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match_accuracy": exact_match[f] / total[f] if total[f] else 0.0,
        }
    return per_field
