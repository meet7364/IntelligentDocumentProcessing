"""Display-layer cleanup for LayoutLM field-extraction output.

`src/extract.py::predict_fields()` collapses ALL words tagged with ANY
B-/I-<FIELD> label, across the entire document, into a single
`" ".join(...)` string per field -- it never tracks where one span ends
and the next begins, so multiple separate spans for the same field (e.g.
two DATE lines on an invoice) get flattened into one garbled
concatenation, and malformed digit groupings in totals pass straight
through unformatted.

This module reads the model's raw per-word predictions (via the same
`_predict_word_labels` internal `predict_fields()` itself calls) and
reconstructs real span boundaries, picks a single plausible span per
field, and normalizes its formatting for display. It never touches
`src/extract.py` -- no BIO-tagging, training, or model-inference logic
is modified here; every value returned is derived from a span the model
itself predicted, never fabricated or assembled from untagged data.
"""
from __future__ import annotations

import re

from src.extract import FIELDS, _predict_word_labels


def group_bio_spans(words: list[str], word_labels: list[str]) -> dict[str, list[str]]:
    """Group per-word (word, label) pairs into per-field ordered span lists.

    Each entry is one maximal contiguous run of `B-<FIELD>`/`I-<FIELD>`
    labels for that field: a `B-` label or a change in field always starts
    a new span; an `O` label closes whatever span is currently open; an
    orphan `I-<FIELD>` with no open span of that field starts a new span
    rather than being silently dropped. A field with no tagged words maps
    to an empty list.
    """
    spans: dict[str, list[str]] = {f: [] for f in FIELDS}
    current_field: str | None = None
    current_words: list[str] = []

    def flush() -> None:
        nonlocal current_field, current_words
        if current_field is not None and current_words:
            spans[current_field].append(" ".join(current_words))
        current_field = None
        current_words = []

    for word, label in zip(words, word_labels):
        if label == "O":
            flush()
            continue
        prefix, _, field = label.partition("-")
        field = field.lower()
        if prefix == "B" or field != current_field:
            flush()
            current_field = field
            current_words = [word]
        else:
            current_words.append(word)
    flush()
    return spans


def _group_bio_spans_boxed(
    words: list[str], boxes: list[list[int]], word_labels: list[str]
) -> dict[str, list[list[list[int]]]]:
    """Box-tracking mirror of `group_bio_spans`: identical B-/I-/O grouping
    control flow, but each flushed span appends its list of per-word
    `boxes` instead of the joined word string. For the same
    `(words, word_labels)` input, `_group_bio_spans_boxed(...)[field]` has
    the same length and span boundaries as `group_bio_spans(...)[field]`.
    """
    spans: dict[str, list[list[list[int]]]] = {f: [] for f in FIELDS}
    current_field: str | None = None
    current_boxes: list[list[int]] = []

    def flush() -> None:
        nonlocal current_field, current_boxes
        if current_field is not None and current_boxes:
            spans[current_field].append(current_boxes)
        current_field = None
        current_boxes = []

    for box, label in zip(boxes, word_labels):
        if label == "O":
            flush()
            continue
        prefix, _, field = label.partition("-")
        field = field.lower()
        if prefix == "B" or field != current_field:
            flush()
            current_field = field
            current_boxes = [box]
        else:
            current_boxes.append(box)
    flush()
    return spans


def field_boxes_for_display(
    words: list[str], boxes: list[list[int]], word_labels: list[str]
) -> dict[str, list[list[int]] | None]:
    """Map each field's *displayed* value back to the exact model-predicted
    word boxes behind it.

    For `company`: `None` if no company-tagged word exists, else the
    flattened concatenation of every company-tagged word's box, in
    document order (matching `clean_fields`'s join-every-span behavior).

    For `date`/`address`/`total`: `None` if that field has zero spans,
    else the boxes of the ONE span `_select_span_index` selects (per
    `_STRATEGIES`) -- guaranteed 1:1 with the span `clean_fields` already
    picks as that field's displayed string. Every returned box is a
    literal member of the input `boxes` list -- never fabricated or
    interpolated geometry.
    """
    spans = group_bio_spans(words, word_labels)
    boxed_spans = _group_bio_spans_boxed(words, boxes, word_labels)

    result: dict[str, list[list[int]] | None] = {}
    if spans["company"]:
        result["company"] = [box for span_boxes in boxed_spans["company"] for box in span_boxes]
    else:
        result["company"] = None

    for field in ("date", "address", "total"):
        i = _select_span_index(spans[field], _STRATEGIES[field])
        result[field] = boxed_spans[field][i] if i is not None else None

    return result


def predict_fields_with_boxes(
    words: list[str],
    boxes: list[list[int]],
    model_dir: str | None = None,
    extractor: tuple | None = None,
) -> tuple[dict[str, str | None], dict[str, list[list[int]] | None]]:
    """Run the underlying model inference exactly once, then return
    `(clean_fields(words, word_labels), field_boxes_for_display(words, boxes, word_labels))`
    for that same `word_labels` -- so callers needing both the display
    strings and their boxes (i.e. `streamlit_app.py`) never run the model
    twice.

    Pass a pre-loaded `extractor=(tokenizer, model)` to skip a fresh
    `from_pretrained` load; omit it to load fresh from `model_dir`.
    """
    word_labels = _predict_word_labels(words, boxes, model_dir, extractor=extractor)
    return clean_fields(words, word_labels), field_boxes_for_display(words, boxes, word_labels)


def _select_span_index(spans: list, strategy: str) -> int | None:
    """Pick the index of one span from a field's span list per `strategy`.

    `"first"` picks `0`, `"last"` picks `len(spans) - 1`, `"longest"` picks
    the index of the span with the most words (ties broken by first
    occurrence, per `max`'s stable behavior). Returns `None` for an empty
    list. Works on any span list (strings or box-groups) since only
    `len()`/`.split()` (for `"longest"`, string spans only) are used.
    """
    if not spans:
        return None
    if strategy == "first":
        return 0
    if strategy == "last":
        return len(spans) - 1
    if strategy == "longest":
        return max(range(len(spans)), key=lambda i: len(spans[i].split()))
    raise ValueError(f"unknown span-selection strategy: {strategy!r}")


def _select_span(spans: list[str], strategy: str) -> str | None:
    """Pick one span from a field's span list per `strategy`.

    `"first"` picks `spans[0]`, `"last"` picks `spans[-1]`, `"longest"`
    picks the span with the most words. Returns `None` for an empty list.
    """
    i = _select_span_index(spans, strategy)
    return spans[i] if i is not None else None


def _whitespace_normalize(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def normalize_date(raw: str) -> str:
    """Collapse whitespace in a raw date span. Never raises -- falls back
    to the whitespace-normalized raw span on any failure."""
    try:
        return _whitespace_normalize(raw)
    except Exception:
        return _whitespace_normalize(raw)


def normalize_address(raw: str) -> str:
    """Collapse whitespace in a raw address span. Never raises -- falls
    back to the whitespace-normalized raw span on any failure."""
    try:
        return _whitespace_normalize(raw)
    except Exception:
        return _whitespace_normalize(raw)


def normalize_total(raw: str) -> str:
    """Re-group a raw total span's digits with standard thousands
    separators, e.g. `normalize_total("22,3000")` -> `"223,000"`: strips
    the malformed source comma, re-groups the surviving digits, and
    invents no decimal point since none was present in the source span.

    Never raises -- falls back to the whitespace-normalized raw span on
    any failure.
    """
    try:
        cleaned = re.sub(r"[^0-9.]", "", raw)
        integer_part, _, decimal_part = cleaned.partition(".")
        integer_part = integer_part.lstrip("0") or "0"
        formatted = f"{int(integer_part):,}"
        if decimal_part:
            formatted = f"{formatted}.{decimal_part}"
        return formatted
    except Exception:
        return _whitespace_normalize(raw)


# Span-selection strategy per field: "first" for date, "longest" for
# address, "last" for total, mirroring `_find_total_span`'s existing
# "grand total is usually last" comment in src/extract.py. Easily changed
# later by editing this dict -- no downstream contract depends on it.
_STRATEGIES = {"date": "first", "address": "longest", "total": "last"}
_NORMALIZERS = {"date": normalize_date, "address": normalize_address, "total": normalize_total}


def clean_fields(words: list[str], word_labels: list[str]) -> dict[str, str | None]:
    """Select and normalize one span per field from the model's raw
    per-word predictions.

    `date`/`address`/`total` each pick one span (per `_STRATEGIES`) and
    normalize it; a field with zero spans maps to `None`. `company` is
    `" ".join(spans["company"])` -- byte-identical to `predict_fields()`'s
    existing behavior for that field, since post-processing only cleans
    up date/address/total.
    """
    spans = group_bio_spans(words, word_labels)
    result: dict[str, str | None] = {}
    for field in ("date", "address", "total"):
        selected = _select_span(spans[field], _STRATEGIES[field])
        result[field] = _NORMALIZERS[field](selected) if selected is not None else None
    result["company"] = " ".join(spans["company"])
    return result


def predict_fields_cleaned(
    words: list[str],
    boxes: list[list[int]],
    model_dir: str | None = None,
    extractor: tuple | None = None,
) -> dict[str, str | None]:
    """Run the identical underlying model inference `predict_fields()`
    uses, then return `clean_fields(words, word_labels)` -- same return
    shape as `predict_fields()`, so this is a drop-in replacement at the
    `run_pipeline()` call site.

    Pass a pre-loaded `extractor=(tokenizer, model)` to skip a fresh
    `from_pretrained` load; omit it to load fresh from `model_dir`.
    """
    word_labels = _predict_word_labels(words, boxes, model_dir, extractor=extractor)
    return clean_fields(words, word_labels)
