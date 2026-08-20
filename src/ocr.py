"""OCR layer: run Tesseract or EasyOCR over a receipt image."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytesseract
from PIL import Image


def run_tesseract(image_path: str | Path) -> str:
    return pytesseract.image_to_string(Image.open(image_path))


@lru_cache(maxsize=1)
def _easyocr_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False)


def run_easyocr(image_path: str | Path) -> str:
    result = _easyocr_reader().readtext(str(image_path), detail=0)
    return "\n".join(result)


def words_and_boxes_from_image(path: str) -> tuple[list[str], list[list[int]]]:
    """Tesseract word-level boxes (image_to_data), normalized to 0-1000 scale.

    Shared word/bbox extraction utility for the field-extraction path (used by
    `streamlit_app.py`). Ported as-is from `src/api.py::_words_and_boxes_from_image`
    to fix that module's duplicated normalization formula (see ARCHITECTURE.md);
    `src/api.py` itself is left unchanged this phase."""
    import pytesseract

    from src.extract import normalize_bbox

    img = Image.open(path)
    img_size = img.size
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(text)
        boxes.append(normalize_bbox([x, y, x + bw, y + bh], img_size))
    return words, boxes


def cer(hypothesis: str, reference: str) -> float:
    """Character error rate = Levenshtein distance / len(reference)."""
    ref, hyp = reference, hypothesis
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cost = 0 if rc == hc else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(hyp)] / len(ref)


def ground_truth_text(box_file: str | Path) -> str:
    """SROIE box/ files are line-level: 8 bbox coords + text, comma-separated."""
    lines = []
    for line in Path(box_file).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split(",", 8)
        if len(parts) == 9:
            lines.append(parts[8])
    return "\n".join(lines)
