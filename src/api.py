"""FastAPI endpoint: upload a receipt/invoice image, get back structured JSON.

Runs the full pipeline: OCR -> doc classification -> field extraction.
Model artifacts are loaded once at startup from models/artifacts/.
"""
from __future__ import annotations

import io
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from classify import DistilBertTextClassifier
from extract import predict_fields
from ocr import run_tesseract

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "models/artifacts")
CLASSIFIER_DIR = os.path.join(ARTIFACTS_DIR, "distilbert-classifier")
EXTRACTOR_DIR = os.path.join(ARTIFACTS_DIR, "layoutlm-extractor")

app = FastAPI(title="IDP Financial Documents")

_classifier: DistilBertTextClassifier | None = None


def _get_classifier() -> DistilBertTextClassifier:
    global _classifier
    if _classifier is None:
        if not os.path.isdir(CLASSIFIER_DIR):
            raise HTTPException(503, f"classifier not found at {CLASSIFIER_DIR}; train it first")
        _classifier = DistilBertTextClassifier(CLASSIFIER_DIR)
    return _classifier


def _words_and_boxes_from_image(path: str) -> tuple[list[str], list[list[int]]]:
    """Tesseract word-level boxes (image_to_data), normalized to 0-1000 scale."""
    import pytesseract

    img = Image.open(path)
    w, h = img.size
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(text)
        boxes.append(
            [
                max(0, min(1000, round(1000 * x / w))),
                max(0, min(1000, round(1000 * y / h))),
                max(0, min(1000, round(1000 * (x + bw) / w))),
                max(0, min(1000, round(1000 * (y + bh) / h))),
            ]
        )
    return words, boxes


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/tiff"):
        raise HTTPException(400, f"unsupported content type: {file.content_type}")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        Image.open(io.BytesIO(contents)).convert("RGB").save(tmp, format="JPEG")
        tmp_path = tmp.name

    try:
        ocr_text = run_tesseract(tmp_path)
        doc_type, confidence = "unknown", None
        if os.path.isdir(CLASSIFIER_DIR):
            clf = _get_classifier()
            doc_type, confidence = clf.predict_with_confidence([ocr_text])[0]

        fields = {}
        if os.path.isdir(EXTRACTOR_DIR):
            words, boxes = _words_and_boxes_from_image(tmp_path)
            if words:
                fields = predict_fields(words, boxes, EXTRACTOR_DIR)

        return {
            "doc_type": doc_type,
            "confidence": confidence,
            "vendor": fields.get("company"),
            "date": fields.get("date"),
            "total": fields.get("total"),
            "address": fields.get("address"),
            "ocr_text": ocr_text,
        }
    finally:
        os.unlink(tmp_path)
