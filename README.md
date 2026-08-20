# IDP for Financial Documents

An end-to-end intelligent document processing (IDP) pipeline: take a
scanned invoice/receipt image, run OCR, classify the document type, and
extract structured fields (vendor, date, address, total). Servable two
ways — a FastAPI endpoint for programmatic access, and a public Streamlit
web app for interactive use.

**Live demo:** [intelligentdocumentprocessing-tflappxeyxiatpb2trqsdg3.streamlit.app](https://intelligentdocumentprocessing-tflappxeyxiatpb2trqsdg3.streamlit.app/)
— upload a scanned invoice/receipt (or try a one-click sample) and see
real OCR, classification, and field-extraction results, including an
interactive panel that traces each extracted field back to the exact
region of the image it came from.

```
Scanned Document (image)
        |
        v
   OCR Layer (Tesseract, EasyOCR fallback)
        |  -> raw text + word-level bounding boxes
        v
   +----------------+----------------------+
   |                                        |
   v                                        v
Document Classification            Key Field Extraction
(fine-tuned DistilBERT,            (fine-tuned LayoutLM
4 classes, on OCR text)            token classifier, on
                                    OCR text + layout)
   |                                        |
   +-------------------+--------------------+
                        v
        Post-processing (src/postprocess.py) --
        dedupe/normalize the model's raw spans
        into a single clean value per field,
        never retraining, never fabricating
                        v
             Structured JSON output
             {doc_type, vendor, date, total, address, confidence}
                        |
           +------------+-------------+
           v                          v
FastAPI endpoint              Streamlit web app
(/process-document)           (streamlit_app.py, deployed to
                               Streamlit Community Cloud)
```

## Datasets

- **[SROIE](https://rrc.cvc.uab.es/?ch=13) (ICDAR 2019)** — ~1,000 scanned
  receipts, via the Kaggle `SROIE2019` package. Used for field extraction.
  Comes with per-word bounding boxes (`box/`), ground-truth field values
  (`entities/`: company, date, address, total), and a bundled
  `layoutlm-base-uncased` checkpoint.
- **RVL-CDIP subset** — the full dataset is 400K images / 38.8GB, far more
  than needed here. We use `vaclavpechtor/rvl_cdip-small-200` (200
  images/class), filtered to 4 classes (`invoice`, `letter`, `form`,
  `email`) — **640 train / 160 validation examples**, not the full
  dataset. This is called out explicitly rather than presented as if it
  were full-scale training.

Both are standard academic benchmarks with known baselines, so results
here are directly comparable rather than resting on an unverifiable claim.

## Results (real, reproduced by rerunning every notebook end-to-end)

**01 — OCR baseline** (25-receipt random sample, seeded):

| engine    | mean CER | mean time/image |
|-----------|----------|------------------|
| Tesseract | 0.436    | 2.73s            |
| EasyOCR   | 0.402    | 29.07s           |

Tesseract is the default OCR engine (comparable accuracy, ~10x faster);
EasyOCR is used as a fallback when Tesseract returns near-empty text (a
proxy for a page-segmentation failure). See the notebook's manual
inspection section for *why* each engine fails on its worst cases —
segmentation drop-outs vs. case/formatting noise in the CER metric
itself, not just raw error counts.

**02 — Document classification** (TF-IDF+LogReg baseline vs. fine-tuned
DistilBERT, 4 classes, 160 validation examples):

| model              | accuracy | macro-F1 |
|--------------------|----------|----------|
| TF-IDF + LogReg    | 0.781    | 0.784    |
| DistilBERT (4 epoch, early stopping) | 0.787 | 0.790 |

The transformer's edge over the baseline is small at this data scale
(640 train examples) — see `notebooks/02_.../` Summary for the
per-class breakdown, which is not uniform (DistilBERT actually
regresses on `form`).

**03 — Field extraction** (fine-tuned LayoutLM, 347 test receipts):

| field   | precision | recall | F1    | exact-match |
|---------|-----------|--------|-------|--------------|
| company | 0.973     | 0.988  | 0.980 | 0.882        |
| date    | 0.981     | 0.983  | 0.982 | 0.948        |
| address | 0.993     | 0.996  | 0.995 | 0.709        |
| total   | 0.817     | 0.850  | 0.833 | 0.732        |

`total` is the hardest field (ambiguity between subtotal/tax/grand-total
lines); `address` has near-perfect word-level F1 but the lowest
exact-match, since a full multi-word span has to match verbatim. See
`notebooks/03_.../` Summary for the full discussion, including how this
connects back to the heuristic BIO-labeling step used to build the
training data in the first place.

For the full reasoning behind every model/parameter choice — and why
each metric was chosen over the alternatives — see `explanation.pdf`.

## Deployment

The Streamlit app is live at
[intelligentdocumentprocessing-tflappxeyxiatpb2trqsdg3.streamlit.app](https://intelligentdocumentprocessing-tflappxeyxiatpb2trqsdg3.streamlit.app/),
hosted on Streamlit Community Cloud's free tier.

- **Model hosting:** both fine-tuned checkpoints (DistilBERT classifier,
  LayoutLM extractor) are published as public Hugging Face Hub repos
  (`meet7364/idp-distilbert-classifier`, `meet7364/idp-layoutlm-extractor`)
  and loaded via `from_pretrained(repo_id)` at runtime — independent of
  the Colab/Drive environment training happens in, and no `HF_TOKEN`
  secret needed on the deploy target.
- **Memory:** peak resident memory with both models loaded, measured live
  via an in-app `psutil` reading after a real document upload, is
  **1548.5 MB** — comfortably within Streamlit Community Cloud's free-tier
  ceiling, empirically confirmed rather than assumed.
- **Extraction quality on unfamiliar documents:** the extractor is
  fine-tuned on SROIE's single-vendor retail receipts; documents with a
  meaningfully different layout (e.g. multi-date invoices, multi-line
  billing addresses) can produce lower-quality field spans. A
  post-processing layer (`src/postprocess.py`) cleans up the model's raw
  output — deduping/normalizing malformed spans — but does not change
  what the model itself predicts. This is a known, documented limitation,
  not a deploy defect (see `.planning/PROJECT.md` for the live
  verification that established this).

## Repo structure

```
idp-financial-documents/
├── data/
│   ├── raw/            <- SROIE (not committed, see .gitignore)
│   └── processed/       <- derived jsonl/csv (not committed)
├── notebooks/
│   ├── 01_ocr_baseline.ipynb
│   ├── 02_classification_baseline_vs_bert.ipynb
│   └── 03_field_extraction_ner.ipynb
├── src/
│   ├── ocr.py            <- Tesseract/EasyOCR wrappers, CER metric
│   ├── classify.py        <- TF-IDF+LogReg baseline, DistilBERT fine-tune
│   ├── extract.py         <- SROIE BIO-labeling heuristics, LayoutLM fine-tune/eval
│   ├── postprocess.py     <- display-layer cleanup of the extractor's raw spans
│   └── api.py              <- FastAPI app
├── streamlit_app.py       <- Streamlit web app (deployed to Community Cloud)
├── assets/samples/        <- committed sample documents for the app's 1-click try buttons
├── .streamlit/config.toml  <- theme config (light-only)
├── models/artifacts/      <- trained weights (not committed, regenerate by rerunning 02/03)
├── packages.txt           <- system packages for Streamlit Cloud (tesseract-ocr)
├── requirements.txt       <- pip dependencies for Streamlit Cloud deployment
├── Dockerfile
├── pyproject.toml / uv.lock
└── explanation.pdf        <- extensive design-decision writeup
```

Notebooks are self-contained (all `src/` logic is inlined into cells) so
each one can be run standalone — e.g. dropped into Google Colab — without
needing the rest of the repo on the path. `src/*.py` stays as the
canonical, importable version of the same code for `api.py`,
`streamlit_app.py`, and local development.

## Setup

```bash
uv sync
```

Everything is locked in `pyproject.toml` / `uv.lock` — `uv sync --frozen`
(used in the Dockerfile) will refuse to run if the lock is out of date,
which is intentional: it forces the committed lock to actually match
what was tested.

Run notebooks via:

```bash
uv run jupyter notebook
```

## Serving

```bash
uv run uvicorn src.api:app --reload
```

- `GET /health` — liveness check.
- `POST /process-document` — upload a JPEG/PNG/TIFF, get back:
  ```json
  {
    "doc_type": "invoice",
    "confidence": 0.94,
    "vendor": "...",
    "date": "...",
    "total": "...",
    "address": "...",
    "ocr_text": "..."
  }
  ```
  Classification/extraction are skipped gracefully (fields come back
  `null`/`"unknown"`) if `models/artifacts/` hasn't been populated yet —
  the API doesn't hard-fail just because a model wasn't trained locally.

## Streamlit app

```bash
uv run streamlit run streamlit_app.py
```

Runs the same pipeline as the FastAPI endpoint, in-process (no separate
API call), against the public Hugging Face Hub checkpoints — so it works
standalone even without `models/artifacts/` populated locally. Includes
one-click sample documents, a confidence indicator, downloadable JSON
results, an interactive extraction panel (hover a field to highlight
where it was found on the document image), and an in-app benchmark
metrics panel sourced from the Results section above.

## Docker

```bash
docker build -t idp-financial-documents .
docker run -p 8000:8000 idp-financial-documents
```

`models/artifacts/` is baked into the image here for simplicity. If the
combined checkpoint size becomes a problem, loading weights from the
Hugging Face Hub at container startup instead is the standard
alternative — a tradeoff worth naming explicitly rather than hitting by
surprise later.

## Known limitations

- **BIO labels for field extraction are heuristic, not hand-annotated.**
  SROIE's `box/` files are line-level, and `entities/` gives clean field
  values with no position info — so per-word labels are derived by
  proportional character-offset splitting + fuzzy string matching (see
  `src/extract.py:label_document`). This is a source of label noise the
  model has to learn through, and some evaluation errors likely trace
  back to this step rather than the model itself.
- **RVL-CDIP is subsetted** (200/class, 4 classes) — results are not
  comparable to papers reporting full 400K-image, 16-class numbers.
- **No OCR-vs-model error attribution notebook yet** (`04_error_analysis`
  from the original plan) — worth building before treating field-level
  F1 numbers as the final word on model quality vs. OCR quality.
