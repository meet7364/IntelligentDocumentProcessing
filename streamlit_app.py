"""Streamlit entry point: upload a scanned invoice/receipt, run the real
OCR -> classify -> extract pipeline in-process, render the result.

Models are loaded once per process (per repo_id) via `st.cache_resource` from
public Hugging Face Hub repos -- never from a Colab/Drive-only local path --
so this app runs standalone in a fresh environment (Streamlit Community
Cloud) with no Colab/Drive access at inference time.

Full UI state coverage (empty/loading/error/populated/partial/overflow/
long-text/zero-one-many, page framing) is wired here on top of the proven
happy-path flow. `run_pipeline()` is the testable seam: it contains
all pipeline logic and is directly callable (see scripts/verify_error_states.py)
without a running Streamlit server. UI glue (spinner/error/warning rendering)
lives outside it, in the module body below.
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import tempfile

import psutil
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.classify import DistilBertTextClassifier
from src.extract import get_extractor
from src.ocr import run_tesseract, words_and_boxes_from_image
from src.postprocess import predict_fields_with_boxes

CLASSIFIER_REPO = os.environ.get("HF_CLASSIFIER_REPO", "meet7364/idp-distilbert-classifier")
EXTRACTOR_REPO = os.environ.get("HF_EXTRACTOR_REPO", "meet7364/idp-layoutlm-extractor")

# 0.5 is the natural midpoint for a binary-ish confidence signal.
LOW_CONFIDENCE_THRESHOLD = 0.5

# Sample-document paths for the one-click "try it without a file" buttons
# Both are fixed, developer-controlled paths -- never derived
# from user input -- so there is no path-traversal surface.
TYPICAL_SAMPLE_PATH = "assets/samples/typical-receipt.jpg"
UNFAMILIAR_SAMPLE_PATH = "scripts/fixtures/Invoice-REMI8GUW-0001.png"

# Field-identity palette for the extraction bbox overlay --
# exact RGB values from the UI-SPEC's "Field-identity palette" table.
# Keys match src.postprocess.field_boxes_for_display's return-dict keys
# exactly, since both originate from src.extract.FIELDS.
FIELD_OVERLAY_COLORS = {
    "company": (31, 119, 180),
    "date": (44, 160, 44),
    "address": (148, 103, 189),
    "total": (255, 140, 0),
}

# Hex form of FIELD_OVERLAY_COLORS, for the CSS field-value chips and the
# SVG bbox overlay below -- same identity colors used in both places, so a
# field's chip border and its highlighted box in the image can never drift
# apart. Kept as four distinct hues (not collapsed to the single interactive
# accent) so the four fields stay visually distinguishable at a glance --
# a deliberate call, since the brief left the exact per-field
# palette open ("Vendor = teal box... Total = same" reads as "same
# treatment", not literally one shared hue).
FIELD_OVERLAY_COLORS_HEX = {k: "#%02x%02x%02x" % v for k, v in FIELD_OVERLAY_COLORS.items()}

# Precision-tool palette -- exact hex values, no default Streamlit colors.
COLOR_BACKGROUND = "#FAFAF7"
COLOR_SURFACE = "#FFFFFF"
COLOR_BORDER = "#E4E2DB"
COLOR_INK = "#1C2321"
COLOR_MUTED = "#6B7268"
COLOR_ACCENT = "#2B5D6E"
COLOR_CONFIDENCE_HIGH = "#3F7D5C"
COLOR_CONFIDENCE_WARN = "#B5652B"

PAGE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
    color: {COLOR_INK};
}}

h1 {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}}

/* Full-bleed working area, not a centered narrow column (the layout spec). */
.block-container {{
    max-width: 1400px;
    padding-left: 3rem;
    padding-right: 3rem;
    padding-top: 2rem;
}}

/* Header: title left, GitHub link right, thin rule below -- no big
   centered hero block. */
.app-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
    flex-wrap: wrap;
}}
.app-header a {{
    color: {COLOR_ACCENT};
    font-size: 0.9rem;
    text-decoration: none;
    white-space: nowrap;
}}
.app-header a:hover {{
    text-decoration: underline;
}}
.app-subtitle {{
    color: {COLOR_MUTED};
    margin-top: 0.25rem;
}}
.app-rule {{
    border: none;
    border-top: 1px solid {COLOR_BORDER};
    margin: 1rem 0 1.5rem 0;
}}

/* Buttons: flat, 1px border, no gradients, sharp-ish radius -- a
   precision tool, not a consumer app. */
.stButton > button, .stDownloadButton > button {{
    background: {COLOR_SURFACE};
    color: {COLOR_INK};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    box-shadow: none;
    font-weight: 500;
    transition: background 120ms ease, border-color 120ms ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background: #F3F2ED;
    border-color: {COLOR_ACCENT};
    color: {COLOR_ACCENT};
}}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
    outline: 2px solid {COLOR_ACCENT};
    outline-offset: 1px;
}}

/* Confidence pill: colored text + border on a low-opacity tint, never a
   solid fill (the visual spec). */
.confidence-pill {{
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 5px;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid;
}}

/* Numbers/metrics always monospace, tabular-nums, right-aligned. */
table, .stTable table {{
    font-variant-numeric: tabular-nums;
}}
table td, .stTable table td {{
    font-family: 'IBM Plex Mono', monospace;
}}
</style>
"""

# Real, reproduced benchmark results -- every value copied
# verbatim (as a string, not a reformatted float, to stay exact-substring
# verifiable against README.md by scripts/verify_benchmarks.py) from
# README's "Results (real, reproduced by rerunning every notebook
# end-to-end)" section. Never re-derived or invented at runtime.
OCR_BENCHMARK = [
    {"engine": "Tesseract", "mean_cer": "0.436", "mean_time": "2.73s"},
    {"engine": "EasyOCR", "mean_cer": "0.402", "mean_time": "29.07s"},
]
CLASSIFICATION_BENCHMARK = [
    {"model": "TF-IDF + LogReg", "accuracy": "0.781", "macro_f1": "0.784"},
    {"model": "DistilBERT (4 epoch, early stopping)", "accuracy": "0.787", "macro_f1": "0.790"},
]
EXTRACTION_BENCHMARK = [
    {"field": "company", "precision": "0.973", "recall": "0.988", "f1": "0.980", "exact_match": "0.882"},
    {"field": "date", "precision": "0.981", "recall": "0.983", "f1": "0.982", "exact_match": "0.948"},
    {"field": "address", "precision": "0.993", "recall": "0.996", "f1": "0.995", "exact_match": "0.709"},
    {"field": "total", "precision": "0.817", "recall": "0.850", "f1": "0.833", "exact_match": "0.732"},
]


def confidence_tier(confidence: float | None) -> tuple[str, str]:
    """Map a confidence value to (label, hex color) using two
    named confidence colors -- confidence-high (sage) for a strong result,
    confidence-low/warning (amber) for anything below the "high" line or
    absent. Reuses LOW_CONFIDENCE_THRESHOLD for the Medium/Low label split;
    the spec does not define a distinct third hue, so Medium and Low
    share the warning amber (only the label differs).
    """
    if confidence is None:
        return "Unknown", COLOR_MUTED
    if confidence >= 0.8:
        return "High confidence", COLOR_CONFIDENCE_HIGH
    if confidence >= LOW_CONFIDENCE_THRESHOLD:
        return "Medium confidence", COLOR_CONFIDENCE_WARN
    return "Low confidence", COLOR_CONFIDENCE_WARN


def _confidence_pill_html(label: str, color_hex: str) -> str:
    """Render the confidence label as a pill: colored text + border on a
    12%-opacity tint of the same color, never a solid fill. `label` is a
    fixed internal string (not user input), but is
    still escaped defensively.
    """
    return (
        f'<span class="confidence-pill" style="color:{color_hex};border-color:{color_hex};'
        f'background:{color_hex}1F;">{html.escape(label)}</span>'
    )


def download_filename(doc_type: str) -> str:
    """Map a doc_type to its JSON download filename, falling back to
    'result.json' when doc_type is 'unknown'.
    """
    if doc_type == "unknown":
        return "result.json"
    return f"{doc_type}_result.json"


@st.cache_resource
def _cached_classifier(repo_id: str) -> DistilBertTextClassifier:
    return DistilBertTextClassifier(repo_id)


@st.cache_resource
def _cached_extractor(repo_id: str):
    return get_extractor(repo_id)


def run_pipeline(tmp_path: str) -> dict:
    """Run OCR -> classify -> extract on an already-saved image file.

    Returns a dict with keys `doc_type, confidence, vendor, date, total,
    address, ocr_text, low_confidence, words, boxes, field_boxes`.
    `low_confidence` is True iff `confidence is None or confidence <
    LOW_CONFIDENCE_THRESHOLD or doc_type == 'unknown'`. `words`/`boxes` are
    the raw OCR word/bbox output (previously discarded); `field_boxes` maps
    each field to the exact model-predicted word boxes behind its displayed
    value from `src.postprocess.field_boxes_for_display`.

    Raises whatever `PIL.Image.open`/OCR/classify/extract raise on a corrupt
    or unreadable file -- this function never swallows an exception into a
    partial result. The caller is responsible for catching that and
    rendering `st.error`, never letting it propagate as a raw traceback.
    """
    # PIL.Image.open lazily reads only the header; force a real decode here
    # so corrupt/non-image bytes raise now, in this function, rather than
    # surfacing later inside Tesseract or silently producing garbage.
    Image.open(tmp_path).load()

    ocr_text = run_tesseract(tmp_path)

    # Mirror src/api.py's graceful-degradation default: skip classification
    # entirely on empty/whitespace-only OCR text rather than feeding the
    # classifier a blank string and trusting whatever label it guesses.
    doc_type, confidence = "unknown", None
    if ocr_text.strip():
        doc_type, confidence = _cached_classifier(CLASSIFIER_REPO).predict_with_confidence([ocr_text])[0]

    words, boxes = words_and_boxes_from_image(tmp_path)
    if words:
        fields, field_boxes = predict_fields_with_boxes(words, boxes, extractor=_cached_extractor(EXTRACTOR_REPO))
    else:
        fields, field_boxes = {}, {}

    low_confidence = confidence is None or confidence < LOW_CONFIDENCE_THRESHOLD or doc_type == "unknown"

    return {
        "doc_type": doc_type,
        "confidence": confidence,
        "vendor": fields.get("company"),
        "date": fields.get("date"),
        "total": fields.get("total"),
        "address": fields.get("address"),
        "ocr_text": ocr_text,
        "low_confidence": low_confidence,
        "words": words,
        "boxes": boxes,
        "field_boxes": field_boxes,
    }


def _image_to_data_uri(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data URI, for embedding directly
    in the self-contained HTML component below (no temp file, no separate
    static-asset route needed)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _box_to_pixels(box: list[int], img_w: int, img_h: int) -> list[int]:
    """Convert a 0-1000-normalized LayoutLM box to pixel coordinates in the
    source image's own coordinate space (matches the SVG's viewBox below,
    so rects line up with the displayed image regardless of render size)."""
    return [round(coord / 1000 * (img_w if i % 2 == 0 else img_h)) for i, coord in enumerate(box)]


# Ticket-row render order -- fixed. Each entry is (display label, the
# FIELD_OVERLAY_COLORS/field_boxes key, the run_pipeline() result dict key)
# -- src.postprocess uses "company" while run_pipeline()'s result dict
# exposes that same field as "vendor", so both keys are carried explicitly
# rather than assumed identical.
_FIELD_ROWS = [
    ("Vendor", "company", "vendor"),
    ("Date", "date", "date"),
    ("Address", "address", "address"),
    ("Total", "total", "total"),
]


def render_extraction_panel(
    source_image: Image.Image,
    field_boxes: dict[str, list[list[int]] | None],
    fields: dict[str, str | None],
) -> None:
    """Render the signature "traced connection" panel: the source document
    on the left with a thin colored bounding
    box around each extracted field's region, and a vertical field "ticket"
    list on the right. Hovering a ticket row highlights its matching
    box(es) on the image; each row also fades/slides in with a 150ms
    stagger on load.

    Built as a single self-contained HTML component (not native
    `st.columns`) so the hover JS can query both the image overlay and the
    ticket rows in one DOM -- two separate Streamlit elements can't easily
    cross-wire hover state. Every dynamic string (field values, the image
    bytes) is either HTML-escaped or base64-encoded before embedding, since
    field values originate from OCR of an untrusted uploaded image.
    """
    img_w, img_h = source_image.size
    data_uri = _image_to_data_uri(source_image)

    svg_rects: list[str] = []
    ticket_rows: list[str] = []
    for i, (label, color_key, result_key) in enumerate(_FIELD_ROWS):
        color = FIELD_OVERLAY_COLORS_HEX[color_key]
        boxes = field_boxes.get(color_key)
        has_match = bool(boxes)
        if has_match:
            for box in boxes:
                x0, y0, x1, y1 = _box_to_pixels(box, img_w, img_h)
                svg_rects.append(
                    f'<rect class="field-box" data-field="{color_key}" x="{x0}" y="{y0}" '
                    f'width="{max(x1 - x0, 1)}" height="{max(y1 - y0, 1)}" '
                    f'style="stroke:{color};" />'
                )
        value = fields.get(result_key)
        display_value = html.escape(value) if value else "Not detected"
        row_class = "field-row" + ("" if has_match else " no-match")
        ticket_rows.append(
            f'<div class="field-row-wrap" style="animation-delay:{i * 150}ms;">'
            f'<div class="{row_class}" data-field="{color_key}" style="--row-color:{color};">'
            f'<span class="row-label">{html.escape(label)}</span>'
            f'<span class="row-value">{display_value}</span>'
            "</div></div>"
        )

    # Fixed display width for the image column; height follows the source
    # image's real aspect ratio so boxes never distort. The ticket column
    # is a plain flex list -- its height is whatever four rows need.
    display_w = 640
    display_h = round(img_h / img_w * display_w) if img_w else 480
    component_height = max(display_h, len(_FIELD_ROWS) * 92 + 40) + 32

    page = f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: 'Inter', sans-serif;
    color: {COLOR_INK};
    background: transparent;
  }}
  .panel {{
    display: flex;
    gap: 1.5rem;
    align-items: flex-start;
  }}
  .image-col {{
    position: relative;
    width: {display_w}px;
    flex-shrink: 0;
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    overflow: hidden;
    background: {COLOR_SURFACE};
  }}
  .image-col img {{ display: block; width: 100%; height: auto; }}
  .image-col svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; }}
  .field-box {{
    fill: none;
    stroke-width: 2;
    opacity: 0.55;
    transition: opacity 150ms ease, stroke-width 150ms ease;
  }}
  .field-box.active {{ stroke-width: 3.5; opacity: 1; }}
  .field-box.dimmed {{ opacity: 0.12; }}

  .ticket-col {{ flex: 1; min-width: 220px; }}
  .field-row-wrap {{
    opacity: 0;
    animation: rowIn 380ms ease-out forwards;
  }}
  @keyframes rowIn {{
    from {{ opacity: 0; transform: translateX(6px); }}
    to {{ opacity: 1; transform: translateX(0); }}
  }}
  .field-row {{
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.5rem;
    background: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-left: 3px solid var(--row-color, #999);
    border-radius: 5px;
    cursor: default;
    transition: background 150ms ease, border-color 150ms ease;
  }}
  .field-row.hovered {{ background: #F6F5F1; border-color: var(--row-color, {COLOR_BORDER}); }}
  .row-label {{
    font-size: 0.75rem;
    font-weight: 500;
    color: {COLOR_MUTED};
  }}
  .row-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.92rem;
    color: {COLOR_INK};
    word-break: break-word;
  }}
</style></head>
<body>
  <div class="panel">
    <div class="image-col" style="height:{display_h}px;">
      <img src="{data_uri}" alt="Uploaded document" />
      <svg viewBox="0 0 {img_w} {img_h}" preserveAspectRatio="none">
        {"".join(svg_rects)}
      </svg>
    </div>
    <div class="ticket-col">
      {"".join(ticket_rows)}
    </div>
  </div>
  <script>
    const rows = document.querySelectorAll('.field-row');
    rows.forEach(row => {{
      const field = row.getAttribute('data-field');
      const boxes = document.querySelectorAll('.field-box[data-field="' + field + '"]');
      const others = document.querySelectorAll('.field-box:not([data-field="' + field + '"])');
      row.addEventListener('mouseenter', () => {{
        row.classList.add('hovered');
        boxes.forEach(b => b.classList.add('active'));
        others.forEach(b => b.classList.add('dimmed'));
      }});
      row.addEventListener('mouseleave', () => {{
        row.classList.remove('hovered');
        boxes.forEach(b => b.classList.remove('active'));
        others.forEach(b => b.classList.remove('dimmed'));
      }});
    }});
  </script>
</body></html>
"""
    components.html(page, height=component_height, scrolling=False)


def current_rss_mb() -> float:
    """Return the current process's real resident memory in MB.

    Must be called AFTER run_pipeline() has completed, so the reading
    captures peak memory (OCR + both models loaded + inference tensors
    all still resident at that point), not idle startup memory.
    """
    return psutil.Process().memory_info().rss / (1024**2)


def main() -> None:
    st.set_page_config(page_title="Financial Document Extractor -- Live Demo", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="app-header">'
        "<h1>Financial Document Extractor &mdash; Live Demo</h1>"
        '<a href="https://github.com/meet7364/idp-financial-documents" target="_blank">'
        "View source &amp; benchmarks on GitHub</a>"
        "</div>"
        '<div class="app-subtitle">Upload a scanned invoice or receipt to see real OCR, '
        "document classification, and field extraction &mdash; no local setup required.</div>"
        '<hr class="app-rule" />',
        unsafe_allow_html=True,
    )

    with st.expander("How this works"):
        st.markdown(
            "1. **OCR** (Tesseract) reads the raw text and word positions from your image. "
            "2. A fine-tuned **DistilBERT** classifier predicts the document type from that text. "
            "3. A fine-tuned **LayoutLM** model extracts vendor, date, address, and total using both "
            "the text and its position on the page. Real accuracy numbers for each step are in the "
            "Benchmark metrics panel below."
        )

    with st.expander("Benchmark metrics (real, reproduced results)"):
        st.markdown(
            "These are the actual evaluation results from this project's notebooks, reproduced "
            "end-to-end -- not estimates. Full write-up: "
            "[GitHub README](https://github.com/meet7364/idp-financial-documents"
            "#results-real-reproduced-by-rerunning-every-notebook-end-to-end)."
        )
        st.write("**OCR baseline**")
        st.table(OCR_BENCHMARK)
        st.write("**Document classification**")
        st.table(CLASSIFICATION_BENCHMARK)
        st.write("**Field extraction**")
        st.table(EXTRACTION_BENCHMARK)

    st.caption("No document handy? Try one of these -- same pipeline, same real results.")
    sample_col1, sample_col2 = st.columns(2)
    sample_clicked = False
    if sample_col1.button("Try a sample: Typical receipt"):
        st.session_state["sample_path"] = TYPICAL_SAMPLE_PATH
        sample_clicked = True
    if sample_col2.button("Try a sample: Unfamiliar document"):
        st.session_state["sample_path"] = UNFAMILIAR_SAMPLE_PATH
        sample_clicked = True

    uploaded = st.file_uploader(
        "Upload a scanned invoice or receipt (JPEG, PNG, or TIFF)",
        type=["jpg", "jpeg", "png", "tiff"],
    )

    # Source-selection priority: a fresh sample click always wins over a file
    # already sitting in the uploader from a prior interaction; a genuinely
    # new upload always clears a previously-selected sample; a sample chosen
    # on an earlier, unrelated rerun (e.g. expanding another section) is
    # still honored via st.session_state persistence.
    active_path = None
    is_upload = False
    if sample_clicked:
        active_path = st.session_state["sample_path"]
        is_upload = False
    elif uploaded:
        st.session_state.pop("sample_path", None)
        is_upload = True
    elif st.session_state.get("sample_path"):
        active_path = st.session_state["sample_path"]
        is_upload = False

    if not is_upload and active_path is None:
        st.subheader("Upload a document to see it extracted in real time")
        st.write(
            "Drag and drop or browse for a JPEG, PNG, or TIFF invoice or receipt above -- "
            "results appear here once processing finishes."
        )
        return

    tmp_path = None
    try:
        if is_upload:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                Image.open(uploaded).convert("RGB").save(tmp, format="JPEG")
        else:
            tmp_path = active_path

        with st.spinner(
            "Running OCR, classification, and extraction... First run after the app has "
            "been idle may take up to a minute while models load -- please wait."
        ):
            result = run_pipeline(tmp_path)
        # Force a full in-memory decode now, while tmp_path is guaranteed to
        # still exist, so the source image bytes remain available for the
        # overlay below even after the finally block below may unlink it.
        source_image = Image.open(tmp_path).convert("RGB")
    except Exception as exc:
        # Log the real exception server-side only -- never interpolate it
        # into the user-facing st.error copy .
        print(f"streamlit_app: pipeline error processing upload: {exc!r}")
        st.error(
            "Something went wrong while processing this document. This is a real error, "
            "not a fabricated result. Try a different file, or check back if the issue persists."
        )
        return
    finally:
        # Only unlink uploader-created temp files -- the persistent sample
        # assets (assets/samples/, scripts/fixtures/) must never be deleted.
        if is_upload and tmp_path is not None:
            os.unlink(tmp_path)

    if st.query_params.get("debug") == "1":
        with st.expander("Debug: memory usage (hidden by default)"):
            st.write(f"Peak process RSS after this upload: {current_rss_mb():.1f} MB")

    if result["low_confidence"]:
        st.warning(
            "Low-confidence result -- this document may be unfamiliar to the model. "
            "Fields below may be incomplete; check the raw OCR text to see what was actually read."
        )

    st.subheader("Where each field was found")

    doc_type_display = html.escape(result["doc_type"])
    label, color_hex = confidence_tier(result["confidence"])
    st.markdown(
        f'<div style="margin-bottom:0.5rem;">'
        f'<strong>Document type:</strong> <span style="font-family:\'IBM Plex Mono\',monospace;">'
        f"{doc_type_display}</span> &nbsp; {_confidence_pill_html(label, color_hex)}</div>",
        unsafe_allow_html=True,
    )
    if result["confidence"] is not None:
        st.progress(result["confidence"])

    if not result["words"]:
        st.caption("Bounding box overlay unavailable -- no word positions were detected for this document.")
        for row_label, _color_key, row_result_key in _FIELD_ROWS:
            value = result[row_result_key]
            st.write(f"**{row_label}:** {value if value else 'Not detected'}")
    else:
        render_extraction_panel(source_image, result["field_boxes"], result)

    st.download_button(
        "Download results as JSON",
        data=json.dumps(result, indent=2),
        file_name=download_filename(result["doc_type"]),
        mime="application/json",
    )

    with st.expander("Raw OCR Text"):
        st.subheader("Raw OCR Text")
        st.text_area("OCR output", value=result["ocr_text"], height=300, disabled=True, label_visibility="collapsed")


main()
