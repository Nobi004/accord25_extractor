"""
ACORD 25 Extractor — Streamlit Web Interface
----------------------------------------------
Clean, minimal UI for document upload and field extraction display.
"""

import json
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

<<<<<<< Updated upstream
# ─── Page Config ──────────────────────────────
=======
# ─── Page Config ──────────────────────────────────────────────────────────────
>>>>>>> Stashed changes
st.set_page_config(
    page_title="ACORD 25 Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

<<<<<<< Updated upstream
# ─── Custom CSS ────────────────────────────────────
=======
# ─── Custom CSS ───────────────────────────────────────────────────────────────
>>>>>>> Stashed changes
st.markdown("""
<style>
    .field-card {
        background: #f8f9fa;
        border-left: 4px solid #2563eb;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px;
    }
    .field-name { font-weight: 600; color: #374151; font-size: 0.85rem; }
    .field-value { color: #111827; font-size: 1rem; margin-top: 2px; }
    .confidence-high { color: #16a34a; }
    .confidence-med  { color: #ca8a04; }
    .confidence-low  { color: #dc2626; }
    .validation-error   { background: #fef2f2; border-left-color: #dc2626; }
    .validation-warning { background: #fffbeb; border-left-color: #f59e0b; }
</style>
""", unsafe_allow_html=True)


<<<<<<< Updated upstream
# ─── Helper Functions ───────────
=======
# ─── Helper Functions ──────────────────────────────────────────────────────────
>>>>>>> Stashed changes

def confidence_color_class(conf: float) -> str:
    if conf >= 0.8:
        return "confidence-high"
    elif conf >= 0.6:
        return "confidence-med"
    return "confidence-low"


def format_confidence(conf: float) -> str:
    pct = conf * 100
    color = confidence_color_class(conf)
    return f'<span class="{color}">{pct:.0f}%</span>'


@st.cache_resource(show_spinner="Loading extraction pipeline...")
def load_pipeline():
    """Load and cache the extraction pipeline (expensive initialization)."""
    from main import ACORD25Pipeline
    return ACORD25Pipeline()


def process_uploaded_file(uploaded_file, pipeline) -> dict:
    """Process an uploaded file and return extraction results."""
    file_bytes = uploaded_file.read()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        # Save temp PDF and process
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            results = pipeline.process_pdf(tmp_path)
            return results[0] if results else {}
        finally:
            os.unlink(tmp_path)
    else:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        return pipeline.process_image(image)


<<<<<<< Updated upstream
# ─── Sidebar ────────────────────────────────────────────────────
=======
# ─── Sidebar ──────────────────────────────────────────────────────────────────
>>>>>>> Stashed changes

with st.sidebar:
    st.title("⚙️ Settings")

    ocr_engine = st.selectbox(
        "OCR Engine",
        ["tesseract", "easyocr"],
        index=0,
        help="Tesseract is faster on CPU. EasyOCR handles rotated/noisy text better.",
    )

    confidence_threshold = st.slider(
        "OCR Confidence Threshold",
        min_value=0, max_value=100, value=40, step=5,
        help="Words below this confidence are filtered out.",
    )

    fuzzy_threshold = st.slider(
        "Fuzzy Match Threshold",
        min_value=0.5, max_value=1.0, value=0.75, step=0.05,
        help="Minimum similarity to match field headers.",
    )

    show_annotated = st.checkbox("Show extraction overlay", value=True)
    show_raw_json = st.checkbox("Show raw JSON", value=False)
    show_ocr_text = st.checkbox("Show raw OCR text", value=False)

    st.divider()
    st.markdown("**Hardware Info**")
    try:
        import torch
        device = "GPU ✅" if torch.cuda.is_available() else "CPU only"
    except ImportError:
        device = "CPU only"
    st.caption(f"Inference device: {device}")


# ─── Main UI ─────────────────────────────────────────────────────────────────

st.title("📄 ACORD 25 Certificate Extractor")
st.markdown("Upload a scanned ACORD 25 certificate (JPG, PNG, or PDF) to extract structured data.")

uploaded_file = st.file_uploader(
    "Upload document",
    type=["jpg", "jpeg", "png", "pdf"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("👆 Upload an ACORD 25 document to get started.")
    st.stop()

<<<<<<< Updated upstream
# ─── Process Document ─────────────────────────────────────────
=======
# ─── Process Document ─────────────────────────────────────────────────────────
>>>>>>> Stashed changes

with st.spinner("Running extraction pipeline..."):
    try:
        pipeline = load_pipeline()
        result = process_uploaded_file(uploaded_file, pipeline)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.exception(e)
        st.stop()

extracted = result.get("extracted", {})
validation = result.get("validation")
field_matches = result.get("field_matches", {})
ocr_result = result.get("ocr_result")
annotated_image = result.get("annotated_image")

<<<<<<< Updated upstream
# ─── Layout: Two Columns ────────────────────────
=======
# ─── Layout: Two Columns ──────────────────────────────────────────────────────
>>>>>>> Stashed changes

col_doc, col_results = st.columns([1, 1], gap="large")

# Left column: document viewer
with col_doc:
    st.subheader("📋 Document")

    # Show annotated or original
    display_image = annotated_image if (show_annotated and annotated_image) else None

    if display_image is None:
        # Reconstruct original from upload
        uploaded_file.seek(0)
        try:
            display_image = Image.open(BytesIO(uploaded_file.read()))
        except Exception:
            pass

    if display_image:
        st.image(display_image, use_container_width=True)
    else:
        st.warning("Could not display document image.")

    if show_ocr_text and ocr_result:
        with st.expander("Raw OCR Text"):
            st.text(ocr_result.full_text)

# Right column: extraction results
with col_results:
    st.subheader("🔍 Extracted Fields")

    # Overall confidence
    meta = extracted.get("_metadata", {})
    overall_conf = meta.get("overall_confidence", 0.0)
    field_confidence = meta.get("field_confidence", {})

    conf_col1, conf_col2 = st.columns(2)
    with conf_col1:
        st.metric("Overall Confidence", f"{overall_conf * 100:.0f}%")
    with conf_col2:
        if validation:
            status = "✅ Valid" if validation.is_valid else "⚠️ Issues Found"
            st.metric("Validation", status)

    st.divider()

    # ── Key Fields Table ──
    st.markdown("**Core Fields**")

    key_fields = [
        ("Producer Name", "producer_name"),
        ("Insured Name", "insured_name"),
        ("Policy Number", "policy_number"),
        ("Effective Date", "effective_date"),
        ("Expiration Date", "expiration_date"),
        ("Certificate Holder", "certificate_holder"),
        ("Additional Insured", "additional_insured"),
        ("Subrogation Waiver", "subrogation_waiver"),
    ]

    for label, key in key_fields:
        value = extracted.get(key, "")
        conf = field_confidence.get(key, 0.0)
        conf_html = format_confidence(conf) if value else ""

        st.markdown(f"""
        <div class="field-card">
            <div class="field-name">{label} {conf_html}</div>
            <div class="field-value">{value or "<em style='color:#9ca3af'>Not found</em>"}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Coverages ──
    coverages = extracted.get("coverages", [])
    if coverages:
        st.markdown("**Coverage Limits**")
        import pandas as pd
        cov_df = pd.DataFrame(coverages)
        st.dataframe(cov_df, use_container_width=True, hide_index=True)

    # ── Validation Issues ──
    if validation and validation.issues:
        st.markdown("**Validation**")
        for issue in validation.issues:
            cls = "validation-error" if issue.severity == "error" else "validation-warning"
            icon = "🔴" if issue.severity == "error" else "🟡"
            st.markdown(f"""
            <div class="field-card {cls}">
                {icon} <strong>{issue.field}</strong>: {issue.message}
            </div>
            """, unsafe_allow_html=True)

    # ── Raw JSON ──
    if show_raw_json:
        with st.expander("Raw JSON Output"):
            # Remove internal metadata for cleaner display
            display_json = {k: v for k, v in extracted.items()
                            if not k.startswith("_")}
            st.json(display_json)

<<<<<<< Updated upstream
# ─── Download Button ──────────────────────────────────────────
=======
# ─── Download Button ──────────────────────────────────────────────────────────
>>>>>>> Stashed changes

output_json = json.dumps(extracted, indent=2, ensure_ascii=False)
st.download_button(
    label="⬇️ Download JSON",
    data=output_json,
    file_name=f"{Path(uploaded_file.name).stem}_extracted.json",
    mime="application/json",
)
