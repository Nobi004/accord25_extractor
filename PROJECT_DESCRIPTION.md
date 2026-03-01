# ACORD 25 Intelligent Document Extraction System

## Project Overview

The ACORD 25 Intelligent Document Extraction System is a fully open-source, production-grade pipeline that automates the extraction of structured data from ACORD 25 Certificates of Insurance. Built entirely on local, CPU-compatible open-source tools, it eliminates dependency on paid cloud APIs (AWS Textract, Google Vision, Azure Form Recognizer) while delivering reliable, validated JSON output from scanned or photographed insurance documents.

The system is designed to handle the messy reality of real-world documents: scan noise, misaligned text, stamped overlays, varying form versions, and inconsistent field positioning — all without requiring a perfectly formatted input.

---

## Problem Statement

Insurance operations teams, contractors, and compliance departments routinely receive ACORD 25 certificates from dozens or hundreds of vendors. Manually extracting key fields — policy numbers, coverage limits, effective dates, certificate holders — is time-consuming, error-prone, and unscalable. Existing automation solutions either rely on expensive cloud APIs with data privacy concerns, or fragile template-matching systems that break the moment a form version changes.

This project solves that problem with a robust, self-hosted extraction pipeline that understands document layout spatially, not just as raw text.

---

## Key Features

- **Zero external dependencies** — runs entirely on-premises, no API keys required
- **CPU-first design** — fully functional without a GPU; GPU support available for speed
- **Dual OCR support** — Tesseract (fast, lightweight) and EasyOCR (robust for degraded scans), swappable via config
- **Spatial layout understanding** — extracts values by understanding where they appear relative to field labels, not just searching raw text
- **Five-layer robustness** — exact match, fuzzy match, spatial proximity, regex fallback, and confidence filtering work in sequence to maximize extraction accuracy
- **Structured, validated output** — every result includes per-field confidence scores, date validation, logical consistency checks, and a machine-readable validation report
- **Optional LayoutLMv3 integration** — architecture is fine-tuning-ready for transformer-based extraction when labeled training data is available
- **Streamlit web interface** — clean browser-based UI with document viewer, annotated overlays, confidence indicators, and one-click JSON download

---

## Technical Architecture

The system is organized as a six-stage modular pipeline:

### Stage 1 — Document Preprocessing
Raw input images (JPG, PNG, or PDF pages) are prepared for OCR through a carefully ordered series of transformations. Contrast enhancement runs first to sharpen edges before deskew detection. CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes uneven lighting common in scanned documents. Non-Local Means denoising reduces scan grain without blurring text. Hough-line transform detects and corrects document rotation up to ±15 degrees. Finally, adaptive Gaussian thresholding binarizes the image to black-and-white, which maximizes OCR accuracy on documents with stamps or background texture.

### Stage 2 — OCR Layer
An abstract engine interface (`BaseOCREngine`) decouples the rest of the pipeline from any specific OCR tool. Both `TesseractEngine` and `EasyOCREngine` implement the same interface, returning a unified `OCRResult` object containing per-word text, bounding boxes, and confidence scores. The active engine is selected via configuration with no code changes required.

### Stage 3 — Layout Parsing
OCR words are clustered into lines using vertical proximity grouping (±12px tolerance), producing spatial `LayoutRegion` objects that preserve the two-dimensional structure of the document. This is the foundation of spatial reasoning: knowing that a word is on the same line or directly below another word is what makes accurate field-value pairing possible. An optional `LayoutLMv3Extractor` class is included for teams that want to fine-tune a transformer model on labeled ACORD 25 data.

### Stage 4 — Field Extraction
The `FieldMapper` runs a five-layer extraction strategy for each of the 13 target fields:

1. **Exact keyword match** — fastest path; checks if known field label text appears verbatim in a region
2. **Fuzzy match** — Levenshtein similarity catches OCR character substitutions (e.g., `PR0DUCER` → `PRODUCER`)
3. **Directional spatial search** — searches *right* of inline labels and *below* block labels, matching ACORD 25's mixed layout conventions
4. **Regex pattern extraction** — date, policy number, and currency patterns provide a fallback on the full OCR text
5. **Confidence scoring** — each match records how it was found and assigns a calibrated confidence score

### Stage 5 — Post-Processing and Validation
Raw extracted values are normalized: dates are converted to ISO 8601 format, currency strings are standardized, names are cleaned and title-cased. A dedicated validation layer then checks date parseability, logical date ordering (expiration must follow effective date), policy number format, name plausibility, and required field completeness. Each issue is categorized as an error or warning with field-level attribution.

### Stage 6 — Streamlit Interface
The web UI presents the document image alongside extracted results in real time. Users can toggle an annotation overlay that highlights extracted regions directly on the document. Per-field confidence scores are color-coded (green/amber/red). Validation issues are displayed inline. The full structured JSON result is available for download with one click. All UI configuration — OCR engine, confidence threshold, fuzzy threshold — is adjustable from the sidebar without restarting the application.

---

## Extracted Fields

| Field | Description |
|---|---|
| Producer Name | Insurance agency or broker |
| Insured Name | Named policyholder |
| Policy Number | Certificate or policy identifier |
| Effective Date | Coverage start date (ISO format) |
| Expiration Date | Coverage end date (ISO format) |
| Certificate Holder | Entity to whom the certificate is issued |
| Additional Insured | Whether additional insured endorsement applies |
| Subrogation Waiver | Whether waiver of subrogation applies |
| General Liability Limit | Per-occurrence and aggregate GL limits |
| Auto Liability Limit | Combined single limit for auto coverage |
| Umbrella / Excess Limit | Umbrella or excess liability limit |
| Workers Comp Limit | Employers liability and WC limits |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Image Processing | OpenCV, Pillow, NumPy |
| OCR (primary) | Tesseract via pytesseract |
| OCR (alternative) | EasyOCR |
| PDF Conversion | pdf2image + poppler |
| String Matching | rapidfuzz |
| Layout Model | HuggingFace Transformers (LayoutLMv3, optional) |
| Deep Learning | PyTorch (CPU-only supported) |
| Web Interface | Streamlit |
| Output Format | JSON with confidence metadata |

---

## Performance Characteristics

| Configuration | Estimated Time per Document |
|---|---|
| CPU + Tesseract (recommended baseline) | 5–15 seconds |
| CPU + EasyOCR | 30–60 seconds |
| GPU + EasyOCR | 3–8 seconds |
| CPU + LayoutLMv3 (fine-tuned) | 45–90 seconds |

Hardware minimum: 4-core CPU, 8 GB RAM, 2 GB disk. No GPU required.

---

## Design Decisions

**Why rule-based spatial extraction instead of pure deep learning?**
Fine-tuned transformer models require labeled training data that is rarely available for private document types. The rule-based spatial layer delivers strong baseline accuracy on day one, while the LayoutLMv3 integration provides a clear upgrade path once labeled data is collected.

**Why abstract the OCR engine?**
Tesseract and EasyOCR have different strengths. Tesseract is faster and more lightweight on CPU. EasyOCR handles rotated and degraded text better. The abstraction means teams can benchmark both on their own document corpus and switch without touching extraction logic.

**Why five extraction layers instead of one?**
No single strategy works for all ACORD 25 variants. Exact matching is fast but fragile. Fuzzy matching handles OCR noise but can produce false positives on short strings. Spatial proximity is accurate but requires a detected header. Regex is reliable for structured values like dates but useless for names. Layering them in order of precision gives the system both accuracy and resilience.

---

## Use Cases

- Insurance operations teams automating vendor certificate review
- Compliance departments maintaining up-to-date coverage records
- Brokers digitizing legacy paper certificate archives
- Construction and real estate firms managing subcontractor insurance requirements
- Any organization processing ACORD 25s at volume without cloud API budget

---

## Project Status

Core pipeline complete and functional. LayoutLMv3 fine-tuning pathway is architecture-ready, pending labeled training data. Evaluation harness included for measuring field-level accuracy against ground truth labels.
