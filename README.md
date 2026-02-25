# ACORD 25 Document Extraction System

A production-grade, fully open-source pipeline for extracting structured data from ACORD 25 certificate of insurance documents. Runs entirely locally — no external APIs, no paid services.

---

## Architecture Overview

```
Input (Image / PDF)
       │
       ▼
┌─────────────────────┐
│  1. Preprocessing   │  OpenCV + Pillow
│  - Normalization    │  Deskew, denoise, adaptive threshold
│  - Enhancement      │  CLAHE, contrast boost
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. OCR Layer       │  Tesseract or EasyOCR (swappable)
│  - Word detection   │  Returns text + bounding boxes + confidence
│  - Confidence score │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Layout Parser   │  Rule-based spatial clustering
│  - Line clustering  │  + Optional LayoutLMv3 (if fine-tuned)
│  - Region detection │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Field Mapper    │  Keyword + fuzzy + spatial proximity
│  - Header detection │  Handles OCR errors, layout variation
│  - Value extraction │  Date/currency regex patterns
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  5. Post-Processing │  Normalize dates, names, currencies
│  + Validation       │  Date range checks, field completeness
└────────┬────────────┘
         │
         ▼
    Structured JSON
    + Streamlit UI
```

---

## Hardware Requirements

### Minimum (CPU-only)
| Component | Requirement |
|-----------|-------------|
| CPU       | 4 cores, x86_64 |
| RAM       | 8 GB |
| Disk      | 2 GB free |
| OS        | Ubuntu 20.04+ / Debian 11+ |

### Recommended (with GPU)
| Component | Requirement |
|-----------|-------------|
| CPU       | 8+ cores |
| RAM       | 16 GB |
| GPU       | NVIDIA 8GB+ VRAM (CUDA 11.8+) |
| Disk      | 5 GB free |

### Runtime Expectations
| Mode              | Per-document time |
|-------------------|-------------------|
| CPU (Tesseract)   | 5–15 seconds      |
| CPU (EasyOCR)     | 30–60 seconds     |
| GPU (EasyOCR)     | 3–8 seconds       |
| CPU + LayoutLMv3  | 45–90 seconds     |

---

## Installation

### 1. System Dependencies

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0

# Verify Tesseract
tesseract --version
```

### 2. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Python Packages

```bash
# CPU-only PyTorch (smaller download, no CUDA needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# All other dependencies
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
python -c "import cv2; print(cv2.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
```

---

## Usage

### Streamlit Web UI

```bash
cd acord25_extractor
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Command Line

```bash
# Process a single image
python main.py path/to/acord25.jpg

# Process a PDF
python main.py path/to/acord25.pdf

# Specify output directory
python main.py path/to/acord25.jpg --output-dir data/outputs/
```

---

## Configuration

All settings are in `config/settings.py`:

```python
OCR_ENGINE = "tesseract"      # "tesseract" | "easyocr"
OCR_CONFIDENCE_THRESHOLD = 40  # Filter low-confidence words
FIELD_FUZZY_THRESHOLD = 0.75   # Field header matching sensitivity
PROXIMITY_RADIUS_PX = 80       # Spatial value search radius
```

---

## Project Structure

```
acord25_extractor/
├── app.py                    # Streamlit web interface
├── main.py                   # Pipeline orchestrator + CLI
├── requirements.txt
├── README.md
├── config/
│   └── settings.py           # All configuration
├── data/
│   ├── samples/              # Place test images here
│   └── outputs/              # JSON extraction results
├── models/
│   └── layout_model.py       # LayoutParser + LayoutLMv3 wrapper
├── ocr/
│   ├── ocr_engine.py         # Tesseract/EasyOCR abstraction
│   └── preprocessing.py      # Image preprocessing pipeline
├── extraction/
│   ├── field_mapper.py       # Field detection and extraction
│   ├── postprocessing.py     # Value normalization + JSON builder
│   └── validation.py         # Data quality validation
└── utils/
    └── helpers.py            # Logging, file I/O, evaluation metrics
```

---

## Output Format

```json
{
  "producer_name": "ABC Insurance Agency",
  "insured_name": "Acme Corporation",
  "policy_number": "GL-123456-01",
  "effective_date": "2024-01-01",
  "expiration_date": "2025-01-01",
  "coverages": [
    { "type": "General Liability", "limit": "$1,000,000" },
    { "type": "Automobile Liability", "limit": "$1,000,000" }
  ],
  "certificate_holder": "City Of Springfield",
  "additional_insured": "Yes",
  "subrogation_waiver": "Yes",
  "limits": {
    "general_liability": "$1,000,000",
    "auto_liability": "$1,000,000",
    "umbrella": "$5,000,000",
    "workers_comp": "$500,000"
  },
  "_metadata": {
    "overall_confidence": 0.82,
    "field_confidence": {
      "producer_name": 0.85,
      "insured_name": 0.85
    },
    "extraction_method": "hybrid_rule_spatial"
  },
  "_validation": {
    "is_valid": true,
    "score": 0.95,
    "issues": []
  }
}
```

---

## Layout Robustness Strategy

ACORD 25 forms vary across versions and insurers. The pipeline handles this through five layers:

1. **Spatial Clustering**: Words are grouped into lines by y-coordinate proximity (±12px tolerance), handling minor scan misalignment.

2. **Fuzzy Header Matching**: Field labels like "PRODUCER" that OCR reads as "PR0DUCER" are still matched via Levenshtein similarity (threshold configurable).

3. **Directional Proximity**: Values are searched *right* of inline labels and *below* block labels, matching ACORD 25's mixed layout.

4. **Regex Patterns**: Dates, policy numbers, and currency amounts are extracted via regex as a fallback when spatial matching fails.

5. **Confidence Thresholds**: Low-confidence OCR words are filtered before extraction to reduce noise.

---

## Fine-tuning LayoutLMv3 (Optional)

To achieve higher accuracy with a fine-tuned model:

1. Collect labeled ACORD 25 samples in FUNSD format
2. Set `LAYOUT_MODEL.use_layout_model = True` in `settings.py`
3. Set `LAYOUT_MODEL.model_path` to your checkpoint path

The `LayoutLMv3Extractor` class in `models/layout_model.py` handles loading and inference. Dataset format documentation is in the class docstring.

---

## Evaluation

```python
from utils.helpers import evaluate_extraction

metrics = evaluate_extraction(predicted_fields, ground_truth_fields)
print(metrics)
# {
#   "exact_match_accuracy": 0.78,
#   "partial_match_accuracy": 0.89,
#   "field_scores": { ... },
#   "fields_evaluated": 9
# }
```

---

## Troubleshooting

**Tesseract not found**: Ensure `tesseract-ocr` is installed and in PATH. Set `pytesseract.pytesseract.tesseract_cmd` if installed at non-standard path.

**PDF conversion fails**: Install `poppler-utils` (`sudo apt-get install poppler-utils`).

**Low confidence scores**: Try enabling EasyOCR for noisy/degraded scans. Adjust `OCR_CONFIDENCE_THRESHOLD` downward.

**Missing fields**: Lower `FIELD_FUZZY_THRESHOLD` (e.g., 0.60) to increase matching sensitivity at the cost of false positives.
