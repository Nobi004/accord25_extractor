
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = DATA_DIR / "outputs"

# OCR Settings
# Options: "tesseract" | "easyocr"
OCR_ENGINE = "tesseract"
# LSTM engine,assume uniform block of text
TESSERACT_CONFIG = r"--oem 3 --psm 6"  
# Minimum confidence score (0-100) to keep OCR result
OCR_CONFIDENCE_THRESHOLD = 40 

# Preprocessing settings
PREPROCESSING = {
    "target_dpi": 300,
    "denoise": True,
    "deskew": True,
    "adaptive_threshold": True,
    "contrast_enhance": True,
}

# Layout model settings
LAYOUT_MODEL = {
    # Set True if LayoutLMv3 fine-tuned model available
    "use_layout_model": False, 
    "model_path": None,  # Path to fine-tuned model checkpoint
    "fallback_to_rules": True,  # Always fall back to rule-based if model fails
}

# Field extraction settings
# Minimum similarity score for fuzzy header matching
FIELD_FUZZY_THRESHOLD = 0.75 
# Pixels radius to search for value near field header
PROXIMITY_RADIUS_PX = 80       

# ACORD 25 target fields
ACORD25_FIELDS = [
    "producer_name",
    "insured_name",
    "policy_number",
    "effective_date",
    "expiration_date",
    "coverages",
    "certificate_holder",
    "additional_insured",
    "subrogation_waiver",
    "general_liability_limit",
    "auto_liability_limit",
    "umbrella_limit",
    "workers_comp_limit",
]

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
