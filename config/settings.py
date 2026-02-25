from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = DATA_DIR / "outputs"

OCR_ENGINE = "tesseract"
TESSERACT_CONFIG = r"--oem 3 --psm 6"
OCR_CONFIDENCE_THRESHOLD = 40

PREPROCESSING = {
    "target_dpi": 300,
    "denoise": True,
    "deskew": True,
    "adaptive_threshold": True,
    "contrast_enhance": True,
}

LAYOUT_MODEL = {
    "use_layout_model": False,
    "model_path": None,
    "fallback_to_rules": True,
}

FIELD_FUZZY_THRESHOLD = 0.75
PROXIMITY_RADIUS_PX = 80
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

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
