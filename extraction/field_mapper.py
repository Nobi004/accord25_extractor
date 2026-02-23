import logging 
import re 
from dataclasses import dataclass 
from typing import Optional 

from models.layout_model import LayoutRegion, cluster_words_into_lines, find_words_near_bbox 
from ocr.ocr_engine import OCRResult,OCRWord

logger = logging.getLogger(__name__)

# ─── Field Keyword Definitions ────────────────────────────────────────────────
# Maps canonical field names to their expected label keywords on ACORD 25.
# Multiple variants handle different ACORD 25 form versions and OCR errors.

FIELD_KEYWORDS: dict[str, list[str]] = {
    "producer_name": [
        "producer", "agency", "producer name", "insurance producer",
        "agent", "broker"
    ],
    "insured_name": [
        "insured", "named insured", "insured name", "name of insured",
        "policyholder", "policy holder"
    ],
    "policy_number": [
        "policy number", "policy no", "policy#", "pol no", "policy num",
        "certificate number", "cert no"
    ],
    "effective_date": [
        "effective date", "eff date", "policy effective", "inception date",
        "from", "start date", "coverage from"
    ],
    "expiration_date": [
        "expiration date", "exp date", "expiry date", "policy expiration",
        "to", "end date", "coverage to", "expires"
    ],
    "certificate_holder": [
        "certificate holder", "cert holder", "holder", "additional insured",
        "certificate is issued to", "issued to"
    ],
    "general_liability_limit": [
        "general liability", "gl limit", "each occurrence", "aggregate",
        "general aggregate", "products", "completed operations"
    ],
    "auto_liability_limit": [
        "automobile liability", "auto liability", "combined single limit",
        "auto limit", "vehicle liability"
    ],
    "umbrella_limit": [
        "umbrella", "excess liability", "umbrella limit", "excess limit"
    ],
    "workers_comp_limit": [
        "workers compensation", "workers comp", "wc limit", "employers liability",
        "el limit"
    ],
    "additional_insured": [
        "additional insured", "addl insured", "additional named insured"
    ],
    "subrogation_waiver": [
        "waiver of subrogation", "subrogation waived", "subrogation",
        "wos"
    ],
}


# ─── Regex Patterns ────────────────────────────────────────────────────────────

DATE_PATTERNS = [
    r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",  # MM/DD/YYYY, MM-DD-YY
    r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b",     # YYYY-MM-DD
]

POLICY_NUMBER_PATTERNS = [
    r"\b([A-Z]{2,4}[-\s]?\d{6,12})\b",                 # Letters + digits
    r"\b(\d{8,14})\b",                                   # Pure digit strings
    r"\b([A-Z0-9]{3,5}-\d{4,8}-\d{2,6})\b",            # Hyphenated format
]

CURRENCY_PATTERNS = [
    r"\$[\d,]+(?:\.\d{2})?",                             # $1,000,000
    r"\d{1,3}(?:,\d{3})+(?:\.\d{2})?",                  # 1,000,000
]


