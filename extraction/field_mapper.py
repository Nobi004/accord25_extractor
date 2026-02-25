import logging
import re
from dataclasses import dataclass
from typing import Optional

from models.layout_model import LayoutRegion, find_words_near_bbox
from ocr.ocr_engine import OCRResult, OCRWord

logger = logging.getLogger(__name__)


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

DATE_PATTERNS = [
    r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
    r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b",
]

POLICY_NUMBER_PATTERNS = [
    r"\b([A-Z]{2,4}[-\s]?\d{6,12})\b",
    r"\b(\d{8,14})\b",
    r"\b([A-Z0-9]{3,5}-\d{4,8}-\d{2,6})\b",
]

CURRENCY_PATTERNS = [
    r"\$[\d,]+(?:\.\d{2})?",
    r"\d{1,3}(?:,\d{3})+(?:\.\d{2})?",
]

def normalize_text(text: str) -> str:
    """Normalize text for matching, handling OCR substitutions."""
    text = text.lower().strip()
    normalized = re.sub(r"\s+", " ", text)
    return normalized


@dataclass
class FieldMatch:
    """Result of a field extraction attempt."""
    field_name: str
    value: str
    confidence: float
    match_method: str
    bbox: Optional[tuple] = None


def fuzzy_match_score(s1: str, s2: str) -> float:
    """Compute similarity between two strings using Levenshtein distance."""
    try:
        from rapidfuzz import fuzz
        return fuzz.partial_ratio(s1.lower(), s2.lower()) / 100.0
    except ImportError:
        s1, s2 = s1.lower(), s2.lower()
        if s1 in s2 or s2 in s1:
            return 0.9
        common = sum(1 for c in s1 if c in s2)
        return common / max(len(s1), len(s2), 1)


def find_field_header(
    regions: list[LayoutRegion],
    field_name: str,
    fuzzy_threshold: float = 0.75,
) -> Optional[LayoutRegion]:
    """Find layout region containing the field header label."""
    keywords = FIELD_KEYWORDS.get(field_name, [])
    if not keywords:
        return None

    best_region: Optional[LayoutRegion] = None
    best_score: float = 0.0

    for region in regions:
        region_text = normalize_text(region.text)

        for keyword in keywords:
            kw_normalized = normalize_text(keyword)

            if kw_normalized in region_text:
                if 1.0 > best_score:
                    best_score = 1.0
                    best_region = region
                break

            score = fuzzy_match_score(kw_normalized, region_text)
            if score >= fuzzy_threshold and score > best_score:
                best_score = score
                best_region = region

    if best_region:
        logger.debug(f"Found header for '{field_name}' with score {best_score:.2f}")

    return best_region


def extract_date(text: str) -> Optional[str]:
    """Extract and normalize date from text."""
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def extract_policy_number(text: str) -> Optional[str]:
    """Extract policy number from text using regex patterns."""
    for pattern in POLICY_NUMBER_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def extract_currency(text: str) -> Optional[str]:
    """Extract currency/limit value from text."""
    for pattern in CURRENCY_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


class FieldMapper:
    """Main field extraction engine for ACORD 25 documents."""

    def __init__(
        self,
        fuzzy_threshold: float = 0.75,
        proximity_radius: int = 80,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.proximity_radius = proximity_radius

    def extract_value_near_header(
        self,
        header_region: LayoutRegion,
        all_words: list[OCRWord],
        field_name: str,
    ) -> Optional[str]:
        """Extract value from words spatially adjacent to a header."""
        hx, hy, hw, hh = header_region.bbox
        header_word_ids = {id(w) for w in header_region.words}
        search_words = [w for w in all_words if id(w) not in header_word_ids]

        right_words = find_words_near_bbox(
            search_words, hx, hy, hw, hh,
            radius_px=self.proximity_radius, direction="right"
        )

        if right_words:
            value = " ".join(w.text for w in sorted(right_words, key=lambda w: w.x))
            return value.strip()

        below_words = find_words_near_bbox(
            search_words, hx, hy, hw, hh,
            radius_px=self.proximity_radius * 2, direction="below"
        )

        if below_words:
            value = " ".join(w.text for w in sorted(below_words, key=lambda w: (w.y, w.x)))
            return value.strip()

        return None

    def map_fields(
        self,
        regions: list[LayoutRegion],
        ocr_result: OCRResult,
    ) -> dict[str, FieldMatch]:
        """Map all ACORD 25 fields from layout regions and OCR result."""
        results: dict[str, FieldMatch] = {}
        all_words = ocr_result.words
        full_text = ocr_result.full_text

        for field_name in FIELD_KEYWORDS.keys():
            header = find_field_header(regions, field_name, self.fuzzy_threshold)

            if header:
                value = self.extract_value_near_header(header, all_words, field_name)

                if value:
                    value = self._postprocess_value(field_name, value)
                    results[field_name] = FieldMatch(
                        field_name=field_name,
                        value=value,
                        confidence=0.85,
                        match_method="spatial_proximity",
                        bbox=header.bbox,
                    )
                    continue

            fallback = self._regex_fallback(field_name, full_text)
            if fallback:
                results[field_name] = FieldMatch(
                    field_name=field_name,
                    value=fallback,
                    confidence=0.60,
                    match_method="regex_fallback",
                )

        logger.info(f"Extracted {len(results)}/{len(FIELD_KEYWORDS)} fields")
        return results

    def _postprocess_value(self, field_name: str, value: str) -> str:
        """Apply field-specific cleaning and normalization."""
        value = value.strip()
        value = re.sub(r"[|\\/]{2,}", "", value)
        value = re.sub(r"\s+", " ", value)

        if "date" in field_name:
            date = extract_date(value)
            if date:
                return date

        if field_name == "policy_number":
            pol = extract_policy_number(value)
            if pol:
                return pol

        if "limit" in field_name:
            currency = extract_currency(value)
            if currency:
                return currency

        return value

    def _regex_fallback(self, field_name: str, full_text: str) -> Optional[str]:
        """Last-resort regex extraction directly on full document text."""
        if "date" in field_name:
            return extract_date(full_text)

        if field_name == "policy_number":
            return extract_policy_number(full_text)

        if "limit" in field_name:
            return extract_currency(full_text)

        return None