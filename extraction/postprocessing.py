import logging
import re 
from datetime import datetime
from typing import Any, Optional

from extraction.field_mapper import FieldMatch

logger = logging.getLogger(__name__)

def normalize_date(date_str: str) -> str:
    date_str = date_str.strip()
    formats = [
        "%m/%d/%Y", "%m/%d/%y",    # MM/DD/YYYY, MM/DD/YY
        "%m-%d-%Y", "%m-%d-%y",    # MM-DD-YYYY
        "%Y/%m/%d", "%Y-%m-%d",    # ISO variants
        "%m.%d.%Y",                 # MM.DD.YYYY
        "%d/%m/%Y",                 # European (DD/MM/YYYY) — less common on ACORD
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # Handle 2-digit years: 00-49 → 2000-2049, 50-99 → 1950-1999
            if dt.year < 100:
                dt = dt.replace(year=dt.year + 2000 if dt.year < 50 else dt.year + 1900)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    logger.debug(f"Could not normalize date: {date_str}")
    return date_str

def normalize_currency(value: str) -> str:
    value = value.strip()
    # Remove spaces in numbers
    value = re.sub(r"(\d)\s(\d)", r"\1\2", value)
    # Ensure $ prefix
    if re.match(r"^\d", value):
        value = "$" + value
    return value

def normalize_name(name: str) -> str:
    # Remove extra whitespace
    name = re.sub(r"\s+", " ", name.strip())
    # Remove OCR artifacts at edges
    name = name.strip(".|,;:-")
    return name.title()

def build_structured_output(
    field_matches: dict,
    overall_confidence: float 
    
) -> dict[str, Any]:
    # Base output structure
    output = {
        "producer_name": "",
        "insured_name": "",
        "policy_number": "",
        "effective_date": "",
        "expiration_date": "",
        "coverages": [],
        "certificate_holder": "",
        "additional_insured": "",
        "subrogation_waiver": "",
        "limits": {
            "general_liability": "",
            "auto_liability": "",
            "umbrella": "",
            "workers_comp": "",
        },
        "_metadata": {
            "overall_confidence": round(overall_confidence, 3),
            "field_confidence": {},
            "extraction_method": "hybrid_rule_spatial",
        }
    }

    field_processors = {
        "producer_name": normalize_name,
        "insured_name": normalize_name,
        "certificate_holder": normalize_name,
        "effective_date": normalize_date,
        "expiration_date": normalize_date,
        "policy_number": str.strip,
        "general_liability_limit": normalize_currency,
        "auto_liability_limit": normalize_currency,
        "umbrella_limit": normalize_currency,
        "workers_comp_limit": normalize_currency,
    }

    for field_name, match in field_matches.items():
        value = match.value
        processor = field_processors.get(field_name, str.strip)

        try:
            value = processor(value)
        except Exception as e:
            logger.warning(f"Post-processing failed for {field_name}: {e}")

        # Map to output structure
        if field_name in ("producer_name", "insured_name", "policy_number",
                          "effective_date", "expiration_date", "certificate_holder",
                          "additional_insured", "subrogation_waiver"):
            output[field_name] = value

        elif field_name == "general_liability_limit":
            output["limits"]["general_liability"] = value
            # Also add to coverages list
            if value:
                output["coverages"].append({
                    "type": "General Liability",
                    "limit": value
                })

        elif field_name == "auto_liability_limit":
            output["limits"]["auto_liability"] = value
            if value:
                output["coverages"].append({
                    "type": "Automobile Liability",
                    "limit": value
                })

        elif field_name == "umbrella_limit":
            output["limits"]["umbrella"] = value
            if value:
                output["coverages"].append({
                    "type": "Umbrella / Excess Liability",
                    "limit": value
                })

        elif field_name == "workers_comp_limit":
            output["limits"]["workers_comp"] = value
            if value:
                output["coverages"].append({
                    "type": "Workers Compensation",
                    "limit": value
                })

        # Record confidence for each field
        output["_metadata"]["field_confidence"][field_name] = round(match.confidence, 3)

    return output


def compute_overall_confidence(field_matches: list[FieldMatch]) -> float:
    priority_fields = {"producer_name", "insured_name", "policy_number",
                       "effective_date", "expiration_date"}

    if not field_matches:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0

    for field_name, match in field_matches.items():
        weight = 2.0 if field_name in priority_fields else 1.0
        weighted_sum += match.confidence * weight
        weight_total += weight

    # Also penalize for missing priority fields
    for pf in priority_fields:
        if pf not in field_matches:
            weight_total += 2.0  # Add denominator weight for missing fields

    return weighted_sum / weight_total if weight_total > 0 else 0.0