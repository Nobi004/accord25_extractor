import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    field: str
    severity: str  # "error" | "warning" | "info"
    message: str


@dataclass
class ValidationResult:
    """Complete validation result for a document."""
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    score: float = 1.0  # 1.0 = fully valid, 0.0 = completely invalid


def validate_date(date_str: str, field_name: str) -> list[ValidationIssue]:
    """Validate that a date string is a real, parseable date."""
    issues = []
    if not date_str:
        return [ValidationIssue(field_name, "warning", f"{field_name} is empty.")]

    # Try common formats
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y"]
    parsed = None
    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue

    if parsed is None:
        issues.append(ValidationIssue(field_name, "error",
                                       f"Cannot parse date '{date_str}'"))
        return issues

    # Check for obviously invalid years
    if parsed.year < 1990 or parsed.year > 2100:
        issues.append(ValidationIssue(field_name, "warning",
                                       f"Date year {parsed.year} looks unusual."))

    return issues


def validate_date_range(
    effective_date: str,
    expiration_date: str,
) -> list[ValidationIssue]:
    """Validate that expiration is after effective date."""
    issues = []
    if not effective_date or not expiration_date:
        return issues

    formats = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"]
    eff, exp = None, None

    for fmt in formats:
        try:
            if eff is None:
                eff = datetime.strptime(effective_date, fmt)
        except ValueError:
            pass
        try:
            if exp is None:
                exp = datetime.strptime(expiration_date, fmt)
        except ValueError:
            pass

    if eff and exp and exp <= eff:
        issues.append(ValidationIssue(
            "expiration_date", "error",
            f"Expiration date ({expiration_date}) must be after effective date ({effective_date})."
        ))

    return issues


def validate_policy_number(policy_number: str) -> list[ValidationIssue]:
    """Validate policy number format."""
    issues = []
    if not policy_number:
        return [ValidationIssue("policy_number", "warning", "Policy number is empty.")]

    # Must be at least 4 characters
    if len(policy_number.replace(" ", "")) < 4:
        issues.append(ValidationIssue("policy_number", "warning",
                                       f"Policy number '{policy_number}' seems too short."))

    # Should contain at least one digit
    if not re.search(r"\d", policy_number):
        issues.append(ValidationIssue("policy_number", "warning",
                                       f"Policy number '{policy_number}' has no digits."))

    return issues


def validate_name(name: str, field_name: str) -> list[ValidationIssue]:
    """Validate name fields are not empty and look reasonable."""
    issues = []
    if not name or len(name.strip()) < 2:
        issues.append(ValidationIssue(field_name, "warning",
                                       f"{field_name} is empty or too short."))
        return issues

    # Flag if it looks like OCR garbage (mostly non-alpha)
    alpha_ratio = sum(1 for c in name if c.isalpha()) / len(name)
    if alpha_ratio < 0.4:
        issues.append(ValidationIssue(field_name, "warning",
                                       f"{field_name} may contain OCR noise: '{name}'"))

    return issues


def validate_document(output: dict[str, Any]) -> ValidationResult:
    """
    Run all validations on extracted document output.

    Args:
        output: Structured extraction output dict

    Returns:
        ValidationResult with issues and overall score
    """
    all_issues: list[ValidationIssue] = []

    # Validate names
    for name_field in ("producer_name", "insured_name", "certificate_holder"):
        all_issues.extend(validate_name(output.get(name_field, ""), name_field))

    # Validate dates
    all_issues.extend(validate_date(output.get("effective_date", ""), "effective_date"))
    all_issues.extend(validate_date(output.get("expiration_date", ""), "expiration_date"))
    all_issues.extend(validate_date_range(
        output.get("effective_date", ""),
        output.get("expiration_date", ""),
    ))

    # Validate policy number
    all_issues.extend(validate_policy_number(output.get("policy_number", "")))

    # Check required fields completeness
    required_fields = ["producer_name", "insured_name", "policy_number",
                       "effective_date", "expiration_date"]
    for rf in required_fields:
        if not output.get(rf):
            all_issues.append(ValidationIssue(rf, "warning",
                                               f"Required field '{rf}' is missing."))

    # Compute score: each error counts -0.15, each warning -0.05
    error_count = sum(1 for i in all_issues if i.severity == "error")
    warning_count = sum(1 for i in all_issues if i.severity == "warning")
    score = max(0.0, 1.0 - error_count * 0.15 - warning_count * 0.05)

    is_valid = error_count == 0

    logger.info(f"Validation: {error_count} errors, {warning_count} warnings, score={score:.2f}")

    return ValidationResult(is_valid=is_valid, issues=all_issues, score=score)
