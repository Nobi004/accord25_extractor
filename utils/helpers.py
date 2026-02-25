import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

from config.settings import LOG_FORMAT, LOG_LEVEL


def setup_logging(level: str = LOG_LEVEL, log_file: str = None) -> logging.Logger:
    """Configure application logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger

def save_json_output(data: dict[str, Any], output_path: Path) -> None:
    """Save extraction results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.getLogger(__name__).info(f"Output saved to {output_path}")


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pdf_to_images(pdf_path: str) -> list:
    """Convert PDF pages to PIL Images using pdf2image."""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=300)
        logging.getLogger(__name__).info(f"Converted PDF to {len(images)} page(s)")
        return images
    except ImportError:
        raise ImportError("pdf2image not installed. Run: pip install pdf2image")
    except Exception as e:
        raise RuntimeError(f"PDF conversion failed: {e}")


def draw_extraction_overlay(
    image,
    field_matches: dict,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
):
    """Draw bounding box overlays on image for extracted fields."""
    import cv2
    import numpy as np
    from PIL import Image

    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for field_name, match in field_matches.items():
        if match.bbox is None:
            continue
        x, y, w, h = match.bbox
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), color, thickness)
        # Add field label above box
        cv2.putText(
            cv_img, field_name.replace("_", " ").title(),
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity between two strings."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0

    len_s1, len_s2 = len(s1), len(s2)
    dp = list(range(len_s2 + 1))

    for i in range(1, len_s1 + 1):
        prev_dp = dp[:]
        dp[0] = i
        for j in range(1, len_s2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[j] = prev_dp[j-1]
            else:
                dp[j] = 1 + min(prev_dp[j], dp[j-1], prev_dp[j-1])

    edit_distance = dp[len_s2]
    return 1.0 - edit_distance / max(len_s1, len_s2)


def evaluate_extraction(
    predicted: dict[str, str],
    ground_truth: dict[str, str],
) -> dict[str, Any]:

    exact_matches = 0
    partial_matches = 0
    field_scores = {}

    all_fields = set(list(predicted.keys()) + list(ground_truth.keys()))

    for field in all_fields:
        pred_val = str(predicted.get(field, "")).strip().lower()
        true_val = str(ground_truth.get(field, "")).strip().lower()

        if pred_val == true_val:
            exact_matches += 1
            partial_matches += 1
            field_scores[field] = {"exact": True, "similarity": 1.0}
        else:
            sim = levenshtein_similarity(pred_val, true_val)
            is_partial = sim >= 0.8
            if is_partial:
                partial_matches += 1
            field_scores[field] = {"exact": False, "similarity": round(sim, 3)}

    n = len(all_fields)
    return {
        "exact_match_accuracy": exact_matches / n if n > 0 else 0.0,
        "partial_match_accuracy": partial_matches / n if n > 0 else 0.0,
        "field_scores": field_scores,
        "fields_evaluated": n,
    }
