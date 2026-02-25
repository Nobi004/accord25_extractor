import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ocr.ocr_engine import OCRResult, OCRWord

logger = logging.getLogger(__name__)


@dataclass
class LayoutRegion:
    """A spatial region on the document."""
    label: str
    words: list[OCRWord]
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def text(self) -> str:
        """Joined text of all words in region."""
        return " ".join(w.text for w in sorted(self.words, key=lambda w: (w.y, w.x)))

def cluster_words_into_lines(
    words: list[OCRWord],
    line_tolerance_px: int = 12
) -> list[list[OCRWord]]:
    """Group words into lines based on vertical proximity."""
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: w.y)
    lines: list[list[OCRWord]] = []
    current_line: list[OCRWord] = [sorted_words[0]]

    for word in sorted_words[1:]:
        last_word = current_line[-1]
        if abs(word.y - last_word.y) <= line_tolerance_px:
            current_line.append(word)
        else:
            lines.append(sorted(current_line, key=lambda w: w.x))
            current_line = [word]

    if current_line:
        lines.append(sorted(current_line, key=lambda w: w.x))

    return lines


def find_words_near_bbox(
    words: list[OCRWord],
    ref_x: int, ref_y: int, ref_w: int, ref_h: int,
    radius_px: int = 80,
    direction: str = "right"
) -> list[OCRWord]:
    """Find words spatially near a reference bounding box (right, below, or any direction)."""
    nearby: list[OCRWord] = []
    ref_cx = ref_x + ref_w / 2
    ref_cy = ref_y + ref_h / 2

    for word in words:
        wc_x = word.cx
        wc_y = word.cy

        if direction == "right":
            if (wc_x > ref_x + ref_w and
                    abs(wc_y - ref_cy) < ref_h * 1.5 and
                    wc_x - (ref_x + ref_w) < radius_px * 2):
                nearby.append(word)

        elif direction == "below":
            if (wc_y > ref_y + ref_h and
                    wc_y - (ref_y + ref_h) < radius_px and
                    abs(wc_x - ref_cx) < ref_w):
                nearby.append(word)

        elif direction == "any":
            dist = np.sqrt((wc_x - ref_cx) ** 2 + (wc_y - ref_cy) ** 2)
            if dist < radius_px:
                nearby.append(word)

    return nearby



class LayoutParser:
    """Rule-based layout parser for ACORD 25 documents."""

    def __init__(self, proximity_radius: int = 80):
        self.proximity_radius = proximity_radius

    def parse(self, ocr_result: OCRResult) -> list[LayoutRegion]:
        """Parse OCR result into spatial layout regions."""
        lines = cluster_words_into_lines(ocr_result.words)
        regions: list[LayoutRegion] = []

        for line in lines:
            if not line:
                continue

            x = min(w.x for w in line)
            y = min(w.y for w in line)
            x_max = max(w.x + w.w for w in line)
            y_max = max(w.y + w.h for w in line)
            avg_conf = np.mean([w.confidence for w in line])

            regions.append(LayoutRegion(
                label="TEXT",
                words=line,
                confidence=float(avg_conf),
                bbox=(x, y, x_max - x, y_max - y),
            ))

        logger.info(f"Layout parser identified {len(regions)} text regions.")
        return regions


class LayoutLMv3Extractor:
    """Optional LayoutLMv3-based extractor for fine-tuned models."""

    def __init__(self, model_path: str):
        try:
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
            import torch

            self.processor = LayoutLMv3Processor.from_pretrained(model_path)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
            self.model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"LayoutLMv3 loaded from {model_path} on {self.device}")
        except ImportError:
            raise ImportError("transformers and torch required for LayoutLMv3.")

    def extract(self, image, ocr_result: OCRResult) -> dict:
        """Run LayoutLMv3 token classification to extract field-value pairs."""
        import torch

        words = [w.text for w in ocr_result.words]
        img_w, img_h = image.size
        boxes = [
            [
                int(w.x / img_w * 1000),
                int(w.y / img_h * 1000),
                int((w.x + w.w) / img_w * 1000),
                int((w.y + w.h) / img_h * 1000),
            ]
            for w in ocr_result.words
        ]

        encoding = self.processor(
            image, words, boxes=boxes,
            return_tensors="pt", truncation=True, padding="max_length"
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        id2label = self.model.config.id2label

        extracted: dict = {}
        for word, pred in zip(words, predictions):
            label = id2label.get(pred, "OTHER")
            if label != "OTHER":
                extracted.setdefault(label, []).append(word)

        return {k: " ".join(v) for k, v in extracted.items()}


def get_layout_parser(
    use_model: bool = False,
    model_path: Optional[str] = None,
    proximity_radius: int = 80,
):
    """Factory: returns rule-based parser with optional LayoutLMv3 extractor."""
    rule_parser = LayoutParser(proximity_radius=proximity_radius)
    model_extractor = None

    if use_model and model_path:
        try:
            model_extractor = LayoutLMv3Extractor(model_path)
        except Exception as e:
            logger.warning(f"LayoutLMv3 load failed ({e}), using rule-based only.")

    return rule_parser, model_extractor
    