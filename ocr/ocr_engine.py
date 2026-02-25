import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)



@dataclass
class OCRWord:
    """Represents a single OCR-detected word with position and confidence."""
    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float

    @property
    def cx(self) -> float:
        """Center x coordinate."""
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        """Center y coordinate."""
        return self.y + self.h / 2

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return bounding box as (x, y, w, h)."""
        return (self.x, self.y, self.w, self.h)


@dataclass
class OCRResult:
    """Complete OCR result for a document image."""
    words: list[OCRWord] = field(default_factory=list)
    full_text: str = ""
    avg_confidence: float = 0.0

    def filter_by_confidence(self, threshold: float) -> "OCRResult":
        """Return new OCRResult with only words above confidence threshold."""
        filtered = [w for w in self.words if w.confidence >= threshold]
        full_text = " ".join(w.text for w in filtered if w.text.strip())
        avg_conf = np.mean([w.confidence for w in filtered]) if filtered else 0.0
        return OCRResult(words=filtered, full_text=full_text, avg_confidence=float(avg_conf))

class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def run(self, image: Image.Image) -> OCRResult:
        """Run OCR on a PIL image and return structured results."""
        ...

    def _build_full_text(self, words: list[OCRWord]) -> str:
        """Reconstruct text preserving approximate line structure."""
        if not words:
            return ""

        # Sort words by vertical position then horizontal
        sorted_words = sorted(words, key=lambda w: (round(w.y / 10) * 10, w.x))

        lines: list[list[OCRWord]] = []
        current_line: list[OCRWord] = []
        last_y = -999

        for word in sorted_words:
            if abs(word.y - last_y) > 15:  # New line threshold
                if current_line:
                    lines.append(current_line)
                current_line = [word]
                last_y = word.y
            else:
                current_line.append(word)

        if current_line:
            lines.append(current_line)

        return "\n".join(" ".join(w.text for w in line) for line in lines)


class TesseractEngine(BaseOCREngine):
    """
    Tesseract OCR engine wrapper.
    Uses pytesseract with detailed word-level bounding box output.
    """

    def __init__(self, config: str = "--oem 3 --psm 6", lang: str = "eng"):
        """
        Args:
            config: Tesseract configuration string.
                    --oem 3: Use LSTM neural net engine
                    --psm 6: Assume uniform block of text
            lang: OCR language
        """
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.config = config
            self.lang = lang
            logger.info("TesseractEngine initialized.")
        except ImportError:
            raise ImportError("pytesseract not installed. Run: pip install pytesseract")

    def run(self, image: Image.Image) -> OCRResult:
        """Run Tesseract on image, return word-level results with bounding boxes."""
        import pandas as pd

        # Get detailed OCR data including bounding boxes and confidence
        data = self.pytesseract.image_to_data(
            image, config=self.config, lang=self.lang,
            output_type=self.pytesseract.Output.DICT
        )

        words: list[OCRWord] = []
        n_boxes = len(data["level"])

        for i in range(n_boxes):
            text = str(data["text"][i]).strip()
            conf = float(data["conf"][i])

            # Skip empty text and invalid confidence values
            if not text or conf < 0:
                continue

            words.append(OCRWord(
                text=text,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
                confidence=conf,
            ))

        full_text = self._build_full_text(words)
        avg_conf = float(np.mean([w.confidence for w in words])) if words else 0.0

        logger.info(f"Tesseract extracted {len(words)} words, avg confidence: {avg_conf:.1f}%")
        return OCRResult(words=words, full_text=full_text, avg_confidence=avg_conf)


class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR engine wrapper.
    Better for noisy/rotated text but slower on CPU than Tesseract.
    """

    def __init__(self, lang_list: list[str] = None, gpu: bool = False):
        """
        Args:
            lang_list: Languages to recognize
            gpu: Use GPU if available
        """
        try:
            import easyocr
            lang_list = lang_list or ["en"]
            self.reader = easyocr.Reader(lang_list, gpu=gpu)
            logger.info(f"EasyOCREngine initialized (GPU={gpu}).")
        except ImportError:
            raise ImportError("easyocr not installed. Run: pip install easyocr")

    def run(self, image: Image.Image) -> OCRResult:
        """Run EasyOCR, return structured results compatible with OCRResult."""
        import numpy as np

        img_array = np.array(image)
        # detail=1 returns bounding boxes; paragraph=False for word-level
        results = self.reader.readtext(img_array, detail=1, paragraph=False)

        words: list[OCRWord] = []
        for (bbox, text, prob) in results:
            text = text.strip()
            if not text:
                continue

            # EasyOCR bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x = int(min(xs))
            y = int(min(ys))
            w = int(max(xs) - x)
            h = int(max(ys) - y)

            words.append(OCRWord(
                text=text,
                x=x, y=y, w=w, h=h,
                confidence=float(prob * 100),
            ))

        full_text = self._build_full_text(words)
        avg_conf = float(np.mean([w.confidence for w in words])) if words else 0.0

        logger.info(f"EasyOCR extracted {len(words)} words, avg confidence: {avg_conf:.1f}%")
        return OCRResult(words=words, full_text=full_text, avg_confidence=avg_conf)


def get_ocr_engine(engine_name: str = "tesseract", **kwargs) -> BaseOCREngine:
    engines = {
        "tesseract": TesseractEngine,
        "easyocr": EasyOCREngine,
    }
    if engine_name not in engines:
        raise ValueError(f"Unsupported OCR engine: {engine_name}. Supported engines: {list(engines.keys())}")
    return engines[engine_name](**kwargs)
