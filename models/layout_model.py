import logging 
from dataclasses import dataclass, field 
from typing import Optional 

import numpy as np 

from ocr.ocr_engine import OCRResult, OCRWord
logger = logging.getLogger(__name__)

@dataclass 
class LayoutRegion:
    label: str 
    words: list[OCRWord]
    confidence: float 
    bbox: tuple[int,int,int,int] # (x, y, w, h) of the region

    @property
    def text(self) -> str:
        return " ".join(w.text for w in sorted(self.words,key=lambda w: (w.y,w.x)))

def cluster_words_into_lines(
    words: list[OCRWord],
    line_tolerance_px: int = 12
    ) -> list[list(OCRWord)]:
    if not words:
        return []
    sorted_words = sorted(words,key=lambda w: w.y)
    lines: list[list(OCRWord)] = []
    current_line: list[OCRWord] = [sorted_words[0]]

    for word in sorted_words[1:]:
        last_word = current_line[-1]

        if abs(word.y - last_word.y) <= line_tolerance_px:
            current_line.append(word)
        else: 
            lines.append(sorted(current_line,key=lambda w: w.x))
            current_line = [word]

    if current_line:
        lines.append(sorted(current_line,key=lambda w: w.x))
        return lines

def find_words_near_bbox(
        words: list[OCRWord],
        ref_x: int,ref_y: int, ref_w: int, ref_h: int,
        radius_px: int = 80,
        direction:str = "right") -> list[OCRWord]:
    
    nearby: list[OCRWord] = []
    ref_cx = ref_x + ref_w/2
    ref_cy = ref_y + ref_h/2
)