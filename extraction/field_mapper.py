import logging 
import re 
from dataclasses import dataclass 
from typing import Optional 

from models.layout_model import LayoutRegion, cluster_words_into_lines, find_words_near_bbox 
from ocr.ocr_engine import OCRResult,OCRWord

logger = logging.getLogger(__name__)

