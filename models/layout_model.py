import logging 
from dataclasses import dataclass, field 
from typing import Optional 

import numpy as np 

from ocr.ocr_engine import OCRResult, OCRWord
logger = logging.getLogger(__name__)