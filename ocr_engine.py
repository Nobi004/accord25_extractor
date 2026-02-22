import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np 
from PIL import Image 

logger = logging.getLogger(__name__)

@dataclass
class OCRWord:
    