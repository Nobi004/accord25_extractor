import json 
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

from config.settings import LOG_FORMAT, LOG_LEVEL


def setup_logging(level: str = LOG_LEVEL,log_file:str = None) -> logging.Logger:
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

def save_json_output(data: dict[str,Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2,ensure_ascii=False)
    logging.getLogger(__name__).info(f"Output saved to {output_path}")
    
def load_json(path: Path) -> dict[str,Any]:
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)
    
def pdf_to_images(pdf_path: str) -> list[str]:
    
    try:
        from pdf2image import convert_from_path
        images= convert_from_path(pdf_path,dpi=300)
        