import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR array."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array to PIL Image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE for adaptive contrast enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Apply Non-Local Means denoising to reduce scan artifacts."""
    return cv2.fastNlMeansDenoisingColored(image, None, h=5, hColor=5,
                                            templateWindowSize=7, searchWindowSize=21)

def deskew_image(image: np.ndarray) -> np.ndarray:
    """Detect and correct document skew using Hough lines (up to ±15 degrees)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        logger.debug("No lines detected for deskew")
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -15 < angle < 15:
                angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image

    logger.info(f"Deskewing by {median_angle:.2f} degrees")
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(image, rotation_matrix, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return deskewed


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding for binarization with uneven lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def enhance_contrast(pil_image: Image.Image, factor: float = 1.5) -> Image.Image:
    """Enhance image contrast using PIL for better OCR performance."""
    enhancer = ImageEnhance.Contrast(pil_image)
    return enhancer.enhance(factor)


def preprocess_image(
    pil_image: Image.Image,
    denoise: bool = True,
    deskew: bool = True,
    adaptive_thresh: bool = True,
    contrast_enhance: bool = True,
) -> Image.Image:
    """Full preprocessing pipeline: contrast, normalize, denoise, deskew, threshold."""
    logger.info("Starting image preprocessing")

    if contrast_enhance:
        pil_image = enhance_contrast(pil_image)

    cv_image = pil_to_cv2(pil_image)
    cv_image = normalize_image(cv_image)

    if denoise:
        cv_image = denoise_image(cv_image)

    if deskew:
        cv_image = deskew_image(cv_image)

    if adaptive_thresh:
        cv_image = adaptive_threshold(cv_image)

    result = cv2_to_pil(cv_image)
    logger.info("Preprocessing complete.")
    return result

def load_image_from_path(path:str) -> Image.Image:
    
    return Image.open(path).convert("RGB")

def resize_for_ocr(image: Image.Image,
                   target_dpi: int =300,
                   current_dpi: Optional[int] = None) -> Image.Image:
    if current_dpi is None:
        dpi_info = image.info.get("dpi",(72,72))
        current_dpi = dpi_info[0] if isinstance(dpi_info,tuple) else 72
        
    if current_dpi == target_dpi:
        return image
    
    scale_factor = target_dpi / current_dpi
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    
    if scale_factor > 1:
        logger.info(f"Upscaling image from {current_dpi} DPI to {target_dpi} DPI for better OCR accuracy.")
        return image.resize(new_size,Image.LANCZOS)
    return image 