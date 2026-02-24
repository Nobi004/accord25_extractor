import json 
import logging 
import sys 
from pathlib import Path   

from typing import Any ,Optional

from PIL import Image
# Add project root to path 
sys.path.insert(0,str(Path(__file__).parent))

from config.settings import (
    OCR_ENGINE, OCR_CONFIDENCE_THRESHOLD, PREPROCESSING,
    LAYOUT_MODEL, FIELD_FUZZY_THRESHOLD, PROXIMITY_RADIUS_PX,
    OUTPUTS_DIR
)

from extraction.field_mapper import FieldMapper
from extraction.postprocessing import build_structured_output,compute_overall_confidence
from extraction.validation import validate_document
from models.layout_model import get_layout_parser
from ocr.ocr_engine import get_ocr_engine
from ocr.preprocessing import preprocess_image,resize_for_ocr
from utils.helpers import setup_logging,save_json_output,pdf_to_images,draw_extraction_overlay

logger = logging.getLogger(__name__)

class ACORD25Pipeline:

    def __init__(self,
                 ocr_engine_name:str = OCR_ENGINE,
                 confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
                 fuzzy_threshold: float = FIELD_FUZZY_THRESHOLD,
                 proximity_radius: int = PROXIMITY_RADIUS_PX,
                 preprocessing_config: dict = None,
                 layout_config: dict = None,
                 ):
        preprocessing_config = preprocessing_config or PREPROCESSING
        layout_config = layout_config or LAYOUT_MODEL

        logger.info("Initializing ACORD2Pipeline....")

        # Initialiizng OCR engine
        self.ocr_engine = get_ocr_engine(ocr_engine_name)
        self.confidence_threshold = confidence_threshold,

        # Initialize layout perser (+ optional LayoutLMv3)
        self.layout_perser,self.layout_model = get_layout_parser(
            use_model=layout_config.get("use_layout_model",False),
            model_path=layout_config.get("model_path"),
            proximity_radius=proximity_radius,
        )

        # Initialize field mapper
        self.field_mapper = FieldMapper(
            fuzzy_threshold=fuzzy_threshold,
            proximity_radius=proximity_radius,
        )

        self.preprocessing_config= preprocessing_config

        logger.info("Pipeline ready.")

    def process_image(self,image: Image.Image) -> dict[str,Any]:
        logger.info("=== Starting ACORD 25 extraction pipeline ===")

        # ____________ Step 1 Preprocessing _______________

        logger.info("Step 1 : Preprocessing image ....")

        image = resize_for_ocr(image)
        preprocessed = preprocess_image(
            image,
            denoise= self.preprocessing_config.get("denoise",True)
            deskew= self.preprocessing_config.get("deskew",True)
            adaptive_thresh= self.preprocessing_config.get("adaptive_thresh",True)
            contrast_enhance= self.preprocessing_config.get("contrast_enhance", True),
        )

        # __________ Step 2 OCR _____________
        logger.info("Step 2: Running OCR ...")
        ocr_result = self.ocr_engine.run(preprocessed)
        logger.info(f"OCR: {len(ocr_result.words)} words, "
                    f"avg confidence: {ocr_result.avg_confidence:.1f}%")
        
        # Filter low confidence words
        ocr_result = ocr_result.filter_by_confidence(self.confidence_threshold)
        logger.info(f"After confidence filter: {len(ocr_result.words)} words remaining")

        # _________Step 3: Layout Parsing ______________
        logger.info("Parsing document layout...")
        regions = self.layout_parser.parse(ocr_result)

        # ___________Step 4 : Field Extraction ___________
        logger.info("Step 4: Extraction Fields...")

        #Try LayoutLMv3 if available
        if self.layout_model:
            try:
                model_output = self.layout_model.extract(image, ocr_result)
                logger.info(f"LayoutLMv3 extracted {len(model_output)} fields.")
                # TODO: merge model output with rule-based for hybrid approach
            except Exception as e:
                logger.warning(f"LayoutLMv3 extraction failed: {e}, using rules.")

        field_matches = self.field_mapper.map_fields(regions,ocr_result)

        # ________________ Step 5 : Post-Processing_________________________
        logger.info("Step 5: Post-processing and structuring output...")
        overall_confidence= compute_overall_confidence(field_matches)
        structured = build_structured_output(field_matches,overall_confidence)