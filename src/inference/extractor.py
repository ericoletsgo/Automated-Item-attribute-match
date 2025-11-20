"""
Florence-2 attribute extraction wrapper.
"""

import re
from PIL import Image

from src.models.florence2_extractor import Florence2Extractor


ENTITY_TYPES = [
    "item_weight",
    "item_volume",
    "voltage",
    "wattage",
    "maximum_weight_recommendation",
    "height",
    "width",
    "depth",
]


class AttributeExtractor:
    """Run attribute extraction on product images."""

    def __init__(self, model_path=None, device="cuda"):
        self.model = Florence2Extractor(model_path=model_path, device=device)

    def extract_all(self, image):
        """Extract all known attribute types from an image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        results = {}
        for entity in ENTITY_TYPES:
            result = self.model.extract(image, entity_name=entity)
            if result["value"] is not None:
                results[entity] = result

        return results

    def extract_single(self, image, entity_name):
        """Extract a specific attribute from an image."""
        return self.model.extract(image, entity_name=entity_name)
