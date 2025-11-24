"""
Florence-2 attribute extraction wrapper.
"""

from PIL import Image

from src.models.florence2_extractor import Florence2Extractor


class AttributeExtractor:
    """Run attribute extraction on product images."""

    def __init__(self, model_path=None, device="cuda"):
        self.model = Florence2Extractor(model_path=model_path, device=device)

    def extract_all(self, image):
        """Extract product attributes from an image using OCR."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        result = self.model.extract(image)
        return result

    def extract_single(self, image, entity_name):
        """Extract a specific attribute from an image."""
        return self.model.extract(image, entity_name=entity_name)
