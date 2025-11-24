"""
End-to-end pipeline combining attribute extraction and similarity search.
"""

from PIL import Image

from src.inference.extractor import AttributeExtractor
from src.inference.matcher import ProductMatcher


class ProductMatcherPipeline:
    """Full pipeline: image to attributes and similar products."""

    def __init__(self, florence2_path, embedder_path, index_path, metadata_path, device="cuda"):
        self.extractor = AttributeExtractor(model_path=florence2_path, device=device)
        self.matcher = ProductMatcher(
            model_path=embedder_path,
            index_path=index_path,
            metadata_path=metadata_path,
            device=device,
        )

    def analyze(self, image, top_k=10):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        attributes = self.extractor.extract_all(image)
        similar = self.matcher.find_similar(image, top_k=top_k)

        return {
            "attributes": attributes,
            "similar_products": similar,
        }
