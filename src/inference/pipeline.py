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
        """Full analysis: extract attributes and find similar products."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        attributes = self.extractor.extract_all(image)
        similar = self.matcher.find_similar(image, top_k=top_k)

        return {
            "attributes": attributes,
            "similar_products": similar,
        }

    def compare(self, image_a, image_b):
        """Compare two product images side by side."""
        if isinstance(image_a, str):
            image_a = Image.open(image_a).convert("RGB")
        if isinstance(image_b, str):
            image_b = Image.open(image_b).convert("RGB")

        attrs_a = self.extractor.extract_all(image_a)
        attrs_b = self.extractor.extract_all(image_b)

        # compute embedding similarity
        emb_a = self.matcher.embedder.embed_image(image_a)
        emb_b = self.matcher.embedder.embed_image(image_b)
        similarity = float((emb_a @ emb_b).item())

        # build comparison table
        all_keys = set(list(attrs_a.keys()) + list(attrs_b.keys()))
        comparison = []
        for key in sorted(all_keys):
            a_val = attrs_a.get(key, {})
            b_val = attrs_b.get(key, {})
            comparison.append({
                "attribute": key,
                "product_a": f"{a_val.get('value', 'N/A')} {a_val.get('unit', '')}".strip(),
                "product_b": f"{b_val.get('value', 'N/A')} {b_val.get('unit', '')}".strip(),
                "match": a_val.get("value") == b_val.get("value") and a_val.get("unit") == b_val.get("unit"),
            })

        return {
            "similarity_score": similarity,
            "comparison": comparison,
            "attributes_a": attrs_a,
            "attributes_b": attrs_b,
        }
