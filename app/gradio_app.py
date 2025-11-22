"""
Gradio app for Product Matcher.
Upload product photos to auto-extract specs, find matching listings, or compare items.
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image


pipeline = None
DEMO_MODE = True


def load_pipeline():
    """Load models if checkpoints exist, otherwise run in demo mode."""
    global pipeline, DEMO_MODE

    florence2_path = Path("checkpoints/florence2-lora")
    embedder_path = Path("checkpoints/contrastive/best_model.pt")
    index_path = Path("data/faiss_index")
    metadata_path = Path("data/metadata.db")

    if all(p.exists() for p in [florence2_path, embedder_path, index_path, metadata_path]):
        from src.inference.pipeline import ProductMatcherPipeline
        pipeline = ProductMatcherPipeline(
            florence2_path=str(florence2_path),
            embedder_path=str(embedder_path),
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            device="cuda",
        )
        DEMO_MODE = False
    else:
        print("Checkpoints not found. Running in demo mode with placeholder outputs.")
        DEMO_MODE = True


def extract_attributes(image):
    """Extract product attributes from uploaded image."""
    if image is None:
        return "Please upload an image."

    if DEMO_MODE:
        return json.dumps({
            "item_weight": {"value": 1.5, "unit": "kilogram", "raw": "1.5 kilogram"},
            "height": {"value": 25.4, "unit": "centimetre", "raw": "25.4 centimetre"},
            "note": "Demo mode - train models to get real predictions",
        }, indent=2)

    results = pipeline.extractor.extract_all(image)
    return json.dumps(results, indent=2)


def find_similar(image):
    """Find similar products in the index."""
    if image is None:
        return [], "Please upload an image."

    if DEMO_MODE:
        return [], "Demo mode - build FAISS index to enable similarity search."

    results = pipeline.matcher.find_similar(image, top_k=10)

    gallery_images = []
    table_data = []
    for r in results:
        meta = r["metadata"]
        table_data.append({
            "rank": len(table_data) + 1,
            "similarity": f"{r['similarity']:.3f}",
            "entity": meta.get("entity_name", ""),
            "value": f"{meta.get('value', '')} {meta.get('unit', '')}",
        })

    df = pd.DataFrame(table_data)
    return gallery_images, df


def compare_products(image_a, image_b):
    """Compare two product images."""
    if image_a is None or image_b is None:
        return "Please upload both images.", ""

    if DEMO_MODE:
        demo_comparison = [
            {"attribute": "item_weight", "product_a": "1.5 kilogram", "product_b": "2.0 kilogram", "match": False},
            {"attribute": "height", "product_a": "25.4 cm", "product_b": "25.4 cm", "match": True},
        ]
        df = pd.DataFrame(demo_comparison)
        return df, "Similarity: 0.82 (Demo mode)"

    result = pipeline.compare(image_a, image_b)
    df = pd.DataFrame(result["comparison"])
    score_html = f"<h3>Overall Similarity: {result['similarity_score']:.2%}</h3>"
    return df, score_html


def build_app():
    """Create the Gradio interface."""
    with gr.Blocks(
        title="Product Matcher",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# Product Matcher")
        gr.Markdown("Upload a product image to automatically extract its specifications, find matching listings, or check if two items are the same product.")

        with gr.Tab("Auto-Extract Specs"):
            gr.Markdown("Upload a product photo to pull out weight, dimensions, voltage, and other specs automatically.")
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Product Photo")
                    extract_btn = gr.Button("Extract Specs", variant="primary")
                with gr.Column():
                    attr_output = gr.JSON(label="Detected Specifications")

            extract_btn.click(fn=extract_attributes, inputs=img_input, outputs=attr_output)

        with gr.Tab("Find Matching Listings"):
            gr.Markdown("Upload a product photo to find the same or similar items across the catalog.")
            with gr.Row():
                with gr.Column():
                    query_img = gr.Image(type="pil", label="Product Photo")
                    search_btn = gr.Button("Search Matches", variant="primary")
                with gr.Column():
                    results_gallery = gr.Gallery(label="Matching Products", columns=5)
                    results_table = gr.Dataframe(label="Match Results")

            search_btn.click(fn=find_similar, inputs=query_img, outputs=[results_gallery, results_table])

        with gr.Tab("Compare Items"):
            gr.Markdown("Upload two product images to check if they are the same item and see how their specs differ.")
            with gr.Row():
                img_a = gr.Image(type="pil", label="Product A")
                img_b = gr.Image(type="pil", label="Product B")
            compare_btn = gr.Button("Compare", variant="primary")
            comparison_table = gr.Dataframe(label="Spec Comparison")
            similarity_html = gr.HTML()

            compare_btn.click(
                fn=compare_products,
                inputs=[img_a, img_b],
                outputs=[comparison_table, similarity_html],
            )

        with gr.Tab("Model Benchmarks"):
            gr.Markdown("## Extraction Accuracy Comparison")
            gr.Markdown("How the fine-tuned Florence-2 model compares to PaddleOCR and Claude Vision on the test set.")

            benchmark_path = Path("results/benchmark_results.json")
            if benchmark_path.exists():
                with open(benchmark_path) as f:
                    bench_data = json.load(f)
                gr.JSON(value=bench_data, label="Benchmark Results")

    return app


if __name__ == "__main__":
    load_pipeline()
    app = build_app()
    app.launch(share=False)
