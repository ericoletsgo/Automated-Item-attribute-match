"""
Gradio app for Product Matcher.
Upload product photos to auto-extract specs, find matching listings, or compare items.
"""

import json
import hashlib
from pathlib import Path

import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image


pipeline = None
DEMO_MODE = True
IMAGE_DIR = Path("data/raw/images")


def load_pipeline():
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
        print("Models loaded successfully.")
    else:
        missing = [str(p) for p in [florence2_path, embedder_path, index_path, metadata_path] if not p.exists()]
        print(f"Missing: {missing}. Running in demo mode.")
        DEMO_MODE = True


def _url_to_image_path(url):
    url_hash = hashlib.md5(url.encode()).hexdigest()
    ext = url.split(".")[-1].split("?")[0]
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    return IMAGE_DIR / f"{url_hash}.{ext}"


def extract_attributes(image):
    if image is None:
        return "{}"

    if DEMO_MODE:
        return json.dumps({"note": "Demo mode"}, indent=2)

    result = pipeline.extractor.extract_all(image)
    output = {"raw_model_output": result["raw_output"]}

    if result["specs"]:
        for spec in result["specs"]:
            label = spec.get("attribute", "spec")
            output[label] = f"{spec['value']} {spec['unit']}"
    else:
        output["note"] = "No specs detected"

    return json.dumps(output, indent=2)


def find_similar(image):
    if image is None:
        return [], "Upload an image first."

    if DEMO_MODE:
        return [], "Demo mode."

    results = pipeline.matcher.find_similar(image, top_k=10)

    gallery_images = []
    table_data = []

    for i, r in enumerate(results):
        meta = r["metadata"]
        img_url = meta.get("image_path", "")

        img_path = _url_to_image_path(img_url) if img_url else None
        if img_path and img_path.exists():
            try:
                pil_img = Image.open(img_path).convert("RGB")
                gallery_images.append((pil_img, f"#{i+1} sim={r['similarity']:.3f}"))
            except Exception:
                pass

        table_data.append({
            "Rank": i + 1,
            "Similarity": f"{r['similarity']:.3f}",
            "Category": meta.get("entity_name", ""),
            "Value": f"{meta.get('value', '')} {meta.get('unit', '')}".strip(),
        })

    df = pd.DataFrame(table_data) if table_data else pd.DataFrame()
    return gallery_images, df


def compare_products(image_a, image_b):
    if image_a is None or image_b is None:
        return "Upload both images.", ""

    if DEMO_MODE:
        return pd.DataFrame(), "Demo mode"

    emb_a = pipeline.matcher.embedder.embed_image(image_a)
    emb_b = pipeline.matcher.embedder.embed_image(image_b)
    similarity = float((emb_a @ emb_b).item())

    result_a = pipeline.extractor.extract_all(image_a)
    result_b = pipeline.extractor.extract_all(image_b)

    rows = [
        {"": "Raw Output", "Product A": result_a["raw_output"], "Product B": result_b["raw_output"]},
    ]

    all_attrs = set()
    a_map, b_map = {}, {}
    for s in result_a["specs"]:
        key = s.get("attribute", f"spec")
        a_map[key] = f"{s['value']} {s['unit']}"
        all_attrs.add(key)
    for s in result_b["specs"]:
        key = s.get("attribute", f"spec")
        b_map[key] = f"{s['value']} {s['unit']}"
        all_attrs.add(key)

    for attr in sorted(all_attrs):
        rows.append({
            "": attr,
            "Product A": a_map.get(attr, "-"),
            "Product B": b_map.get(attr, "-"),
        })

    df = pd.DataFrame(rows)

    if similarity > 0.85:
        verdict = "Likely the same product category"
    elif similarity > 0.65:
        verdict = "Related products"
    else:
        verdict = "Different product types"

    score_html = f"<h3>Embedding Similarity: {similarity:.2%} — {verdict}</h3>"
    return df, score_html


def build_app():
    with gr.Blocks(title="Product Matcher") as app:
        gr.Markdown("# Product Matcher")
        gr.Markdown("Upload product images to extract specs, find matching listings, or compare items. "
                     "Powered by fine-tuned Florence-2 (LoRA) for OCR and SigLIP contrastive embeddings for similarity.")

        with gr.Tab("Extract Specs"):
            gr.Markdown("Upload a product photo to read weight, dimensions, voltage, or other specs from the image.")
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Product Photo")
                    extract_btn = gr.Button("Extract", variant="primary")
                with gr.Column():
                    attr_output = gr.JSON(label="Extracted Attributes")
            extract_btn.click(fn=extract_attributes, inputs=img_input, outputs=attr_output)

        with gr.Tab("Find Similar"):
            gr.Markdown("Upload a product photo to find the most similar items in the catalog (182k indexed products).")
            with gr.Row():
                with gr.Column(scale=1):
                    query_img = gr.Image(type="pil", label="Query Image")
                    search_btn = gr.Button("Search", variant="primary")
                with gr.Column(scale=2):
                    results_gallery = gr.Gallery(label="Top Matches", columns=5, height=400)
                    results_table = gr.Dataframe(label="Match Details")
            search_btn.click(fn=find_similar, inputs=query_img, outputs=[results_gallery, results_table])

        with gr.Tab("Compare"):
            gr.Markdown("Upload two product images to check similarity and compare extracted specs.")
            with gr.Row():
                img_a = gr.Image(type="pil", label="Product A")
                img_b = gr.Image(type="pil", label="Product B")
            compare_btn = gr.Button("Compare", variant="primary")
            similarity_html = gr.HTML()
            comparison_table = gr.Dataframe(label="Attribute Comparison")
            compare_btn.click(
                fn=compare_products,
                inputs=[img_a, img_b],
                outputs=[comparison_table, similarity_html],
            )

        with gr.Tab("Training Results"):
            gr.Markdown("## Model Training Summary")
            gr.Markdown("""
### Florence-2 (Attribute Extraction)
- **Model**: microsoft/Florence-2-large with LoRA (rank 32, 0.9% trainable params)
- **Training**: 10,000 steps, batch 4 x 8 gradient accumulation = effective 32
- **Final loss**: 2.03 (train), 2.00 (val)
- **Hardware**: NVIDIA T4, ~3.5 hours

### SigLIP Contrastive (Product Similarity)
- **Model**: google/siglip-base-patch16-224 + projection head (768 -> 512 -> 256)
- **Training**: 12 epochs (3 frozen + 9 fine-tuned), SupCon loss
- **Final loss**: 4.54 (train), 4.52 (val)
- **Hardware**: NVIDIA T4, ~24 hours

### FAISS Index
- **Vectors**: 182,408 product embeddings (256-d)
- **Index type**: IVFFlat with inner product similarity

[View W&B training runs](https://wandb.ai/ericnie12310-university-of-waterloo/product-matcher)
""")

    return app


if __name__ == "__main__":
    load_pipeline()
    app = build_app()
    app.launch(share=False)
