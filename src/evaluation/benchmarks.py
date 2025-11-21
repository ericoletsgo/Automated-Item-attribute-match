"""
Benchmark Florence-2 against PaddleOCR and Claude Vision API.
"""

import json
import time
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.evaluation.metrics import compute_f1, exact_match, fuzzy_match


def run_florence2_benchmark(extractor, test_records, image_dir):
    """Evaluate fine-tuned Florence-2 on test set."""
    predictions = []
    ground_truths = []
    latencies = []

    for record in tqdm(test_records, desc="Florence-2"):
        img_path = image_dir / _get_image_filename(record["image_link"])
        if not img_path.exists():
            continue

        start = time.time()
        result = extractor.extract_single(Image.open(img_path), record["entity_name"])
        latencies.append(time.time() - start)

        predictions.append(result)
        ground_truths.append({"value": record["value"], "unit": record["unit"]})

    exact_f1 = compute_f1(predictions, ground_truths, exact_match)
    fuzzy_f1 = compute_f1(predictions, ground_truths, fuzzy_match)

    return {
        "model": "Florence-2 (fine-tuned)",
        "exact_f1": exact_f1,
        "fuzzy_f1": fuzzy_f1,
        "avg_latency_ms": sum(latencies) / len(latencies) * 1000,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
        "num_samples": len(predictions),
    }


def run_paddleocr_benchmark(test_records, image_dir):
    """Evaluate PaddleOCR + regex baseline on test set."""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    predictions = []
    ground_truths = []
    latencies = []

    for record in tqdm(test_records, desc="PaddleOCR"):
        img_path = image_dir / _get_image_filename(record["image_link"])
        if not img_path.exists():
            continue

        start = time.time()
        result = ocr.ocr(str(img_path), cls=True)
        latency = time.time() - start
        latencies.append(latency)

        # extract text and try to parse the target entity
        all_text = " ".join(
            line[1][0] for line in (result[0] or []) if line[1]
        )
        parsed = _regex_extract(all_text, record["entity_name"])
        predictions.append(parsed)
        ground_truths.append({"value": record["value"], "unit": record["unit"]})

    exact_f1 = compute_f1(predictions, ground_truths, exact_match)
    fuzzy_f1 = compute_f1(predictions, ground_truths, fuzzy_match)

    return {
        "model": "PaddleOCR + regex",
        "exact_f1": exact_f1,
        "fuzzy_f1": fuzzy_f1,
        "avg_latency_ms": sum(latencies) / len(latencies) * 1000,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
        "num_samples": len(predictions),
    }


def run_claude_benchmark(test_records, image_dir, max_samples=500):
    """Evaluate Claude Vision API (expensive, limit samples)."""
    import anthropic
    import base64

    client = anthropic.Anthropic()
    predictions = []
    ground_truths = []
    latencies = []

    samples = test_records[:max_samples]

    for record in tqdm(samples, desc="Claude Vision"):
        img_path = image_dir / _get_image_filename(record["image_link"])
        if not img_path.exists():
            continue

        with open(img_path, "rb") as f:
            img_data = base64.standard_b64encode(f.read()).decode("utf-8")

        start = time.time()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}},
                    {"type": "text", "text": f"Extract the {record['entity_name']} from this product image. Reply with just the number and unit, like '2.4 kilogram'."},
                ],
            }],
        )
        latencies.append(time.time() - start)

        text = response.content[0].text.strip()
        parsed = _parse_value_unit(text)
        predictions.append(parsed)
        ground_truths.append({"value": record["value"], "unit": record["unit"]})

    exact_f1 = compute_f1(predictions, ground_truths, exact_match)
    fuzzy_f1 = compute_f1(predictions, ground_truths, fuzzy_match)

    return {
        "model": "Claude Vision (claude-sonnet-4-20250514)",
        "exact_f1": exact_f1,
        "fuzzy_f1": fuzzy_f1,
        "avg_latency_ms": sum(latencies) / len(latencies) * 1000,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
        "num_samples": len(predictions),
    }


def _regex_extract(text, entity_name):
    """Try to find a value+unit in OCR text for a given entity type."""
    patterns = {
        "item_weight": r"(\d+\.?\d*)\s*(kg|kilogram|gram|g|pound|lb|oz|ounce)",
        "voltage": r"(\d+\.?\d*)\s*(v|volt|mv|millivolt|kv|kilovolt)",
        "wattage": r"(\d+\.?\d*)\s*(w|watt|kw|kilowatt)",
        "item_volume": r"(\d+\.?\d*)\s*(ml|millilitre|l|litre|gallon|fl oz)",
        "height": r"(\d+\.?\d*)\s*(cm|centimetre|m|metre|mm|millimetre|inch|in|ft|foot)",
        "width": r"(\d+\.?\d*)\s*(cm|centimetre|m|metre|mm|millimetre|inch|in|ft|foot)",
        "depth": r"(\d+\.?\d*)\s*(cm|centimetre|m|metre|mm|millimetre|inch|in|ft|foot)",
    }

    pattern = patterns.get(entity_name, r"(\d+\.?\d*)\s*(\w+)")
    match = re.search(pattern, text.lower())
    if match:
        return {"value": float(match.group(1)), "unit": match.group(2)}
    return None


def _parse_value_unit(text):
    """Parse a simple 'value unit' string."""
    match = re.match(r"([\d.]+)\s*(.+)", text.strip())
    if match:
        return {"value": float(match.group(1)), "unit": match.group(2).strip().lower()}
    return None


def _get_image_filename(url):
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()
    ext = url.split(".")[-1].split("?")[0]
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    return f"{url_hash}.{ext}"
