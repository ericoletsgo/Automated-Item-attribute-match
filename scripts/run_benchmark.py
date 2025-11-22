"""
Run the full benchmark suite comparing Florence-2 vs PaddleOCR vs Claude Vision.
"""

import json
from pathlib import Path

from src.evaluation.benchmarks import (
    run_florence2_benchmark,
    run_paddleocr_benchmark,
    run_claude_benchmark,
)
from src.inference.extractor import AttributeExtractor


def main():
    test_path = Path("data/splits/test.json")
    image_dir = Path("data/raw/images")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(test_path) as f:
        test_records = json.load(f)

    # limit to 5000 samples for benchmarking
    test_records = test_records[:5000]
    print(f"Running benchmarks on {len(test_records)} samples")

    results = []

    # Florence-2
    print("\n--- Florence-2 (fine-tuned) ---")
    extractor = AttributeExtractor(model_path="checkpoints/florence2-lora", device="cuda")
    florence2_results = run_florence2_benchmark(extractor, test_records, image_dir)
    results.append(florence2_results)
    print(json.dumps(florence2_results, indent=2))

    # PaddleOCR
    print("\n--- PaddleOCR + regex ---")
    paddle_results = run_paddleocr_benchmark(test_records, image_dir)
    results.append(paddle_results)
    print(json.dumps(paddle_results, indent=2))

    # Claude Vision (limited samples due to cost)
    print("\n--- Claude Vision ---")
    claude_results = run_claude_benchmark(test_records, image_dir, max_samples=500)
    results.append(claude_results)
    print(json.dumps(claude_results, indent=2))

    # save results
    output_path = results_dir / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
