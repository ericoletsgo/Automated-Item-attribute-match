"""
Evaluation metrics for attribute extraction and similarity matching.
"""

import re


def exact_match(predicted, ground_truth):
    """Check if predicted value and unit match exactly."""
    if predicted is None or ground_truth is None:
        return False
    return (
        predicted.get("value") == ground_truth.get("value")
        and predicted.get("unit") == ground_truth.get("unit")
    )


def fuzzy_match(predicted, ground_truth, tolerance=0.05):
    """Check if predicted value is within tolerance of ground truth."""
    if predicted is None or ground_truth is None:
        return False

    pred_val = predicted.get("value")
    true_val = ground_truth.get("value")

    if pred_val is None or true_val is None:
        return False

    # units must be in the same family
    if not units_compatible(predicted.get("unit", ""), ground_truth.get("unit", "")):
        return False

    # allow some numeric tolerance
    if true_val == 0:
        return pred_val == 0
    return abs(pred_val - true_val) / abs(true_val) <= tolerance


def units_compatible(unit_a, unit_b):
    """Check if two units are in the same measurement family."""
    families = {
        "weight": {"gram", "kilogram", "milligram", "ounce", "pound", "ton", "kg", "g", "mg", "oz", "lb"},
        "length": {"centimetre", "metre", "millimetre", "inch", "foot", "yard", "cm", "m", "mm", "in", "ft"},
        "volume": {"millilitre", "litre", "centilitre", "gallon", "fluid ounce", "ml", "l", "cl"},
        "power": {"watt", "kilowatt", "w", "kw"},
        "voltage": {"volt", "millivolt", "kilovolt", "v", "mv", "kv"},
    }

    unit_a = unit_a.lower().strip()
    unit_b = unit_b.lower().strip()

    for family_units in families.values():
        if unit_a in family_units and unit_b in family_units:
            return True

    return unit_a == unit_b


def compute_f1(predictions, ground_truths, match_fn=exact_match):
    """Compute F1 score over a set of predictions."""
    tp = 0
    fp = 0
    fn = 0

    for pred, truth in zip(predictions, ground_truths):
        if pred is not None and truth is not None:
            if match_fn(pred, truth):
                tp += 1
            else:
                fp += 1
        elif pred is not None:
            fp += 1
        elif truth is not None:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def recall_at_k(query_labels, retrieved_labels, k=10):
    """Compute Recall@K for similarity search."""
    hits = 0
    total = len(query_labels)

    for query_label, retrieved in zip(query_labels, retrieved_labels):
        top_k = retrieved[:k]
        if query_label in top_k:
            hits += 1

    return hits / total if total > 0 else 0
