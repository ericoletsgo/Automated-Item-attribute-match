"""Tests for evaluation metrics."""

from src.evaluation.metrics import exact_match, fuzzy_match, units_compatible, compute_f1


def test_exact_match_true():
    pred = {"value": 2.4, "unit": "kilogram"}
    truth = {"value": 2.4, "unit": "kilogram"}
    assert exact_match(pred, truth) is True


def test_exact_match_false_value():
    pred = {"value": 2.5, "unit": "kilogram"}
    truth = {"value": 2.4, "unit": "kilogram"}
    assert exact_match(pred, truth) is False


def test_exact_match_false_unit():
    pred = {"value": 2.4, "unit": "gram"}
    truth = {"value": 2.4, "unit": "kilogram"}
    assert exact_match(pred, truth) is False


def test_fuzzy_match_within_tolerance():
    pred = {"value": 2.38, "unit": "kilogram"}
    truth = {"value": 2.4, "unit": "kilogram"}
    assert fuzzy_match(pred, truth, tolerance=0.05) is True


def test_fuzzy_match_outside_tolerance():
    pred = {"value": 3.0, "unit": "kilogram"}
    truth = {"value": 2.4, "unit": "kilogram"}
    assert fuzzy_match(pred, truth, tolerance=0.05) is False


def test_units_compatible_same_family():
    assert units_compatible("kilogram", "gram") is True
    assert units_compatible("cm", "inch") is True
    assert units_compatible("volt", "millivolt") is True


def test_units_compatible_different_family():
    assert units_compatible("kilogram", "volt") is False
    assert units_compatible("cm", "litre") is False


def test_compute_f1():
    predictions = [
        {"value": 2.4, "unit": "kilogram"},
        {"value": 5.0, "unit": "volt"},
        None,
    ]
    ground_truths = [
        {"value": 2.4, "unit": "kilogram"},
        {"value": 5.0, "unit": "watt"},
        {"value": 1.0, "unit": "litre"},
    ]
    result = compute_f1(predictions, ground_truths, exact_match)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["f1"] == 2 / 3
