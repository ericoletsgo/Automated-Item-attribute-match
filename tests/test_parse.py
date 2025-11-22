"""Tests for attribute parsing."""

from src.data.parse_attributes import parse_entity_value


def test_parse_simple():
    value, unit = parse_entity_value("2.4 kilogram")
    assert value == 2.4
    assert unit == "kilogram"


def test_parse_integer():
    value, unit = parse_entity_value("120 volt")
    assert value == 120.0
    assert unit == "volt"


def test_parse_multi_word_unit():
    value, unit = parse_entity_value("15.6 fluid ounce")
    assert value == 15.6
    assert unit == "fluid ounce"


def test_parse_none():
    value, unit = parse_entity_value(None)
    assert value is None
    assert unit is None


def test_parse_nan():
    import pandas as pd
    value, unit = parse_entity_value(pd.NA)
    assert value is None
    assert unit is None
