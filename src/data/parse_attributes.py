"""
Parse the Amazon ML Challenge CSV into structured attribute records.
Handles unit normalization and creates train/val/test splits.
"""

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_CSV = Path("data/raw/train.csv")
SPLITS_DIR = Path("data/splits")

# known units for each entity type
UNIT_MAP = {
    "item_weight": ["gram", "kilogram", "milligram", "microgram", "ounce", "pound", "ton"],
    "item_volume": ["millilitre", "litre", "centilitre", "gallon", "fluid ounce", "cup", "pint", "quart"],
    "voltage": ["volt", "millivolt", "kilovolt"],
    "wattage": ["watt", "kilowatt"],
    "maximum_weight_recommendation": ["gram", "kilogram", "milligram", "ounce", "pound", "ton"],
    "height": ["centimetre", "metre", "millimetre", "inch", "foot", "yard"],
    "width": ["centimetre", "metre", "millimetre", "inch", "foot", "yard"],
    "depth": ["centimetre", "metre", "millimetre", "inch", "foot", "yard"],
}


def parse_entity_value(raw_value):
    """Extract numeric value and unit from a string like '2.4 kilogram'."""
    if pd.isna(raw_value):
        return None, None

    raw_value = str(raw_value).strip().lower()
    match = re.match(r"([\d.]+)\s*(.+)", raw_value)
    if match:
        value = float(match.group(1))
        unit = match.group(2).strip()
        return value, unit
    return None, None


def build_records(df):
    """Convert dataframe rows into structured records."""
    records = []
    for _, row in df.iterrows():
        value, unit = parse_entity_value(row.get("entity_value"))

        record = {
            "index": int(row.name) if hasattr(row, "name") else len(records),
            "image_link": row["image_link"],
            "group_id": row.get("group_id", ""),
            "entity_name": row["entity_name"],
            "raw_value": str(row.get("entity_value", "")),
            "value": value,
            "unit": unit,
        }
        records.append(record)
    return records


def create_splits(records, train_ratio=0.7, val_ratio=0.15):
    """Stratified split by entity_name."""
    labels = [r["entity_name"] for r in records]

    train_records, temp_records, _, temp_labels = train_test_split(
        records, labels, train_size=train_ratio, stratify=labels, random_state=42
    )

    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_records, test_records = train_test_split(
        temp_records, train_size=val_ratio_adjusted, stratify=temp_labels, random_state=42
    )

    return train_records, val_records, test_records


def main():
    if not RAW_CSV.exists():
        print(f"CSV not found at {RAW_CSV}. Run data/download.py first.")
        return

    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df)} rows")

    records = build_records(df)
    valid_records = [r for r in records if r["value"] is not None]
    print(f"Valid records with parsed values: {len(valid_records)}/{len(records)}")

    train, val, test = create_splits(valid_records)
    print(f"Split sizes - train: {len(train)}, val: {len(val)}, test: {len(test)}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = SPLITS_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
