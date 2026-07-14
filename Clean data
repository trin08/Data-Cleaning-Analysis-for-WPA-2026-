"""
clean_data.py
=============
Stage 1 of the WPA "Human or AI?" analysis pipeline.

Takes the raw Google Forms export and produces a single analysis-ready
CSV (Cleaned_WPA_Dataset.csv) consumed by score_and_analyze.py.

Missing Likert-scale survey items are left as NaN rather than imputed;
downstream statistics use pairwise deletion (nan_policy='omit' / .dropna()).

Run:
    python clean_data.py
"""

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

RAW_FILENAME = "WPA_ AI or Human_  (Responses) - Form Responses 1 (1).csv"
OUTPUT_FILENAME = "Cleaned_WPA_Dataset.csv"

# Google Forms exports these columns in a fixed order, so renaming by
# index is stable. This gives every later script a human-readable handle
# instead of "column index 4 is screen time."
COLUMN_RENAME_MAP = {
    1: "Consent",
    2: "Age",
    3: "Role",
    4: "Field_of_Study",
    5: "Screen_Time_Daily",
    6: "Social_Media_Time_Daily",
}

ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

MIN_ADULT_AGE = 18
MAX_PLAUSIBLE_SCREEN_TIME_HRS = 24


def convert_word_age(value):
    """Convert a written-out age to an int, e.g. 'twenty-two' -> 22.

    Handles both single words ('twenty') and hyphen/space-joined compounds
    ('twenty-two', 'twenty two'). Already-numeric values pass through
    unchanged.
    """
    if not isinstance(value, str):
        return value

    cleaned = value.strip().lower()
    if cleaned in ONES:
        return ONES[cleaned]
    if cleaned in TENS:
        return TENS[cleaned]

    parts = re.split(r"[\s-]+", cleaned)
    if len(parts) == 2 and parts[0] in TENS and parts[1] in ONES:
        return TENS[parts[0]] + ONES[parts[1]]

    return value


def load_raw(path: str = RAW_FILENAME) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {df.columns[i]: name for i, name in COLUMN_RENAME_MAP.items()}
    df = df.rename(columns=rename)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Age: convert written-out ages, coerce to numeric ---
    df["Age"] = df["Age"].apply(convert_word_age)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # --- Consent & age eligibility filter ---
    df = df[df["Consent"] == "I agree"]
    df = df[df["Age"] >= MIN_ADULT_AGE]

    # --- Duplicates & stray whitespace ---
    df = df.drop_duplicates()
    text_cols = df.select_dtypes(["object"]).columns
    df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())

    # --- Standardize Role capitalization (e.g. "student " -> "Student") ---
    if "Role" in df.columns:
        df["Role"] = df["Role"].str.title()

    # --- Screen time: coerce to numeric, cap implausible outliers ---
    if "Screen_Time_Daily" in df.columns:
        df["Screen_Time_Daily"] = pd.to_numeric(
            df["Screen_Time_Daily"], errors="coerce"
        )
        df.loc[
            df["Screen_Time_Daily"] > MAX_PLAUSIBLE_SCREEN_TIME_HRS,
            "Screen_Time_Daily",
        ] = MAX_PLAUSIBLE_SCREEN_TIME_HRS

    # --- Drop fully-empty columns and the now-redundant consent column ---
    df = df.dropna(axis=1, how="all")
    if "Consent" in df.columns:
        df = df.drop(columns=["Consent"])

    return df


def main():
    raw = load_raw()
    cleaned = clean(raw)
    cleaned.to_csv(OUTPUT_FILENAME, index=False)
    print("Processing complete.")
    print(f"Original shape: {raw.shape}")
    print(f"Cleaned shape:  {cleaned.shape}")
    print(f"File saved as:  {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
