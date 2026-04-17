"""
Enhanced validation script with quality metrics and statistics
(FULLY JSON-SAFE VERSION)
"""

import pandas as pd
from pathlib import Path
from collections import Counter
import json
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "synthetic" / "dreams_50M_enhanced.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "dreams_validated_enhanced.csv"
STATS_FILE = OUTPUT_DIR / "validation_stats.json"

# ============================================================
# ALLOWED VALUES
# ============================================================

ALLOWED_CAREERS = {
    "actor", "entrepreneur", "doctor", "engineer", "teacher",
    "scientist", "lawyer", "politician", "cricketer", "footballer",
    "boxer", "musician", "artist", "journalist", "pilot",
    "architect", "designer", "psychologist", "data_scientist",
    "game_developer"
}

ALLOWED_RISK = {"low", "medium", "high"}
ALLOWED_SOCIAL = {"private", "public"}
ALLOWED_INCOME = {"stable", "variable"}
ALLOWED_BALANCE = {"good", "medium", "poor"}
ALLOWED_EDUCATION = {"low", "medium", "high", "very_high"}
ALLOWED_AGE = {"very_young", "young", "preteen", "teen"}
ALLOWED_INTENSITY = {"mild", "moderate", "strong", "very_strong"}

# ============================================================
# STATISTICS
# ============================================================

stats = {
    "total_input": 0,
    "total_valid": 0,
    "dropped_nulls": 0,
    "dropped_text_length": 0,
    "dropped_career": 0,
    "dropped_ambiguity": 0,
    "dropped_popularity": 0,
    "dropped_other": 0,
    "career_distribution": Counter(),
    "ambiguity_distribution": Counter(),
    "popularity_distribution": Counter(),
    "risk_distribution": Counter(),
    "social_distribution": Counter(),
    "age_distribution": Counter(),
    "intensity_distribution": Counter(),
    "compound_dreams": 0,
    "contextual_details": 0
}

# ============================================================
# JSON SAFE CONVERTER (CRITICAL FIX)
# ============================================================

def make_json_safe(obj):
    """Recursively convert numpy / pandas types to JSON-safe Python types"""

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    return obj

# ============================================================
# VALIDATION LOGIC
# ============================================================

def validate_chunk(df: pd.DataFrame) -> pd.DataFrame:
    global stats

    initial = len(df)
    stats["total_input"] += initial

    required_columns = {
        "text", "career_primary", "career_secondary", "ambiguity_level",
        "popularity", "motivation_primary", "motivation_secondary",
        "risk_appetite", "social_orientation", "age_context", "intensity",
        "has_context", "personality_trait", "income_stability",
        "work_life_balance", "education_level"
    }

    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    # 1. Null check
    before = len(df)
    df = df.dropna(subset=["text", "career_primary", "popularity"])
    stats["dropped_nulls"] += before - len(df)

    # 2. Text length
    before = len(df)
    df = df[df["text"].str.len().between(15, 500)]
    stats["dropped_text_length"] += before - len(df)

    # 3. Career
    before = len(df)
    df = df[df["career_primary"].isin(ALLOWED_CAREERS)]
    df = df[
        (df["career_secondary"] == "") |
        (df["career_secondary"].isin(ALLOWED_CAREERS))
    ]
    stats["dropped_career"] += before - len(df)

    # 4. Ambiguity
    before = len(df)
    df = df[df["ambiguity_level"].isin([0, 1, 2])]
    stats["dropped_ambiguity"] += before - len(df)

    # 5. Popularity
    before = len(df)
    df = df[df["popularity"].between(1, 5)]
    stats["dropped_popularity"] += before - len(df)

    # 6. Other categorical
    before = len(df)
    df = df[df["risk_appetite"].isin(ALLOWED_RISK)]
    df = df[df["social_orientation"].isin(ALLOWED_SOCIAL)]
    df = df[df["income_stability"].isin(ALLOWED_INCOME)]
    df = df[df["work_life_balance"].isin(ALLOWED_BALANCE)]
    df = df[df["education_level"].isin(ALLOWED_EDUCATION)]
    stats["dropped_other"] += before - len(df)

    # Optional defaults
    df.loc[df["age_context"].isna(), "age_context"] = "unknown"
    df.loc[df["intensity"].isna(), "intensity"] = "moderate"

    # Update stats
    stats["total_valid"] += len(df)
    stats["career_distribution"].update(df["career_primary"])
    stats["ambiguity_distribution"].update(df["ambiguity_level"])
    stats["popularity_distribution"].update(df["popularity"])
    stats["risk_distribution"].update(df["risk_appetite"])
    stats["social_distribution"].update(df["social_orientation"])
    stats["age_distribution"].update(df["age_context"])
    stats["intensity_distribution"].update(df["intensity"])
    stats["compound_dreams"] += int((df["ambiguity_level"] == 2).sum())
    stats["contextual_details"] += int((df["has_context"] == 1).sum())

    return df

# ============================================================
# SAVE STATISTICS (FIXED)
# ============================================================

def save_statistics():
    stats_serializable = {
        k: dict(v) if isinstance(v, Counter) else v
        for k, v in stats.items()
    }

    stats_serializable = make_json_safe(stats_serializable)

    with open(STATS_FILE, "w") as f:
        json.dump(stats_serializable, f, indent=2)

    print(f"\n📁 Statistics saved to: {STATS_FILE}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Starting enhanced validation")
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}\n")

    if not INPUT_FILE.exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    chunksize = 500_000
    first_write = True

    for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=chunksize), 1):
        print(f"Processing chunk {i} ({len(chunk):,} rows)")
        validated = validate_chunk(chunk)

        validated.to_csv(
            OUTPUT_FILE,
            mode="a",
            index=False,
            header=first_write
        )

        first_write = False
        print(f"  ✅ Valid rows: {len(validated):,}")
        print(f"  📊 Total valid so far: {stats['total_valid']:,}\n")

    print("🎉 VALIDATION COMPLETE")
    save_statistics()
    print(f"✅ Final output saved to: {OUTPUT_FILE}")

# ============================================================

if __name__ == "__main__":
    main()
