import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "synthetic" / "dreams_10M.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "dreams_validated.csv"

# Allowed values
ALLOWED_CAREERS = {
    "actor", "entrepreneur", "doctor", "engineer", "teacher",
    "scientist", "lawyer", "politician", "cricketer", "footballer",
    "boxer", "musician", "artist", "journalist", "pilot",
    "architect", "designer", "psychologist", "data_scientist",
    "game_developer"
}

ALLOWED_RISK = {"low", "medium", "high"}
ALLOWED_SOCIAL = {"private", "public"}

# Validation Logic
def validate_chunk(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Required columns
    required_columns = {
        "text", "career_primary", "career_secondary", "ambiguity_level",
        "popularity", "motivation_primary", "motivation_secondary",
        "risk_appetite", "social_orientation"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing columns: {required_columns - set(df.columns)}"
        )

    # 2. Drop nulls
    df = df.dropna(subset=["text", "career_primary", "popularity"])

    # 3. Text sanity
    df = df[df["text"].str.len() > 20]

    # 4. Career sanity
    df = df[df["career_primary"].isin(ALLOWED_CAREERS)]
    df = df[
        (df["career_secondary"] == "") |
        (df["career_secondary"].isin(ALLOWED_CAREERS))
    ]

    # 5. Ambiguity sanity
    df = df[df["ambiguity_level"].isin([0, 1, 2])]

    # 6. Popularity bounds (1–5)
    df = df[df["popularity"].between(1, 5)]

    # 7. Risk & social sanity
    df = df[df["risk_appetite"].isin(ALLOWED_RISK)]
    df = df[df["social_orientation"].isin(ALLOWED_SOCIAL)]

    return df

# Main (chunk-safe)
def main():
    print("🚀 Starting validation (chunked)")

    chunksize = 500_000
    first_write = True
    total_rows = 0

    for chunk in pd.read_csv(INPUT_FILE, chunksize=chunksize):
        validated = validate_chunk(chunk)
        total_rows += len(validated)

        validated.to_csv(
            OUTPUT_FILE,
            mode="a",
            index=False,
            header=first_write
        )
        first_write = False

        print(f"✅ Validated rows so far: {total_rows:,}")

    print("🎉 Validation complete")
    print("Final saved file:", OUTPUT_FILE)
    print("Total validated rows:", total_rows)


if __name__ == "__main__":
    main()