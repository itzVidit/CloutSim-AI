import pandas as pd
import pathlib

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "synthetic" / "dreams_1M.csv"
OUTPUT_DIR = BASE_DIR / "data" / "proceed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "dreams_validated.csv"

# -----------------------------
# Allowed Careers
# -----------------------------
ALLOWED_CAREERS = {
    "doctor", "teacher", "engineer", "farmer", "electrician",
    "f1 driver", "entrepreneur", "boxer", "cricketer", "footballer"
}

# -----------------------------
# Validation Logic
# -----------------------------
def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting Data Validation")
    print("Total rows:", len(df))

    # 1. Required columns check
    required_columns = {"text", "career", "popularity"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")

    # 2. Drop nulls
    df = df.dropna()

    # 3. Career sanity
    invalid_career = ~df["career"].isin(ALLOWED_CAREERS)
    if invalid_career.any():
        print("Removing invalid careers:", df[invalid_career]["career"].unique())
        df = df[~invalid_career]

    # 4. Popularity bounds (1–10)
    before = len(df)
    df = df[(df["popularity"] >= 1) & (df["popularity"] <= 10)]
    print(f"Removed {before - len(df)} rows due to invalid popularity")

    # 5. Text sanity
    df = df[df["text"].str.len() > 10]

    print("Validation complete")
    print("Final row count:", len(df))

    return df.reset_index(drop=True)

# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(INPUT_FILE)
    validated_df = validate_dataset(df)
    validated_df.to_csv(OUTPUT_FILE, index=False)
    print("Saved validated dataset to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()