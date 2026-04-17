import pandas as pd
import random
from pathlib import Path
import os

# -----------------------------
# Output Path
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
output_dir = BASE_DIR / "data" / "synthetic"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "dreams_1M.csv"

# -----------------------------
# Career Templates (Popularity 1–10)
# -----------------------------
career_templates = {
    "doctor": {
        "actions": ["healing patients", "saving lives", "working in hospitals"],
        "motivations": ["help people", "serve society"],
        "popularity_range": (6, 9)
    },
    "teacher": {
        "actions": ["teaching students", "explaining concepts", "guiding children"],
        "motivations": ["share knowledge", "shape future"],
        "popularity_range": (5, 8)
    },
    "engineer": {
        "actions": ["writing code", "building systems", "solving problems"],
        "motivations": ["create technology", "think logically"],
        "popularity_range": (6, 9)
    },
    "farmer": {
        "actions": ["growing crops", "working in fields", "harvesting food"],
        "motivations": ["feed people", "live close to nature"],
        "popularity_range": (3, 6)
    },
    "electrician": {
        "actions": ["fixing wiring", "installing electrical systems"],
        "motivations": ["solve practical problems"],
        "popularity_range": (3, 6)
    },
    "f1 driver": {
        "actions": ["racing cars", "driving at high speed"],
        "motivations": ["win championships", "push limits"],
        "popularity_range": (8, 10)
    },
    "entrepreneur": {
        "actions": ["starting a company", "building a startup"],
        "motivations": ["lead people", "build something big"],
        "popularity_range": (6, 10)
    },
    "boxer": {
        "actions": ["training hard", "fighting in the ring"],
        "motivations": ["become champion", "prove strength"],
        "popularity_range": (6, 9)
    },
    "cricketer": {
        "actions": ["playing cricket", "scoring runs"],
        "motivations": ["represent country", "entertain fans"],
        "popularity_range": (8, 10)
    },
    "footballer": {
        "actions": ["playing football", "scoring goals"],
        "motivations": ["win trophies", "be loved by fans"],
        "popularity_range": (8, 10)
    }
}

# -----------------------------
# Sentence Generator
# -----------------------------
starters = [
    "When I was a child, I dreamed of",
    "I always imagined myself",
    "Since childhood, I wanted to",
    "I used to dream about"
]

def generate_row():
    career = random.choice(list(career_templates.keys()))
    data = career_templates[career]

    action = random.choice(data["actions"])
    motivation = random.choice(data["motivations"])

    # popularity ∈ [1, 10]
    popularity = random.randint(*data["popularity_range"])

    text = f"{random.choice(starters)} {action} and {motivation}."

    return {
        "text": text,
        "career": career,
        "popularity": popularity
    }

# -----------------------------
# STREAMING GENERATION
# -----------------------------
TOTAL_ROWS = 1_000_000
CHUNK_SIZE = 50_000
chunks = TOTAL_ROWS // CHUNK_SIZE

write_header = not os.path.exists(output_path)

for i in range(chunks):
    rows = [generate_row() for _ in range(CHUNK_SIZE)]
    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        mode="a",
        index=False,
        header=write_header
    )

    write_header = False
    print(f"✅ Generated {(i + 1) * CHUNK_SIZE:,} rows")

print("🎉 DONE: 1 Million rows generated")
print("Saved to:", output_path)
