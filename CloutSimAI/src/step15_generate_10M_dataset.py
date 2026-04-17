import csv, random
from pathlib import Path
from data_blueprints import CAREER_BLUEPRINTS
from ambiguity_patterns import AMBIGUOUS_PAIRS

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "data" / "synthetic" / "dreams_10M.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

TOTAL = 10_000_000
CHUNK = 100_000
AMBIGUITY_RATE = 0.25

HEADERS = [
    "text","career_primary","career_secondary","ambiguity_level",
    "popularity","motivation_primary","motivation_secondary",
    "risk_appetite","social_orientation"
]

def make_sentence(action):
    return f"I always dreamed of {action}."

write_header = not OUT.exists()

with open(OUT, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(HEADERS)

    written = 0
    while written < TOTAL:
        rows = []
        for _ in range(CHUNK):
            if random.random() < AMBIGUITY_RATE:
                c1, c2 = random.choice(AMBIGUOUS_PAIRS)
                b1, b2 = CAREER_BLUEPRINTS[c1], CAREER_BLUEPRINTS[c2]
                action = random.choice(b1["actions"] + b2["actions"])
                rows.append([
                    make_sentence(action), c1, c2, 2,
                    random.choice(b1["popularity"]),
                    b1["motivations"][0], b2["motivations"][0],
                    b1["risk"], b1["social"]
                ])
            else:
                c = random.choice(list(CAREER_BLUEPRINTS.keys()))
                b = CAREER_BLUEPRINTS[c]
                rows.append([
                    make_sentence(random.choice(b["actions"])),
                    c, "", 0,
                    random.choice(b["popularity"]),
                    b["motivations"][0], b["motivations"][1],
                    b["risk"], b["social"]
                ])
        writer.writerows(rows)
        written += CHUNK
        print(f"Generated {written:,}")

print("✅ DONE 10M rows")