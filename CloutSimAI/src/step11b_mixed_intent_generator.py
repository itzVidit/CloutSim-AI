import pandas as pd
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "synthetic"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE=OUTPUT_DIR/"dreams_mixed_intent.csv"

#mixed intent templates
MIXED_TEMPLATES = [
    {
        "text": "I loved performing for people but also dreamed of building something big and leading others",
        "careers": ["actor", "entrepreneur"]
    },
    {
        "text": "I enjoyed solving logical problems quietly but also liked teaching and guiding others",
        "careers": ["engineer", "teacher"]
    },
    {
        "text": "I wanted fame and attention but also wanted to serve society and help people",
        "careers": ["actor", "doctor"]
    },
    {
        "text": "I liked competition and pressure but also dreamed of running a successful company",
        "careers": ["f1 driver", "entrepreneur"]
    }
]

ROWS_PER_TEMPLATE=5000
rows=[]

for tp1 in MIXED_TEMPLATES:
    for _ in range(ROWS_PER_TEMPLATE):
        chosen_career=random.choice(tp1["careers"])
        popularity=random.randint(3,5)

        rows.append({
            "text":tp1["text"],
            "career":chosen_career,
            "popularity":popularity
        })

df=pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE,index=False)

print("✅ Mixed-intent dataset generated")
print("Rows:", len(df))
print("Saved to:", OUTPUT_FILE)