"""
Comprehensive data quality analysis and insights generation
(FULLY JSON-SAFE, NO TUPLE-KEY ERRORS)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "dreams_validated_enhanced.csv"
OUTPUT_DIR = BASE_DIR / "data" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 1_000_000

# ============================================================
# JSON SAFE CONVERTER (KEY + VALUE SAFE)
# ============================================================

def make_json_safe(obj):
    """Recursively make object JSON serializable (keys + values)"""

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, Counter):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, pd.Series):
        return make_json_safe(obj.to_dict())

    if isinstance(obj, pd.DataFrame):
        return make_json_safe(obj.to_dict())

    return obj

# ============================================================
# ANALYZER
# ============================================================

class DataQualityAnalyzer:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.insights = {}

    # --------------------------------------------------------

    def analyze_text_diversity(self):
        print("\n📝 ANALYZING TEXT DIVERSITY...")

        sample = self.df['text'].sample(min(100_000, len(self.df)))

        word_counts = sample.str.split().str.len()
        words = " ".join(sample).lower().split()
        common_words = Counter(words).most_common(30)

        self.insights["text_diversity"] = {
            "uniqueness_ratio": sample.nunique() / len(sample),
            "avg_word_count": word_counts.mean(),
            "min_word_count": word_counts.min(),
            "max_word_count": word_counts.max(),
            "common_words": common_words
        }

        print(f"  ✅ Text uniqueness: {sample.nunique() / len(sample):.2%}")
        print(f"  ✅ Avg words per text: {word_counts.mean():.1f}")
        print(f"  ✅ Top 10 words: {', '.join(w for w, _ in common_words[:10])}")

    # --------------------------------------------------------

    def analyze_career_correlations(self):
        print("\n🔗 ANALYZING CAREER CORRELATIONS...")

        career_popularity = (
            self.df.groupby("career_primary")["popularity"]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False)
        )

        self.insights["career_correlations"] = {
            "popularity_by_career": career_popularity.to_dict()
        }

        print("  ✅ Top 5 popular careers:")
        for career, row in career_popularity.head().iterrows():
            print(f"     {career:20s}: {row['mean']:.2f} (±{row['std']:.2f})")

    # --------------------------------------------------------

    def analyze_ambiguity_patterns(self):
        print("\n🔀 ANALYZING AMBIGUITY PATTERNS...")

        ambiguous = self.df[self.df["ambiguity_level"] == 2]

        if ambiguous.empty:
            return

        pairs = ambiguous.apply(
            lambda r: tuple(sorted((r["career_primary"], r["career_secondary"]))),
            axis=1
        )

        pair_counts = Counter(pairs).most_common(20)

        self.insights["ambiguity"] = {
            "total_ambiguous": len(ambiguous),
            "common_pairs": [
                {"pair": list(pair), "count": count}
                for pair, count in pair_counts
            ]
        }

        print(f"  ✅ Total ambiguous cases: {len(ambiguous):,}")
        print("  ✅ Top 5 ambiguous pairs:")
        for pair, count in pair_counts[:5]:
            print(f"     {pair[0]} ↔ {pair[1]}: {count:,}")

    # --------------------------------------------------------

    def analyze_temporal_patterns(self):
        print("\n📅 ANALYZING TEMPORAL PATTERNS...")

        age_dist = self.df["age_context"].value_counts()

        self.insights["temporal"] = {
            "age_distribution": age_dist.to_dict()
        }

        for age, count in age_dist.items():
            print(f"     {age:15s}: {count:8,} ({count / len(self.df) * 100:5.2f}%)")

    # --------------------------------------------------------

    def analyze_balance_metrics(self):
        print("\n⚖️ ANALYZING BALANCE METRICS...")

        balance_popularity = self.df.groupby("work_life_balance")["popularity"].mean()

        self.insights["balance"] = {
            "balance_popularity": balance_popularity.to_dict()
        }

        for b, v in balance_popularity.items():
            print(f"     {b:10s}: {v:.2f}")

    # --------------------------------------------------------

    def check_data_biases(self):
        print("\n⚠️ CHECKING FOR DATA BIASES...")

        biases = []

        career_counts = self.df["career_primary"].value_counts()
        imbalance = career_counts.max() / career_counts.min()

        if imbalance > 3:
            biases.append(
                f"Career imbalance: {imbalance:.1f}x "
                f"(max={career_counts.max():,}, min={career_counts.min():,})"
            )

        self.insights["biases"] = biases

        for b in biases:
            print(f"     • {b}")

    # --------------------------------------------------------

    def generate_training_recommendations(self):
        print("\n🎯 GENERATING TRAINING RECOMMENDATIONS...")

        recs = [
            "✅ Rich features available: age_context, intensity, personality_trait",
            "✅ Consider creating combined features: risk×popularity, education×income",
            "✅ Recommended split: 80% train, 10% validation, 10% test",
            "✅ Use stratified split on career_primary",
        ]

        ambig_ratio = (self.df["ambiguity_level"] > 0).mean()
        recs.append(
            f"✅ {ambig_ratio:.1%} ambiguous cases - consider multi-label classification"
        )

        self.insights["recommendations"] = recs

        for r in recs:
            print(f"    {r}")

    # --------------------------------------------------------

    def save_insights(self):
        output_file = OUTPUT_DIR / "data_insights.json"

        safe_insights = make_json_safe(self.insights)

        with open(output_file, "w") as f:
            json.dump(safe_insights, f, indent=2)

        print(f"\n💾 Insights saved to: {output_file}")

    # --------------------------------------------------------

    def run_full_analysis(self):
        print("\n" + "=" * 70)
        print("DATA QUALITY ANALYSIS")
        print("=" * 70)

        self.analyze_text_diversity()
        self.analyze_career_correlations()
        self.analyze_ambiguity_patterns()
        self.analyze_temporal_patterns()
        self.analyze_balance_metrics()
        self.check_data_biases()
        self.generate_training_recommendations()
        self.save_insights()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Starting data quality analysis")
    print(f"Input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, nrows=SAMPLE_SIZE)
    print(f"✅ Loaded {len(df):,} rows")

    analyzer = DataQualityAnalyzer(df)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
