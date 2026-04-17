# generate_enhanced_50M.py
"""
Enhanced data generator with 50M rows and much richer patterns
Features:
- Multiple sentence templates and structures
- Age-based context variations
- Intensity variations
- Personality trait correlations
- Compound dreams (multiple careers mentioned)
- Contextual details (family influence, life events)
"""

import csv
import random
from pathlib import Path
from data_blueprints import (
    CAREER_BLUEPRINTS, 
    AMBIGUOUS_PAIRS,
    PERSONALITY_TRAITS,
    AGE_CONTEXT,
    INTENSITY_LEVELS
)

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "data" / "synthetic" / "dreams_50M_enhanced.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

TOTAL = 50_000_000  # 50 million rows
CHUNK = 200_000  # Larger chunks for efficiency
AMBIGUITY_RATE = 0.30  # 30% ambiguous cases
COMPOUND_RATE = 0.15  # 15% mention multiple dreams
CONTEXTUAL_RATE = 0.40  # 40% include contextual details

HEADERS = [
    "text",
    "career_primary",
    "career_secondary",
    "ambiguity_level",
    "popularity",
    "motivation_primary",
    "motivation_secondary",
    "risk_appetite",
    "social_orientation",
    "age_context",
    "intensity",
    "has_context",
    "personality_trait",
    "income_stability",
    "work_life_balance",
    "education_level"
]


def generate_contextual_additions():
    """Generate contextual phrases about why they wanted this career"""
    contexts = [
        "because I loved watching them on TV",
        "after seeing my {relation} do it",
        "because I wanted to help people",
        "because I was good at {skill}",
        "after watching a documentary about it",
        "because it looked exciting",
        "because I wanted to be famous",
        "because I wanted to make a difference",
        "because I was fascinated by {subject}",
        "after reading books about it",
        "because I wanted to earn a lot of money",
        "because I wanted to be independent",
        "because I admired {profession}s",
        "after a school project on it",
        "because my teacher encouraged me",
        "because I wanted to change the world",
        "because it seemed adventurous",
        "because I wanted to create things",
        "because I was curious about how things work",
        "because I enjoyed {activity}"
    ]
    
    relations = ["father", "mother", "uncle", "aunt", "neighbor", "friend's parent", "cousin"]
    skills = ["math", "art", "speaking", "writing", "building things", "helping others"]
    subjects = ["science", "technology", "nature", "people", "stories", "space"]
    activities = ["solving puzzles", "drawing", "reading", "playing sports", "performing"]
    
    context = random.choice(contexts)
    context = context.replace("{relation}", random.choice(relations))
    context = context.replace("{skill}", random.choice(skills))
    context = context.replace("{subject}", random.choice(subjects))
    context = context.replace("{activity}", random.choice(activities))
    context = context.replace("{profession}", random.choice(list(CAREER_BLUEPRINTS.keys())).replace("_", " "))
    
    return context


def generate_sentence_templates(career, action, intensity_key):
    """Generate diverse sentence structures"""
    blueprint = CAREER_BLUEPRINTS[career]
    intensity_phrases = INTENSITY_LEVELS[intensity_key]
    intensity_phrase = random.choice(intensity_phrases)
    
    # Get age context
    age_key = random.choice(list(AGE_CONTEXT.keys()))
    age_phrases = AGE_CONTEXT[age_key]
    age_phrase = random.choice(age_phrases)
    
    templates = [
        # Basic templates
        f"{intensity_phrase} {action}.",
        f"When I was {age_phrase}, {intensity_phrase.lower()} {action}.",
        f"As a child, {intensity_phrase.lower()} {action}.",
        
        # More complex structures
        f"{intensity_phrase} {action} and making a difference.",
        f"Ever since {age_phrase}, {intensity_phrase.lower()} {action}.",
        f"Growing up, {intensity_phrase.lower()} {action}.",
        f"{intensity_phrase} {action} professionally.",
        f"My childhood dream was {action}.",
        
        # With emotional context
        f"{intensity_phrase} {action} more than anything.",
        f"Nothing excited me more than the idea of {action}.",
        f"I couldn't imagine anything better than {action}.",
        
        # With motivation
        f"{intensity_phrase} {action} to {random.choice(['help others', 'make money', 'be famous', 'create something', 'make an impact'])}.",
        
        # Specific age mentions
        f"At age {random.randint(5, 15)}, {intensity_phrase.lower()} {action}.",
        f"Since I was {random.randint(5, 12)}, {intensity_phrase.lower()} {action}.",
    ]
    
    sentence = random.choice(templates)
    
    # Add contextual detail sometimes
    if random.random() < CONTEXTUAL_RATE:
        context = generate_contextual_additions()
        sentence = sentence.rstrip('.') + f" {context}."
        has_context = 1
    else:
        has_context = 0
    
    return sentence, age_key, intensity_key, has_context


def generate_compound_dream():
    """Generate text mentioning multiple career aspirations"""
    careers = random.sample(list(CAREER_BLUEPRINTS.keys()), 2)
    c1_bp = CAREER_BLUEPRINTS[careers[0]]
    c2_bp = CAREER_BLUEPRINTS[careers[1]]
    
    action1 = random.choice(c1_bp["actions"])
    action2 = random.choice(c2_bp["actions"])
    
    templates = [
        f"I dreamed of either {action1} or {action2}.",
        f"I wanted to be someone who could do both {action1} and {action2}.",
        f"First I wanted {action1}, but then I thought about {action2}.",
        f"I couldn't decide between {action1} and {action2}.",
        f"Sometimes I dreamed of {action1}, other times {action2}.",
        f"I was torn between {action1} and {action2}.",
    ]
    
    return random.choice(templates), careers[0], careers[1], 2


def determine_personality_trait(career):
    """Determine personality trait based on career"""
    for trait, careers in PERSONALITY_TRAITS.items():
        if career in careers:
            return trait
    return "balanced"


def generate_row():
    """Generate a single row of data"""
    
    # Decide row type
    row_type = random.random()
    
    if row_type < COMPOUND_RATE:
        # Compound dream (mentions multiple careers)
        text, c1, c2, ambiguity = generate_compound_dream()
        blueprint = CAREER_BLUEPRINTS[c1]
        career_primary = c1
        career_secondary = c2
        age_key = random.choice(list(AGE_CONTEXT.keys()))
        intensity_key = random.choice(list(INTENSITY_LEVELS.keys()))
        has_context = 1
        
    elif row_type < (COMPOUND_RATE + AMBIGUITY_RATE):
        # Ambiguous case
        c1, c2 = random.choice(AMBIGUOUS_PAIRS)
        b1, b2 = CAREER_BLUEPRINTS[c1], CAREER_BLUEPRINTS[c2]
        
        # Mix actions from both careers
        all_actions = b1["actions"] + b2["actions"]
        action = random.choice(all_actions)
        
        intensity_key = random.choice(list(INTENSITY_LEVELS.keys()))
        text, age_key, intensity_key, has_context = generate_sentence_templates(c1, action, intensity_key)
        
        career_primary = c1
        career_secondary = c2
        ambiguity = 2
        blueprint = b1
        
    else:
        # Clear single career case
        career = random.choice(list(CAREER_BLUEPRINTS.keys()))
        blueprint = CAREER_BLUEPRINTS[career]
        action = random.choice(blueprint["actions"])
        
        # Sometimes use childhood phrases instead
        if random.random() < 0.3:
            action = random.choice(blueprint["childhood_phrases"])
        
        intensity_key = random.choice(list(INTENSITY_LEVELS.keys()))
        text, age_key, intensity_key, has_context = generate_sentence_templates(career, action, intensity_key)
        
        career_primary = career
        career_secondary = ""
        ambiguity = 0
    
    # Extract metadata
    popularity = random.choice(blueprint["popularity"])
    motivation_primary = blueprint["motivations"][0]
    motivation_secondary = blueprint["motivations"][1] if len(blueprint["motivations"]) > 1 else blueprint["motivations"][0]
    risk = blueprint["risk"]
    social = blueprint["social"]
    personality = determine_personality_trait(career_primary)
    income_stability = blueprint["income_stability"]
    work_life_balance = blueprint["work_life_balance"]
    education_level = blueprint["education_required"]
    
    return [
        text,
        career_primary,
        career_secondary,
        ambiguity,
        popularity,
        motivation_primary,
        motivation_secondary,
        risk,
        social,
        age_key,
        intensity_key,
        has_context,
        personality,
        income_stability,
        work_life_balance,
        education_level
    ]


def main():
    """Main generation loop"""
    print(f"🚀 Starting generation of {TOTAL:,} rows")
    print(f"Output: {OUT}")
    print(f"Chunk size: {CHUNK:,}")
    print()
    
    write_header = not OUT.exists()
    
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow(HEADERS)
            print("✅ Headers written")
        
        written = 0
        
        while written < TOTAL:
            rows = []
            for _ in range(CHUNK):
                rows.append(generate_row())
            
            writer.writerows(rows)
            written += CHUNK
            
            progress = (written / TOTAL) * 100
            print(f"Progress: {written:,} / {TOTAL:,} ({progress:.1f}%)")
    
    print()
    print("=" * 60)
    print("🎉 GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Total rows: {TOTAL:,}")
    print(f"Output file: {OUT}")
    print(f"Ambiguous cases: ~{int(TOTAL * AMBIGUITY_RATE):,}")
    print(f"Compound dreams: ~{int(TOTAL * COMPOUND_RATE):,}")
    print(f"Contextual details: ~{int(TOTAL * CONTEXTUAL_RATE):,}")


if __name__ == "__main__":
    main()