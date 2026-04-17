# enhanced_career_blueprints.py
"""
Enhanced career blueprints with richer metadata for ML training
"""

CAREER_BLUEPRINTS = {
    "actor": {
        "actions": [
            "performing on stage", "acting in films", "entertaining crowds",
            "auditioning for roles", "rehearsing scripts", "working with directors",
            "winning awards", "attending premieres", "doing theater",
            "playing different characters", "expressing emotions through acting",
            "being on TV shows", "doing voice acting", "performing in plays"
        ],
        "childhood_phrases": [
            "being in movies", "being famous", "being on screen",
            "making people laugh", "telling stories through acting",
            "being a star", "performing for audiences"
        ],
        "motivations": ["creative", "fame", "expression", "attention"],
        "skills": ["charisma", "memorization", "emotional_expression", "confidence"],
        "environment": ["stage", "studio", "public_events"],
        "risk": "high",
        "social": "public",
        "popularity": [4, 5],
        "income_stability": "variable",
        "work_life_balance": "poor",
        "education_required": "medium",
        "related_interests": ["theater", "film", "storytelling", "entertainment"]
    },
    "entrepreneur": {
        "actions": [
            "building a company", "leading a startup", "scaling businesses",
            "creating new products", "pitching to investors", "managing teams",
            "taking risks", "innovating solutions", "disrupting industries",
            "making business deals", "building my own empire",
            "being my own boss", "launching startups", "creating jobs"
        ],
        "childhood_phrases": [
            "starting my own business", "being a business owner",
            "making money from ideas", "running a company",
            "being independent", "creating something of my own"
        ],
        "motivations": ["leadership", "impact", "independence", "wealth"],
        "skills": ["leadership", "risk_taking", "strategic_thinking", "networking"],
        "environment": ["office", "meetings", "networking_events"],
        "risk": "high",
        "social": "public",
        "popularity": [3, 4, 5],
        "income_stability": "variable",
        "work_life_balance": "poor",
        "education_required": "medium",
        "related_interests": ["business", "innovation", "technology", "finance"]
    },
    "doctor": {
        "actions": [
            "healing patients", "working in hospitals", "saving lives",
            "diagnosing diseases", "performing surgeries", "helping sick people",
            "doing medical research", "treating injuries", "caring for patients",
            "working in emergency rooms", "making people healthy",
            "studying medicine", "examining patients"
        ],
        "childhood_phrases": [
            "helping sick people", "saving lives", "curing diseases",
            "working in a hospital", "wearing a white coat",
            "using a stethoscope", "making people feel better"
        ],
        "motivations": ["service", "empathy", "helping", "prestige"],
        "skills": ["analytical_thinking", "empathy", "precision", "knowledge"],
        "environment": ["hospital", "clinic", "laboratory"],
        "risk": "medium",
        "social": "private",
        "popularity": [1, 2],
        "income_stability": "stable",
        "work_life_balance": "poor",
        "education_required": "very_high",
        "related_interests": ["science", "biology", "health", "helping_others"]
    },
    "engineer": {
        "actions": [
            "solving technical problems", "building systems", "writing code",
            "designing machines", "creating software", "fixing things",
            "inventing new technology", "working with computers",
            "developing apps", "building robots", "coding programs",
            "designing solutions", "working with circuits"
        ],
        "childhood_phrases": [
            "building things", "creating inventions", "working with technology",
            "making robots", "designing machines", "fixing computers",
            "solving puzzles", "understanding how things work"
        ],
        "motivations": ["logic", "creation", "problem_solving", "innovation"],
        "skills": ["analytical_thinking", "problem_solving", "technical", "systematic"],
        "environment": ["office", "laboratory", "workshop"],
        "risk": "low",
        "social": "private",
        "popularity": [1, 2],
        "income_stability": "stable",
        "work_life_balance": "good",
        "education_required": "high",
        "related_interests": ["technology", "science", "mathematics", "building"]
    },
    "teacher": {
        "actions": [
            "teaching students", "guiding learners", "explaining concepts",
            "inspiring young minds", "helping children learn",
            "preparing lessons", "grading papers", "mentoring students",
            "making learning fun", "shaping futures", "working in classrooms",
            "educating the next generation"
        ],
        "childhood_phrases": [
            "teaching kids", "being like my teacher", "helping others learn",
            "explaining things", "working in a school",
            "inspiring students", "making a difference in education"
        ],
        "motivations": ["service", "knowledge", "nurturing", "impact"],
        "skills": ["communication", "patience", "empathy", "organization"],
        "environment": ["classroom", "school", "educational_institutions"],
        "risk": "low",
        "social": "public",
        "popularity": [1, 2],
        "income_stability": "stable",
        "work_life_balance": "good",
        "education_required": "high",
        "related_interests": ["education", "children", "learning", "mentoring"]
    },
    "scientist": {
        "actions": [
            "conducting research", "discovering theories", "analyzing data",
            "running experiments", "making discoveries", "studying nature",
            "working in labs", "testing hypotheses", "publishing papers",
            "exploring the unknown", "advancing knowledge",
            "solving mysteries", "investigating phenomena"
        ],
        "childhood_phrases": [
            "making discoveries", "doing experiments", "learning about science",
            "finding new things", "working in a laboratory",
            "understanding the world", "being like Einstein"
        ],
        "motivations": ["curiosity", "discovery", "knowledge", "innovation"],
        "skills": ["analytical_thinking", "research", "patience", "creativity"],
        "environment": ["laboratory", "research_facility", "university"],
        "risk": "medium",
        "social": "private",
        "popularity": [1, 2],
        "income_stability": "stable",
        "work_life_balance": "medium",
        "education_required": "very_high",
        "related_interests": ["science", "research", "discovery", "learning"]
    },
    "lawyer": {
        "actions": [
            "arguing cases", "representing clients", "defending justice",
            "working in courtrooms", "studying law", "writing legal documents",
            "negotiating deals", "protecting rights", "prosecuting criminals",
            "advising clients", "interpreting laws", "winning cases"
        ],
        "childhood_phrases": [
            "defending people", "fighting for justice", "working in courts",
            "arguing cases", "protecting the innocent",
            "understanding the law", "being in a courtroom"
        ],
        "motivations": ["power", "justice", "prestige", "advocacy"],
        "skills": ["argumentation", "analytical_thinking", "communication", "persuasion"],
        "environment": ["courtroom", "office", "law_firm"],
        "risk": "medium",
        "social": "public",
        "popularity": [2, 3],
        "income_stability": "stable",
        "work_life_balance": "poor",
        "education_required": "very_high",
        "related_interests": ["justice", "debate", "politics", "advocacy"]
    },
    "politician": {
        "actions": [
            "leading people", "shaping policies", "addressing crowds",
            "running for office", "making speeches", "serving the public",
            "passing legislation", "campaigning", "representing constituents",
            "changing the country", "making laws", "debating issues"
        ],
        "childhood_phrases": [
            "leading people", "being in government", "making change",
            "running for president", "helping my country",
            "making important decisions", "giving speeches"
        ],
        "motivations": ["power", "impact", "service", "influence"],
        "skills": ["leadership", "communication", "charisma", "negotiation"],
        "environment": ["parliament", "public_events", "government_buildings"],
        "risk": "high",
        "social": "public",
        "popularity": [4, 5],
        "income_stability": "variable",
        "work_life_balance": "poor",
        "education_required": "medium",
        "related_interests": ["politics", "governance", "public_service", "debate"]
    },
    "cricketer": {
        "actions": [
            "playing professional cricket", "representing teams",
            "scoring centuries", "taking wickets", "winning matches",
            "training daily", "playing in stadiums", "representing my country",
            "being in the national team", "winning tournaments",
            "playing IPL", "breaking records"
        ],
        "childhood_phrases": [
            "playing cricket", "being like Sachin", "scoring runs",
            "playing for India", "hitting sixes", "being a cricket star",
            "playing in big stadiums", "winning matches"
        ],
        "motivations": ["fame", "competition", "glory", "excellence"],
        "skills": ["athleticism", "focus", "teamwork", "perseverance"],
        "environment": ["stadium", "cricket_ground", "training_facility"],
        "risk": "high",
        "social": "public",
        "popularity": [4, 5],
        "income_stability": "variable",
        "work_life_balance": "poor",
        "education_required": "low",
        "related_interests": ["sports", "competition", "fitness", "teamwork"]
    },
    "footballer": {
        "actions": [
            "playing football", "scoring goals", "training intensely",
            "representing clubs", "winning championships",
            "playing in world cup", "being a professional athlete",
            "dribbling past defenders", "playing for big teams",
            "breaking records", "becoming a legend"
        ],
        "childhood_phrases": [
            "playing football", "scoring goals", "being like Messi",
            "playing in the world cup", "being a football star",
            "playing for big clubs", "winning trophies"
        ],
        "motivations": ["fame", "competition", "glory", "excellence"],
        "skills": ["athleticism", "teamwork", "discipline", "coordination"],
        "environment": ["stadium", "training_ground", "football_pitch"],
        "risk": "high",
        "social": "public",
        "popularity": [4, 5],
        "income_stability": "variable",
        "work_life_balance": "poor",
        "education_required": "low",
        "related_interests": ["sports", "competition", "fitness", "teamwork"]
    },
    "boxer": {
        "actions": [
            "training for fights", "competing in the ring",
            "winning championships", "sparring", "building strength",
            "fighting professionally", "defending titles",
            "knockout victories", "training in the gym"
        ],
        "childhood_phrases": [
            "being a fighter", "winning fights", "being strong",
            "training like a champion", "becoming a boxing champion",
            "fighting in the ring", "being tough"
        ],
        "motivations": ["strength", "competition", "respect", "discipline"],
        "skills": ["physical_strength", "discipline", "courage", "strategy"],
        "environment": ["boxing_ring", "gym", "training_facility"],
        "risk": "high",
        "social": "public",
        "popularity": [3, 4],
        "income_stability": "variable",
        "work_life_balance": "poor",
        "education_required": "low",
        "related_interests": ["sports", "fitness", "combat", "discipline"]
    },
    "musician": {
        "actions": [
            "composing music", "performing concerts", "playing instruments",
            "writing songs", "recording albums", "touring the world",
            "making people dance", "creating melodies",
            "performing live", "producing music", "singing on stage"
        ],
        "childhood_phrases": [
            "making music", "being a singer", "playing instruments",
            "writing songs", "performing concerts", "being a rock star",
            "creating beautiful music", "being famous for music"
        ],
        "motivations": ["creative", "expression", "passion", "fame"],
        "skills": ["creativity", "musical_talent", "expression", "performance"],
        "environment": ["stage", "studio", "concert_hall"],
        "risk": "medium",
        "social": "public",
        "popularity": [3, 4, 5],
        "income_stability": "variable",
        "work_life_balance": "poor",
        "education_required": "medium",
        "related_interests": ["music", "art", "performance", "creativity"]
    },
    "artist": {
        "actions": [
            "painting artwork", "creating visual art", "sculpting",
            "drawing portraits", "exhibiting work", "expressing creativity",
            "designing art", "working in studios", "selling paintings",
            "creating masterpieces", "doing digital art"
        ],
        "childhood_phrases": [
            "making art", "painting pictures", "being creative",
            "drawing all day", "creating beautiful things",
            "expressing myself through art", "being an artist"
        ],
        "motivations": ["creative", "expression", "passion", "freedom"],
        "skills": ["creativity", "visual_thinking", "patience", "technique"],
        "environment": ["studio", "gallery", "workshop"],
        "risk": "medium",
        "social": "private",
        "popularity": [2, 3],
        "income_stability": "variable",
        "work_life_balance": "good",
        "education_required": "medium",
        "related_interests": ["art", "creativity", "design", "expression"]
    },
    "journalist": {
        "actions": [
            "reporting news", "investigating stories", "interviewing people",
            "writing articles", "uncovering truth", "covering events",
            "breaking news", "doing field reporting", "exposing corruption",
            "telling important stories", "working for media"
        ],
        "childhood_phrases": [
            "reporting news", "finding the truth", "being on TV news",
            "writing stories", "investigating mysteries",
            "telling people what's happening", "being a reporter"
        ],
        "motivations": ["truth", "impact", "curiosity", "justice"],
        "skills": ["communication", "investigation", "writing", "curiosity"],
        "environment": ["newsroom", "field", "press_events"],
        "risk": "medium",
        "social": "public",
        "popularity": [2, 3],
        "income_stability": "stable",
        "work_life_balance": "poor",
        "education_required": "medium",
        "related_interests": ["news", "writing", "investigation", "current_affairs"]
    },
    "pilot": {
        "actions": [
            "flying aircraft", "handling emergencies", "transporting passengers",
            "navigating through skies", "landing planes", "doing pre-flight checks",
            "flying internationally", "commanding the cockpit",
            "operating commercial flights"
        ],
        "childhood_phrases": [
            "flying planes", "being in the cockpit", "traveling the world",
            "commanding aircraft", "flying high in the sky",
            "being a captain", "controlling planes"
        ],
        "motivations": ["precision", "responsibility", "adventure", "prestige"],
        "skills": ["precision", "decision_making", "technical", "calm_under_pressure"],
        "environment": ["cockpit", "airport", "aircraft"],
        "risk": "high",
        "social": "private",
        "popularity": [2, 3],
        "income_stability": "stable",
        "work_life_balance": "poor",
        "education_required": "high",
        "related_interests": ["aviation", "travel", "technology", "adventure"]
    },
    "architect": {
        "actions": [
            "designing buildings", "planning structures", "creating blueprints",
            "working with construction", "designing homes",
            "planning cities", "making architectural drawings",
            "designing skyscrapers", "creating beautiful spaces"
        ],
        "childhood_phrases": [
            "designing buildings", "creating structures", "planning cities",
            "making blueprints", "building skyscrapers",
            "designing beautiful homes", "creating spaces"
        ],
        "motivations": ["design", "creation", "aesthetics", "legacy"],
        "skills": ["spatial_thinking", "creativity", "technical", "attention_to_detail"],
        "environment": ["office", "construction_site", "studio"],
        "risk": "medium",
        "social": "private",
        "popularity": [2, 3],
        "income_stability": "stable",
        "work_life_balance": "medium",
        "education_required": "high",
        "related_interests": ["design", "construction", "art", "engineering"]
    },
    "designer": {
        "actions": [
            "designing interfaces", "creating visuals", "working on graphics",
            "making websites", "designing logos", "creating user experiences",
            "doing creative work", "making digital designs",
            "working with clients", "designing products"
        ],
        "childhood_phrases": [
            "designing things", "making beautiful graphics",
            "creating digital art", "designing websites",
            "making logos", "being creative with computers"
        ],
        "motivations": ["creative", "design", "aesthetics", "innovation"],
        "skills": ["creativity", "visual_thinking", "technical", "user_empathy"],
        "environment": ["office", "studio", "remote"],
        "risk": "medium",
        "social": "private",
        "popularity": [2, 3],
        "income_stability": "stable",
        "work_life_balance": "good",
        "education_required": "medium",
        "related_interests": ["design", "art", "technology", "creativity"]
    },
    "psychologist": {
        "actions": [
            "counseling people", "studying behavior", "helping with mental health",
            "doing therapy sessions", "understanding emotions",
            "treating mental illness", "researching psychology",
            "helping people cope", "analyzing behavior"
        ],
        "childhood_phrases": [
            "helping people with problems", "understanding minds",
            "helping sad people", "being a therapist",
            "studying behavior", "making people feel better mentally"
        ],
        "motivations": ["empathy", "understanding", "helping", "insight"],
        "skills": ["empathy", "listening", "analytical_thinking", "patience"],
        "environment": ["clinic", "office", "hospital"],
        "risk": "low",
        "social": "private",
        "popularity": [1, 2],
        "income_stability": "stable",
        "work_life_balance": "good",
        "education_required": "very_high",
        "related_interests": ["psychology", "helping_others", "science", "counseling"]
    },
    "data_scientist": {
        "actions": [
            "analyzing data", "building models", "working with AI",
            "finding patterns", "creating algorithms", "predicting trends",
            "working with big data", "doing machine learning",
            "visualizing data", "solving business problems"
        ],
        "childhood_phrases": [
            "working with data", "solving puzzles with numbers",
            "working with computers", "finding patterns",
            "doing AI work", "analyzing information"
        ],
        "motivations": ["logic", "insight", "innovation", "problem_solving"],
        "skills": ["analytical_thinking", "programming", "mathematics", "statistics"],
        "environment": ["office", "remote", "tech_company"],
        "risk": "low",
        "social": "private",
        "popularity": [2, 3],
        "income_stability": "stable",
        "work_life_balance": "good",
        "education_required": "high",
        "related_interests": ["data", "technology", "mathematics", "AI"]
    },
    "game_developer": {
        "actions": [
            "building games", "designing gameplay", "programming game engines",
            "creating virtual worlds", "making fun experiences",
            "working on game graphics", "developing for consoles",
            "creating mobile games", "testing games"
        ],
        "childhood_phrases": [
            "making video games", "creating games", "building game worlds",
            "designing fun games", "working in gaming industry",
            "programming games", "making people have fun"
        ],
        "motivations": ["creative", "technology", "entertainment", "innovation"],
        "skills": ["programming", "creativity", "problem_solving", "teamwork"],
        "environment": ["office", "game_studio", "remote"],
        "risk": "medium",
        "social": "private",
        "popularity": [3, 4],
        "income_stability": "stable",
        "work_life_balance": "poor",
        "education_required": "high",
        "related_interests": ["gaming", "programming", "art", "storytelling"]
    }
}


# Enhanced ambiguity patterns with reasoning
AMBIGUOUS_PAIRS = [
    # Creative expression ambiguity
    ("actor", "musician"),  # Both performance-based
    ("musician", "artist"),  # Both creative expression
    ("artist", "designer"),  # Both visual creativity
    ("actor", "politician"),  # Both public speaking/charisma
    
    # Tech/analytical ambiguity
    ("engineer", "data_scientist"),  # Both technical/analytical
    ("engineer", "architect"),  # Both design/build systems
    ("data_scientist", "game_developer"),  # Both code and create
    ("engineer", "scientist"),  # Both problem-solving
    
    # Helping professions ambiguity
    ("doctor", "psychologist"),  # Both healthcare
    ("teacher", "psychologist"),  # Both helping/guiding
    ("doctor", "scientist"),  # Both research/science
    
    # Leadership ambiguity
    ("entrepreneur", "politician"),  # Both leadership
    ("entrepreneur", "actor"),  # Both fame/public attention
    ("lawyer", "politician"),  # Both public influence
    
    # Sports ambiguity
    ("cricketer", "footballer"),  # Both team sports
    ("footballer", "boxer"),  # Both athletic competition
    ("cricketer", "entrepreneur"),  # Both risk/glory
    
    # Communication/storytelling
    ("journalist", "politician"),  # Both public communication
    ("journalist", "teacher"),  # Both sharing knowledge
    ("actor", "teacher"),  # Both performance/presentation
    
    # Design/creation
    ("architect", "designer"),  # Both design-focused
    ("designer", "game_developer"),  # Both create experiences
    ("artist", "game_developer"),  # Both creative worlds
    
    # Complex analytical
    ("scientist", "data_scientist"),  # Both research/analysis
    ("psychologist", "data_scientist"),  # Both pattern analysis
    ("lawyer", "scientist"),  # Both analytical thinking
]


# Additional metadata for richer dataset
PERSONALITY_TRAITS = {
    "extroverted": ["actor", "politician", "teacher", "entrepreneur", "musician", "journalist"],
    "introverted": ["engineer", "scientist", "data_scientist", "artist", "psychologist"],
    "competitive": ["cricketer", "footballer", "boxer", "entrepreneur", "lawyer"],
    "creative": ["actor", "musician", "artist", "designer", "game_developer", "architect"],
    "analytical": ["engineer", "scientist", "data_scientist", "doctor", "lawyer", "psychologist"],
    "helper": ["doctor", "teacher", "psychologist", "lawyer"],
    "adventurous": ["pilot", "entrepreneur", "journalist", "actor"],
}

AGE_CONTEXT = {
    "very_young": ["3-6 years", "when I was very little", "in kindergarten"],
    "young": ["7-10 years", "in primary school", "as a young child"],
    "preteen": ["11-13 years", "in middle school", "as a preteen"],
    "teen": ["14-17 years", "in high school", "as a teenager"]
}

INTENSITY_LEVELS = {
    "mild": ["I thought about", "I considered", "I was interested in"],
    "moderate": ["I wanted to become", "I dreamed of", "I wished to be"],
    "strong": ["I always dreamed of", "I was obsessed with", "My biggest dream was"],
    "very_strong": ["All I ever wanted was", "My life goal was", "I was determined to be"]
}