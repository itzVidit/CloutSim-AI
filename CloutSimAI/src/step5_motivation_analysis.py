def analyze_motivation(text):
    text = text.lower()

    motivations = []

    if any(word in text for word in ["help", "heal", "serve", "care"]):
        motivations.append("Helping Others")

    if any(word in text for word in ["famous", "known", "millions", "popular"]):
        motivations.append("Fame")

    if any(word in text for word in ["lead", "company", "startup", "business"]):
        motivations.append("Leadership")

    if any(word in text for word in ["stage", "perform", "acting", "expression"]):
        motivations.append("Creative Expression")

    if not motivations:
        motivations.append("Unclear")

    return motivations


# -----------------------------
# Test
# -----------------------------
text = "I dreamed of being known by millions and performing on big stages"

print("Input:")
print(text)

print("\nDetected Motivations:")
print(analyze_motivation(text))
