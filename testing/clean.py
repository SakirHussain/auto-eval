import json
import ftfy  # Fixes encoding issues like â€œ -> “
import re

# Install ftfy if not installed: pip install ftfy

# Paths
DATASET_FILE = "formatted_dataset.json"
CLEANED_OUTPUT_FILE = "cleaned_dataset.json"

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return text  # Skip non-strings

    # Fix broken encoding
    text = ftfy.fix_text(text)

    # Replace smart quotes with standard quotes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Remove HTML tags like <br>
    text = re.sub(r"<.*?>", " ", text)

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Load dataset
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Clean dataset
for entry in dataset:
    entry["question"] = clean_text(entry["question"])
    entry["ideal_answer"] = clean_text(entry["ideal_answer"])
    entry["rubric"] = [clean_text(r) for r in entry["rubric"]]

    for student in entry["student_answers"]:
        student["student_id"] = clean_text(student["student_id"])
        student["answer"] = clean_text(student["answer"])

# Save cleaned dataset
with open(CLEANED_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"✅ Fixed encoding and saved cleaned dataset to {CLEANED_OUTPUT_FILE}")
