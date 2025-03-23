import os
import json
import chardet

# Paths to dataset files
DATASET_ROOT = "D:\\college\\capstone\\auto eval\\testing"  # Update this
RAW_DIR = os.path.join(DATASET_ROOT, "raw")
SCORES_DIR = os.path.join(DATASET_ROOT, "scores")
QUESTIONS_FILE = os.path.join(DATASET_ROOT, "questions.txt")
ANSWERS_FILE = os.path.join(DATASET_ROOT, "answers.txt")
RUBRICS_FILE = os.path.join(DATASET_ROOT, "generated_rubrics_dataset.json")


# Function to detect encoding and read file content
def read_file_content(filepath):
    with open(filepath, "rb") as f:
        raw_data = f.read()
        detected_encoding = chardet.detect(raw_data)["encoding"] or "utf-8"

    # Read file with detected encoding
    with open(filepath, "r", encoding=detected_encoding, errors="replace") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Step 1: Load Questions and Ideal Answers
questions = {}
with open(QUESTIONS_FILE, "r", encoding="utf-8") as qf, open(ANSWERS_FILE, "r", encoding="utf-8") as af:
    for line_num, (question, ideal_answer) in enumerate(zip(qf, af), start=1):
        question_key = question.strip().split(" ")[0]  # Extract x.y format
        questions[question_key] = {
            "question": question.strip(),
            "ideal_answer": ideal_answer.strip()
        }

# Step 2: Load Rubrics
with open(RUBRICS_FILE, "r", encoding="utf-8") as rf:
    rubrics_data = json.load(rf)

# Step 3: Process Each Question File in `raw/`
dataset = []

for question_file in sorted(os.listdir(RAW_DIR)):  # e.g., "1.1", "1.2", etc.
    raw_path = os.path.join(RAW_DIR, question_file)

    if not os.path.isfile(raw_path):
        continue  # Skip if not a file

    question_info = questions.get(question_file, {"question": "", "ideal_answer": ""})
    rubric = rubrics_data.get(question_info["question"], [])

    student_answers = []

    # Read student answers (each line corresponds to a different student)
    student_lines = read_file_content(raw_path)

    # Read evaluator scores
    scores_path = os.path.join(SCORES_DIR, question_file)
    if not os.path.isdir(scores_path):
        continue  # Skip if no scores found for this question

    evaluators = ["ave", "me", "other"]
    evaluator_scores = {evaluator: [] for evaluator in evaluators}

    for evaluator in evaluators:
        score_file = os.path.join(scores_path, evaluator)
        if os.path.exists(score_file):
            scores_raw = read_file_content(score_file)
            evaluator_scores[evaluator] = [float(score) for score in scores_raw if score.replace(".", "").isdigit()]

    # Ensure each student answer aligns with scores
    for idx, student_answer in enumerate(student_lines):
        student_id = f"student_{idx+1}"
        scores = [
            evaluator_scores[evaluator][idx] if idx < len(evaluator_scores[evaluator]) else None
            for evaluator in evaluators
        ]

        student_answers.append({
            "student_id": student_id,
            "answer": student_answer,
            "human_scores": scores  # Scores from 3 evaluators
        })

    dataset.append({
        "question": question_info["question"],
        "ideal_answer": question_info["ideal_answer"],
        "rubric": rubric,
        "student_answers": student_answers
    })

# Step 4: Save the Structured Dataset to JSON
OUTPUT_FILE = "formatted_dataset.json"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"✅ Dataset successfully saved to {OUTPUT_FILE}")