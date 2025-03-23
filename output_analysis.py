import json
import numpy as np
import scipy.stats as stats
from collections import Counter

# Path to results file
RESULTS_FILE = "procot_evaluation_results.json"

# Load evaluation results
with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

# Extract scores for analysis
human_avg_scores = np.array([entry["human_avg_score"] for entry in results])
procot_scores = np.array([entry["procot_score"] for entry in results])

# Compute absolute errors
absolute_errors = np.abs(human_avg_scores - procot_scores)

# Compute statistical metrics
mae = np.mean(absolute_errors)  # Mean Absolute Error
mse = np.mean((human_avg_scores - procot_scores) ** 2)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
correlation, _ = stats.pearsonr(human_avg_scores, procot_scores) if len(human_avg_scores) > 1 else (None, None)

# Compute Percentage Error Distribution
error_ranges = {
    "0-1 marks": 0,
    "1-2 marks": 0,
    "2-3 marks": 0,
    "3-4 marks": 0,
    "4+ marks": 0
}

for error in absolute_errors:
    if error < 1:
        error_ranges["0-1 marks"] += 1
    elif error < 2:
        error_ranges["1-2 marks"] += 1
    elif error < 3:
        error_ranges["2-3 marks"] += 1
    elif error < 4:
        error_ranges["3-4 marks"] += 1
    else:
        error_ranges["4+ marks"] += 1

# Convert to percentages
total_evals = len(absolute_errors)
error_percentages = {k: (v / total_evals) * 100 for k, v in error_ranges.items()}

# Compute overall error percentage (percentage of cases where ProCoT did not perfectly match the human score)
overall_error_percentage = np.mean(absolute_errors > 0) * 100

# Compute Over-Scoring, Under-Scoring, and Perfect Matches
perfect_matches = np.sum(procot_scores == human_avg_scores)
over_scoring = np.sum(procot_scores > human_avg_scores)
under_scoring = np.sum(procot_scores < human_avg_scores)

perfect_percentage = (perfect_matches / total_evals) * 100
over_scoring_percentage = (over_scoring / total_evals) * 100
under_scoring_percentage = (under_scoring / total_evals) * 100

# Sort results by the highest scoring errors
results_sorted = sorted(results, key=lambda x: abs(x["human_avg_score"] - x["procot_score"]), reverse=True)

# Print Top 10 Highest Errors for Debugging
print("\n🔍 **Top 10 Cases Where ProCoT Deviated the Most from Human Scores**\n")
for i in range(min(10, len(results_sorted))):
    entry = results_sorted[i]
    print(f" Q: {entry['question']}")
    print(f"   Student Answer: {entry['student_answer'][:200]}...")  # Print first 200 chars
    print(f"   Human Avg Score: {entry['human_avg_score']} | ProCoT Score: {entry['procot_score']}")
    print(f"   Absolute Error: {abs(entry['human_avg_score'] - entry['procot_score'])}")
    print("-" * 100)

# Print Overall Metrics
print("\n Overall Evaluation Metrics")
print(f" Mean Absolute Error (MAE): {mae:.4f}")
print(f" Mean Squared Error (MSE): {mse:.4f}")
print(f" Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f" Pearson Correlation Coefficient: {correlation:.4f}" if correlation is not None else " Pearson Correlation Coefficient: Not enough data")
print(f" Overall Error Percentage: {overall_error_percentage:.2f}% of cases had an error.")

# Print Over-Scoring, Under-Scoring, and Perfect Match Statistics
print("\nProCoT Scoring Distribution")
print(f" Perfect Matches (ProCoT = Human Score): {perfect_matches} ({perfect_percentage:.2f}%)")
print(f" Over-Scoring Cases (ProCoT > Human Score): {over_scoring} ({over_scoring_percentage:.2f}%)")
print(f" Under-Scoring Cases (ProCoT < Human Score): {under_scoring} ({under_scoring_percentage:.2f}%)")

# Print Error Distribution
print("\nError Percentage Distribution")
for range_label, percent in error_percentages.items():
    print(f"   - {range_label}: {percent:.2f}% of cases")

# Save error details to a file for analysis
ERROR_ANALYSIS_FILE = "procot_error_analysis.json"
with open(ERROR_ANALYSIS_FILE, "w", encoding="utf-8") as f:
    json.dump(results_sorted[:50], f, indent=4, ensure_ascii=False)

print(f"\nError analysis saved to {ERROR_ANALYSIS_FILE}")
