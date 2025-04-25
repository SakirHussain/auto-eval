import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def compute_sbert_sim(a, b):
    emb1 = sbert_model.encode([a])[0]
    emb2 = sbert_model.encode([b])[0]
    return float(cosine_similarity([emb1], [emb2])[0][0])

def compute_tfidf_sim(a, b):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([a, b])
    return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])


# Load the JSON file with softened scores
with open("procot_eval_results_softened.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Create a DataFrame
df = pd.DataFrame(data)

# Compute similarity columns if not already present
if "sbert_similarity" not in df.columns:
    df["sbert_similarity"] = df.apply(lambda row: compute_sbert_sim(row["student_answer"], row["ideal_answer"]), axis=1)

if "tfidf_similarity" not in df.columns:
    df["tfidf_similarity"] = df.apply(lambda row: compute_tfidf_sim(row["student_answer"], row["ideal_answer"]), axis=1)


# Compute additional columns: absolute error for ProCoT and Softened scores
df["answer_length"] = df["student_answer"].apply(lambda x: len(x.split()))
df["error_procot"] = np.abs(df["human_avg_score"] - df["procot_score"])
df["error_softened"] = np.abs(df["human_avg_score"] - df["softened_score"])
df["diff_softened_procot"] = df["softened_score"] - df["procot_score"]

### Plot 1: Histogram of Score Distributions ###
# This plot shows the overall distribution of scores from human grading, raw ProCoT scores, and softened scores.
plt.figure(figsize=(10,6))
sns.histplot(df["human_avg_score"], color="green", label="Human Avg Score", kde=True, bins=20, stat="density")
sns.histplot(df["procot_score"], color="red", label="ProCoT Score", kde=True, bins=20, stat="density")
sns.histplot(df["softened_score"], color="blue", label="Softened Score", kde=True, bins=20, stat="density")
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("score_distribution.png")
plt.show()

### Plot 2: Scatter Plot - Human Avg Score vs. ProCoT Score ###
# This plot visualizes the correlation between human scores and raw ProCoT scores.
plt.figure(figsize=(8,6))
sns.scatterplot(x="human_avg_score", y="procot_score", data=df, color="red", label="ProCoT Score")
plt.plot([df["human_avg_score"].min(), df["human_avg_score"].max()],
         [df["human_avg_score"].min(), df["human_avg_score"].max()], "k--", label="Ideal")
plt.title("Human Avg Score vs. ProCoT Score")
plt.xlabel("Human Avg Score")
plt.ylabel("ProCoT Score")
plt.legend()
plt.tight_layout()
plt.savefig("scatter_human_vs_procot.png")
plt.show()

### Plot 3: Scatter Plot - Human Avg Score vs. Softened Score ###
# This plot compares human scores with the softened (adjusted) scores to assess improvement in alignment.
plt.figure(figsize=(8,6))
sns.scatterplot(x="human_avg_score", y="softened_score", data=df, color="blue", label="Softened Score")
plt.plot([df["human_avg_score"].min(), df["human_avg_score"].max()],
         [df["human_avg_score"].min(), df["human_avg_score"].max()], "k--", label="Ideal")
plt.title("Human Avg Score vs. Softened Score")
plt.xlabel("Human Avg Score")
plt.ylabel("Softened Score")
plt.legend()
plt.tight_layout()
plt.savefig("scatter_human_vs_softened.png")
plt.show()

### Plot 4: Error Distribution Histograms ###
# This histogram shows the distribution of absolute errors for both the raw ProCoT and softened scores.
plt.figure(figsize=(10,6))
sns.histplot(df["error_procot"], color="red", label="ProCoT Error", kde=True, bins=20, stat="density")
sns.histplot(df["error_softened"], color="blue", label="Softened Error", kde=True, bins=20, stat="density")
plt.title("Error Distribution: Absolute Errors")
plt.xlabel("Absolute Error")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("error_distribution.png")
plt.show()

### Plot 5: Boxplot for Score Comparison ###
# The boxplot summarizes the central tendency and variability of the human, ProCoT, and softened scores.
plt.figure(figsize=(8,6))
df_box = df[["human_avg_score", "procot_score", "softened_score"]]
sns.boxplot(data=df_box)
plt.title("Boxplot of Scores")
plt.xlabel("Score Type")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("boxplot_scores.png")
plt.show()

# Explanations for the plots:
# 1. Distribution of Scores (score_distribution.png): 
#    - Illustrates how human, raw ProCoT, and softened scores are distributed.
#    - Helps to visualize overall bias or shifts in scoring.
#
# 2. Scatter Plot - Human vs. ProCoT (scatter_human_vs_procot.png): 
#    - Demonstrates the correlation between human grading and raw ProCoT scores.
#    - Highlights the systematic underscoring.
#
# 3. Scatter Plot - Human vs. Softened (scatter_human_vs_softened.png):
#    - Shows the improved alignment between human scores and the adjusted (softened) scores.
#
# 4. Error Distribution (error_distribution.png): 
#    - Provides insight into the magnitude and frequency of errors.
#    - Compares how much the softened scores reduce the absolute error compared to raw scores.
#
# 5. Boxplot (boxplot_scores.png):
#    - Offers a summary of the score distributions, indicating medians, quartiles, and outliers.
#    - Useful for comparing central tendencies and variabilities across score types.

# ------------------------------
# Plot 1: Correlation Heatmap
# Purpose: To show relationships among numeric features: human_avg_score, procot_score, softened_score, answer_length, and errors.
plt.figure(figsize=(10, 8))
numeric_cols = ["human_avg_score", "procot_score", "softened_score", "answer_length", "error_procot", "error_softened"]
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# ------------------------------
# Plot 2: Pairplot
# Purpose: To visualize pairwise relationships among the key numerical variables.
sns.pairplot(df[numeric_cols])
plt.suptitle("Pairplot of Key Numerical Variables", y=1.02)
plt.savefig("pairplot.png")
plt.show()

# ------------------------------
# Plot 3: Scatter Plot - Human Avg Score vs. ProCoT Score
# Purpose: To assess the relationship and bias between human scores and raw ProCoT scores.
plt.figure(figsize=(8,6))
sns.regplot(x="human_avg_score", y="procot_score", data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'black'})
plt.title("Human Avg Score vs. ProCoT Score")
plt.xlabel("Human Avg Score")
plt.ylabel("ProCoT Score")
plt.tight_layout()
plt.savefig("scatter_human_vs_procot.png")
plt.show()

# ------------------------------
# Plot 4: Scatter Plot - Human Avg Score vs. Softened Score
# Purpose: To show the improved alignment of softened scores with human scores.
plt.figure(figsize=(8,6))
sns.regplot(x="human_avg_score", y="softened_score", data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'black'})
plt.title("Human Avg Score vs. Softened Score")
plt.xlabel("Human Avg Score")
plt.ylabel("Softened Score")
plt.tight_layout()
plt.savefig("scatter_human_vs_softened.png")
plt.show()

# ------------------------------
# Plot 5: Histogram of Answer Length
# Purpose: To understand the distribution of student answer lengths, which might impact evaluation.
plt.figure(figsize=(8,6))
sns.histplot(df["answer_length"], bins=20, kde=True, color="purple")
plt.title("Distribution of Answer Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("answer_length_distribution.png")
plt.show()

# ------------------------------
# Plot 6: Residual Error Histogram for Softened Scores
# Purpose: To visualize the distribution of absolute errors (residuals) between human scores and softened scores.
plt.figure(figsize=(8,6))
sns.histplot(df["error_softened"], bins=20, kde=True, color="blue")
plt.title("Residual Error Distribution (Human Avg Score vs. Softened Score)")
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("residual_error_softened.png")
plt.show()

# ------------------------------
# Plot 7: Difference Between Softened and ProCoT Scores
# Purpose: To quantify how much the softener model shifts scores relative to raw ProCoT scores.
plt.figure(figsize=(8,6))
sns.histplot(df["diff_softened_procot"], bins=20, kde=True, color="orange")
plt.title("Difference: Softened Score - ProCoT Score")
plt.xlabel("Score Difference")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("difference_softened_vs_procot.png")
plt.show()

# ------------------------------
# Plot 8: Pie Chart of Scoring Cases
# Purpose: To show the proportions of exact matches, over-scoring, and under-scoring cases.
exact_matches = sum(df["softened_score"] == df["human_avg_score"])
over_scoring = sum(df["softened_score"] > df["human_avg_score"])
under_scoring = sum(df["softened_score"] < df["human_avg_score"])
labels = ["Exact Matches", "Over-Scoring", "Under-Scoring"]
sizes = [exact_matches, over_scoring, under_scoring]
colors = ["green", "blue", "red"]

plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)
plt.title("Distribution of Scoring Cases (Softened Scores vs. Human Avg Score)")
plt.tight_layout()
plt.savefig("scoring_cases_pie_chart.png")
plt.show()

# ------------------------------
# Plot 9: Cumulative Distribution Function (CDF) of Softened Score Errors
# Purpose: To illustrate the cumulative proportion of evaluations with error less than or equal to a given value.
sorted_errors = np.sort(df["error_softened"])
cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
plt.figure(figsize=(8,6))
plt.plot(sorted_errors, cdf, marker=".", linestyle="none")
plt.xlabel("Absolute Error (Softened Score)")
plt.ylabel("Cumulative Proportion")
plt.title("CDF of Absolute Errors (Softened Scores)")
plt.grid(True)
plt.tight_layout()
plt.savefig("cdf_absolute_errors.png")
plt.show()

# Explanations:
# 1. Correlation Heatmap: Reveals interrelationships among key variables, helping to understand how each metric relates to others.
# 2. Pairplot: Provides pairwise comparisons and distributions, offering a comprehensive view of variable interactions.
# 3. Scatter Plots (Human vs. ProCoT & Human vs. Softened): Demonstrate the bias in raw scores and the improved alignment after softening.
# 4. Histogram of Answer Lengths: Shows distribution of student answer lengths, which may affect scoring.
# 5. Residual Error Histogram: Visualizes the magnitude and distribution of discrepancies between human and softened scores.
# 6. Difference Histogram: Quantifies the adjustment applied by the softener model.
# 7. Pie Chart of Scoring Cases: Summarizes the proportions of exact, over-, and under-scoring instances.
# 8. CDF of Absolute Errors: Illustrates what percentage of evaluations fall within a specific error margin, giving insight into overall accuracy.

# These plots collectively help "juice out" as much information as possible from the evaluations,
# offering both detailed error analysis and a broad view of the score distributions.

# Create a correlation heatmap including the softened score (colored version)
plt.figure(figsize=(10, 8))
features_with_softened = ["procot_score", "sbert_similarity", "tfidf_similarity", "answer_length", "softened_score"]
corr_matrix = df[features_with_softened].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"fontsize": 14})
plt.title("Correlation Heatmap: Features vs. Softened Score", fontsize=16)
plt.tight_layout()
plt.savefig("correlation_heatmap_with_softened_color.png")
plt.show()