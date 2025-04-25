import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import math
import json


with open("procot_eval_with_ideal.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def compute_sbert_sim(a, b):
    emb1 = sbert_model.encode([a])[0]
    emb2 = sbert_model.encode([b])[0]
    return float(cosine_similarity([emb1], [emb2])[0][0])

def compute_tfidf_sim(a, b):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([a, b])
    return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])


print("Computing Features...")

df['human_avg_score'] = df['human_scores'].apply(lambda x: sum(x)/len(x))
df['answer_length'] = df['student_answer'].apply(lambda x: len(x.split()))

df['sbert_similarity'] = df.apply(lambda row: compute_sbert_sim(row['student_answer'], row['ideal_answer']), axis=1)
df['tfidf_similarity'] = df.apply(lambda row: compute_tfidf_sim(row['student_answer'], row['ideal_answer']), axis=1)


X = df[['procot_score', 'sbert_similarity', 'tfidf_similarity', 'answer_length']]
y = df['human_avg_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)


final_scores = []
softener_scores = []
feature_names = ['procot_score', 'sbert_similarity', 'tfidf_similarity', 'answer_length']

for index, row in df.iterrows():
    procot = row['procot_score']
    student_ans = row['student_answer']
    ideal_ans = row['ideal_answer']

    sbert = compute_sbert_sim(student_ans, ideal_ans)
    tfidf = compute_tfidf_sim(student_ans, ideal_ans)
    length = len(student_ans.split())

    features = pd.DataFrame(np.array([[procot, sbert, tfidf, length]]), columns=feature_names)
    soft_score = model.predict(features)[0]
    soft_score = math.ceil(soft_score)
    lambda_ = 0
    final_score = lambda_ * procot + (1 - lambda_) * soft_score

    softener_scores.append(soft_score)
    final_scores.append(final_score)

df['softener_score'] = softener_scores
df['final_score'] = final_scores


def evaluate_scores(name, model_scores):
    human_scores = df['human_avg_score']
    errors = np.abs(human_scores - model_scores)

    mae = mean_absolute_error(human_scores, model_scores)
    mse = mean_squared_error(human_scores, model_scores)
    rmse = np.sqrt(mse)
    corr, _ = pearsonr(human_scores, model_scores)
    total_cases = len(human_scores)
    error_cases = sum(errors > 0)
    error_percentage = (error_cases / total_cases) * 100

    # Perfect, over, under scoring
    perfect = sum(human_scores == model_scores)
    over = sum(model_scores > human_scores)
    under = sum(model_scores < human_scores)

    # Error Buckets
    buckets = {
        "0-1": sum((errors > 0) & (errors <= 1)),
        "1-2": sum((errors > 1) & (errors <= 2)),
        "2-3": sum((errors > 2) & (errors <= 3)),
        "3-4": sum((errors > 3) & (errors <= 4)),
        "4+": sum(errors > 4)
    }

    # Total Percentage Error
    tpe = (errors.sum() / human_scores.sum()) * 100

    print(f"\n{name} Evaluation Metrics")
    print(f" MAE: {mae:.4f}")
    print(f" MSE: {mse:.4f}")
    print(f" RMSE: {rmse:.4f}")
    print(f" Pearson Correlation Coefficient: {corr:.4f}")
    print(f" Overall Error Percentage: {error_percentage:.2f}% of cases had an error.")
    print(f"\n {name} Scoring Distribution")
    print(f" Perfect Matches: {perfect} ({(perfect/total_cases)*100:.2f}%)")
    print(f" Over-Scoring Cases: {over} ({(over/total_cases)*100:.2f}%)")
    print(f" Under-Scoring Cases: {under} ({(under/total_cases)*100:.2f}%)")

    print(f"\n Error Percentage Distribution")
    for key, val in buckets.items():
        print(f"  - {key} marks: {(val/total_cases)*100:.2f}% of cases")

    print(f"\n Total Percentage Error: {tpe:.2f}%\n")


evaluate_scores("ProCoT", df['procot_score'])
evaluate_scores("Final Adjusted", df['final_score'])


plt.figure(figsize=(16, 12))

# Histogram of Scores
plt.subplot(2, 2, 1)
sns.histplot(df['human_avg_score'], color='green', label='Human', kde=True)
sns.histplot(df['procot_score'], color='red', label='ProCoT', kde=True)
sns.histplot(df['final_score'], color='blue', label='Final Adjusted', kde=True)
plt.title('Score Distributions')
plt.legend()

# Scatter Plot: Human vs ProCoT
plt.subplot(2, 2, 2)
sns.scatterplot(x=df['human_avg_score'], y=df['procot_score'], color='red', label='ProCoT')
plt.xlabel('Human Score')
plt.ylabel('ProCoT Score')
plt.title('Human vs ProCoT Score')

# Scatter Plot: Human vs Final Adjusted
plt.subplot(2, 2, 3)
sns.scatterplot(x=df['human_avg_score'], y=df['final_score'], color='blue', label='Final Adjusted')
plt.xlabel('Human Score')
plt.ylabel('Final Adjusted Score')
plt.title('Human vs Final Adjusted Score')

# Error Distribution
plt.subplot(2, 2, 4)
procot_errors = np.abs(df['human_avg_score'] - df['procot_score'])
final_errors = np.abs(df['human_avg_score'] - df['final_score'])
sns.histplot(procot_errors, color='red', label='ProCoT Error', kde=True)
sns.histplot(final_errors, color='blue', label='Final Score Error', kde=True)
plt.title('Error Distributions')
plt.legend()

plt.tight_layout()
plt.show()

sample_idx = random.randint(0, len(df)-1)

sample = df.iloc[sample_idx]
print("\n====== Sample Evaluation ======")
print(f"Question: {sample['question']}")
print(f"Student Answer: {sample['student_answer']}")
print(f"Human Scores: {sample['human_scores']} (Avg: {sample['human_avg_score']})")
print(f"ProCoT Score: {sample['procot_score']}")
print(f"Softened Score: {sample['final_score']:.2f}")
print("================================\n")

# Load the original JSON file
with open("procot_eval_with_ideal.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare an empty list to store updated records
result_list = []

# Iterate over every record in the original JSON file
for record in data:
    procot = record["procot_score"]
    student_ans = record["student_answer"]
    ideal_ans = record["ideal_answer"]

    # Compute similarity metrics and answer length
    sbert_val = compute_sbert_sim(student_ans, ideal_ans)
    tfidf_val = compute_tfidf_sim(student_ans, ideal_ans)
    length = len(student_ans.split())

    # Create a features DataFrame for the current record (ensure the feature_names list is defined)
    features = pd.DataFrame([[procot, sbert_val, tfidf_val, length]], columns=['procot_score', 'sbert_similarity', 'tfidf_similarity', 'answer_length'])
    
    # Predict the softened score using the trained regressor model
    soft_score = model.predict(features)[0]
    soft_score = math.ceil(soft_score)  # Use math.ceil to round up, as done previously

    # Create a new record with the additional "softened_score" field
    new_record = {
        "question": record["question"],
        "student_id": record["student_id"],
        "student_answer": student_ans,
        "human_scores": record["human_scores"],
        "human_avg_score": record["human_avg_score"],
        "procot_score": procot,
        "ideal_answer": ideal_ans,
        "softened_score": soft_score
    }
    result_list.append(new_record)

# Write the updated records to a new JSON file
with open("procot_eval_results_softened.json", "w", encoding="utf-8") as outfile:
    json.dump(result_list, outfile, indent=4, ensure_ascii=False)

print("New JSON file 'procot_eval_results_softened.json' created with softened scores.")
