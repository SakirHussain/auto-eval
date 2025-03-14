# import nltk
from nltk.tokenize import sent_tokenize
# nltk.download('punkt', quiet=True)

from sentence_transformers import SentenceTransformer
# import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline

# Initialize models/pipelines (SBERT + zero-shot)
print("[DEBUG] Initializing SBERT model and zero-shot classifier...")
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
print("[DEBUG] Models loaded successfully!\n")

def sbert_similarity(text: str, question: str) -> float:
    """Compute SBERT-based cosine similarity for (text, question)."""
    if not text.strip():
        return 0.0  # If text is empty, treat similarity as 0
    text_emb = sbert_model.encode([text], convert_to_numpy=True)[0]
    q_emb = sbert_model.encode([question], convert_to_numpy=True)[0]
    return float(cosine_similarity(text_emb.reshape(1, -1), q_emb.reshape(1, -1))[0][0])

def tfidf_similarity(text: str, question: str) -> float:
    """Compute TF-IDF-based cosine similarity for (text, question)."""
    if not text.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text, question])
    return float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])

def zero_shot_relevance(text: str, question: str) -> float:
    """
    Returns the 'relevance' probability (0-1) by doing a zero-shot classification
    that sees if 'text' is relevant or irrelevant to the question.
    We incorporate the question into the hypothesis_template for better context.
    """
    if not text.strip():
        return 0.0

    zsc_result = zero_shot_classifier(
        sequences=text,
        candidate_labels=["relevant", "irrelevant"],
        hypothesis_template="This text is {} to the question: " + question
    )
    # zsc_result has keys: 'labels', 'scores', 'sequence'
    # The 'relevant' label's score is what we want
    label_scores = dict(zip(zsc_result["labels"], zsc_result["scores"]))
    relevant_score = label_scores.get("relevant", 0.0)
    return float(relevant_score)


def filter_irrelevant_content(
    student_answer: str,
    question: str,
) -> str:
    """
    Single-pass approach:
      1) We split the student's answer into sentences (in order).
      2) Maintain a running "accepted_text."
      3) For each new sentence:
         - Form candidate_text = accepted_text + " " + new_sentence
         - Compare SBERT, TF-IDF, and zero-shot confidence for:
             accepted_text vs. question (old scores)
             candidate_text vs. question (new scores)
         - If the new score is >= old score in each method, that method "votes" yes.
         - If we have 2+ votes, we accept the sentence and update accepted_text.
      4) Return the final accepted_text.
    """

    print("[DEBUG] Starting filter_irrelevant_content (single pass, context-based).")
    print("[DEBUG] Student Answer:\n", student_answer)
    print("[DEBUG] Question:\n", question)

    # 1) Tokenize
    sentences = sent_tokenize(student_answer)
    print(f"[DEBUG] Found {len(sentences)} sentences.")

    # 2) Initialize accepted_text
    accepted_text = ""
    # Precompute old scores for accepted_text (initially empty -> 0)
    old_sbert_score = sbert_similarity(accepted_text, question)
    old_tfidf_score = tfidf_similarity(accepted_text, question)
    old_zsc_score = zero_shot_relevance(accepted_text, question)

    for idx, sent in enumerate(sentences):
        sent_str = sent.strip()
        if not sent_str:
            continue

        print(f"\n[DEBUG] Sentence {idx+1}: {sent_str}")

        # Build candidate text
        if accepted_text.strip():
            candidate_text = accepted_text + " " + sent_str
        else:
            candidate_text = sent_str  # If nothing accepted yet

        print("[DEBUG] Candidate text:")
        print(candidate_text)

        # 3) Compute new scores
        new_sbert_score = sbert_similarity(candidate_text, question)
        new_tfidf_score = tfidf_similarity(candidate_text, question)
        new_zsc_score = zero_shot_relevance(candidate_text, question)

        print(f"[DEBUG] Old SBERT: {old_sbert_score:.4f}, New SBERT: {new_sbert_score:.4f}")
        print(f"[DEBUG] Old TF-IDF: {old_tfidf_score:.4f}, New TF-IDF: {new_tfidf_score:.4f}")
        print(f"[DEBUG] Old ZSC: {old_zsc_score:.4f}, New ZSC: {new_zsc_score:.4f}")

        # If new >= old, that's a "vote" from each method
        votes = 0
        if new_sbert_score >= old_sbert_score:
            votes += 1
            print("[DEBUG] SBERT vote: YES")
        else:
            print("[DEBUG] SBERT vote: NO")

        if new_tfidf_score >= old_tfidf_score:
            votes += 1
            print("[DEBUG] TF-IDF vote: YES")
        else:
            print("[DEBUG] TF-IDF vote: NO")

        if new_zsc_score >= old_zsc_score:
            votes += 1
            print("[DEBUG] ZSC vote: YES")
        else:
            print("[DEBUG] ZSC vote: NO")

        print(f"[DEBUG] Total votes: {votes}")

        # 4) Majority: if 2 or more methods say "YES," we accept the sentence
        if votes >= 2:
            print("[DEBUG] Accepting this sentence.")
            accepted_text = candidate_text
            old_sbert_score = new_sbert_score
            old_tfidf_score = new_tfidf_score
            old_zsc_score = new_zsc_score
        else:
            print("[DEBUG] Rejecting this sentence (not enough votes).")

    print("\n[DEBUG] Final Accepted Text:")
    print(accepted_text)
    return accepted_text


if __name__ == "__main__":
    # Example usage
    question = (
        "Explain how the Agile process can be applied to a mobile app project "
        "focused on public transport, bike-sharing, and ride-hailing."
    )
    student_answer = """
        Agile is a software philosophy with iterative development. 
        I enjoy pizza with extra cheese on weekends. 
        With frequent feedback, the team can adapt quickly. 
        Also, I recently watched a great movie about dinosaurs!
        Pair programming and sprints are essential for success.
    """

    print("[DEBUG] Example question:\n", question)
    print("[DEBUG] Example answer:\n", student_answer)

    filtered_text = filter_irrelevant_content(student_answer, question)
    print("\n[DEBUG] FINAL OUTPUT:")
    print(filtered_text)
