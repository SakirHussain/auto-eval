# import nltk
from nltk.tokenize import sent_tokenize
# nltk.download('punkt', quiet=True)

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# 1) Initialize models/pipelines
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
        return 0.0
    emb1 = sbert_model.encode([text], convert_to_numpy=True)[0]
    emb2 = sbert_model.encode([question], convert_to_numpy=True)[0]
    return float(cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0])


def tfidf_similarity(text: str, question: str) -> float:
    """Compute TF-IDF-based cosine similarity for (text, question)."""
    if not text.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text, question])
    return float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])


def zero_shot_relevance(text: str, question: str) -> float:
    """Return the 'relevant' probability (0-1) using a zero-shot classifier."""
    if not text.strip():
        return 0.0
    zsc_result = zero_shot_classifier(
        sequences=text,
        candidate_labels=["relevant", "irrelevant"],
        hypothesis_template="This text is {} to the question: " + question
    )
    label_scores = dict(zip(zsc_result["labels"], zsc_result["scores"]))
    return float(label_scores.get("relevant", 0.0))


def filter_irrelevant_content(
    student_answer: str,
    question: str,
    window_size=3,
    tolerance=0.034
) -> str:
    """
    Rolling Context + Dual-Check Filtering:
    1) Tokenize student answer into sentences.
    2) Tokenize question into individual sentences.
    3) For each new sentence:
       a) Build recent_context = last N accepted sentences.
       b) Compute old TF-IDF and old ZSC scores for recent_context vs full question.
       c) Form candidate_text = recent_context + new_sentence.
       d) Compute TF-IDF for candidate_text vs question and single_sentence vs question.
       e) TF-IDF method votes YES if either sim + tolerance >= old_tfidf.
       f) Compute SBERT sim of candidate_text and single_sentence against each question sentence,
          take the maximum; vote YES if max >= sbert_threshold.
       g) Compute ZSC for candidate_text and single_sentence vs question; vote YES if sim + tolerance >= old_zsc.
       h) If at least two of three methods vote YES, accept the sentence and update old_tfidf and old_zsc.
    4) Return the joined accepted sentences.
    """
    print("[DEBUG] Rolling Context + Dual-Check Filtering.")
    print(f"[DEBUG] window_size={window_size}, tolerance={tolerance}")

    # 1) Split into sentences
    sentences = sent_tokenize(student_answer)
    question_sents = sent_tokenize(question)
    print(f"[DEBUG] Found {len(sentences)} student sentences; {len(question_sents)} question sentences.")

    accepted_sentences = []
    old_tfidf = 0.0
    old_zsc   = 0.0
    sbert_threshold = 0.80

    for idx, sent in enumerate(sentences):
        sent_str = sent.strip()
        if not sent_str:
            continue

        print(f"\n[DEBUG] Sentence {idx+1}: {sent_str}")
        # Build rolling context
        recent = accepted_sentences[-window_size:]
        context_text = " ".join(recent)

        # Compute old TF-IDF and ZSC on context vs full question
        old_tfidf = tfidf_similarity(context_text, question)
        old_zsc   = zero_shot_relevance(context_text, question)
        print(f"[DEBUG] Old TF-IDF: {old_tfidf:.4f}, Old ZSC: {old_zsc:.4f}")

        # Candidate context
        candidate_text = (context_text + " " + sent_str).strip() if context_text else sent_str

        # TF-IDF similarities
        cand_tfidf   = tfidf_similarity(candidate_text, question)
        single_tfidf = tfidf_similarity(sent_str,           question)
        tfidf_vote   = (
            (cand_tfidf + tolerance >= old_tfidf) or
            (single_tfidf + tolerance >= old_tfidf)
        )
        print(f"[DEBUG] TF-IDF -> Candidate: {cand_tfidf:.4f}, Single: {single_tfidf:.4f}, Vote: {tfidf_vote}")

        # SBERT (cosine) threshold check against each question sentence
        cand_vals   = [sbert_similarity(candidate_text, q) for q in question_sents]
        single_vals = [sbert_similarity(sent_str,      q) for q in question_sents]
        cand_max    = max(cand_vals,   default=0.0)
        single_max  = max(single_vals, default=0.0)
        sbert_vote  = (cand_max >= sbert_threshold) or (single_max >= sbert_threshold)
        print(f"[DEBUG] SBERT -> Candidate max: {cand_max:.4f}, Single max: {single_max:.4f}, Vote: {sbert_vote}")

        # Zero-shot relevance (delta-check)
        cand_zsc   = zero_shot_relevance(candidate_text, question)
        single_zsc = zero_shot_relevance(sent_str,      question)
        zsc_vote   = (
            (cand_zsc + tolerance >= old_zsc) or
            (single_zsc + tolerance >= old_zsc)
        )
        print(f"[DEBUG] ZSC -> Candidate: {cand_zsc:.4f}, Single: {single_zsc:.4f}, Vote: {zsc_vote}")

        # Majority vote
        votes = sum([tfidf_vote, sbert_vote, zsc_vote])
        print(f"[DEBUG] Votes => TF-IDF: {tfidf_vote}, SBERT: {sbert_vote}, ZSC: {zsc_vote} (Total={votes})")

        if votes >= 2:
            print("[DEBUG] Accepting this sentence.")
            accepted_sentences.append(sent_str)
            old_tfidf = cand_tfidf
            old_zsc   = cand_zsc
        else:
            print("[DEBUG] Rejecting this sentence.")

    # Return accepted text
    final_text = " ".join(accepted_sentences)
    print("\n[DEBUG] FINAL ACCEPTED TEXT:")
    print(final_text)
    return final_text


if __name__ == "__main__":
    # Example usage
    question = '''
    The city council plans to develop a mobile app to enhance urban mobility by providing residents with information on public transport, bike-sharing, and ride-hailing options. Due to changing transportation policies and user needs, the app’s requirements are evolving. With a limited budget and the need for a quick release, the council aims to roll out features in phases, starting with essential transport information and later adding real-time updates and payment integration.
    a. How will you implement the Agile process model for the above scenario ? (5 Marks)
    b. Discuss how eXtreme Programming (XP) can support the development of the mobile app.(5 Marks)
    '''
    
    student_answer = """
        Part(a) Agile is philosophy that revolves around agility in software development and customer satisfaction.
        It involves integrating the customer to be a part of the development team in order to recueve quick feedback and fast implementations.
        In the case of a mobile application in improve urban mobility, we will rely on building the application in increments. This will require the application to have high modularity.
        The modules can be as follows : bikesharing, ride hailing, proximity radar, ride selection/scheduling. But i love having pizzas on a wednesday afternoon which be pivtol in this case as well.
        The bike sharing and ride hailing modules are mainly UI based and can be developed in one sprint. The feedback can be obtained from a select group of citizens or lauch a test application in beta state to all phones.
        The core logic - proximity radar, to define how close or far awat te application must look for a ride and ride selection is all about selecting a ride for the user without clashing with other users.
        This is developed in subsequent sprint cycles and can be tested by limited area lauch to citizens to bring out all the runtime errors and bugs. Addtionally Agile is all about speed and i want more speed.

        Part(b) eXtreme progreamming relies on maily very fast development and mazimizing customer satisfaction.
        Since quick release is important along with subsequent rollouts this is a good SDLC model.
        The plannig is the first phase of the SDLC model. Here the requirements, need not be rigid or well defined or even formally defined. The requirements are communicated roughly and the production can begin. Here a ride application with public transport, bike sharing and ride hailing.
        Based on this alone, the architecture/software architecture can be obtained.
        Once the software architecture is defined for the interation, the coding/implementation begins.
        Coding is usually pair programming. The modules selected such as UI, bikesharing, ride hailing and public transport are developed.
        Once they are developed, they are tested agasint the member of the team or in this case a public jury/citizen jury is used to check the appeal of the UI.
        If it is satisfactory, the component is com pleted and implemented into the application, if not, the feedback is sent as an input for the next iteration and the process is repeated again.
    
    """

    print("[DEBUG] Example question:\n", question)
    print("[DEBUG] Example answer:\n", student_answer)

    filtered_text = filter_irrelevant_content(student_answer, question)
    print("\n[DEBUG] FINAL OUTPUT:")
    print(filtered_text)
