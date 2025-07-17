from typing import Optional, Dict, Any, List, Union
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from langchain_ollama import OllamaLLM
from guardrails import Guard
from langchain.prompts import PromptTemplate

# Import the filtering function
from student_answer_noncollab_filtering import filter_irrelevant_content
from answer_clustering import StudentAnswerClustering

sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = OllamaLLM(model="gemma3:latest", temperature=0.1)

ITERATIVE_RAIL = """
<rail version="0.1">
<output>
  <object>
    <property name="thought_process" type="string" description="Reasoning behind the refined score."/>
    <property name="refined_score" type="number" description="Final refined score after reconsideration."/>
    <required name="thought_process"/>
    <required name="refined_score"/>
  </object>
</output>
</rail>
"""
iterative_guard = Guard.from_rail_string(ITERATIVE_RAIL)



# === Iterative refinement template ===
ITERATIVE_PROMPT_TEMPLATE = PromptTemplate(
    template="""
    You are evaluating a student's answer based on an academic rubric. You previously scored this answer as {previous_score}/{max_score}, where the {max_score} is defined as full credit while 0 is defined as no credit for that particular rubric alone.
    The current rubric item being evaluated is also provided. You must strictly evaluate the student's answer only considering the rubric item.
    
    Here is the information provided:

    Question:
    {question}

    Student Answer:
    {student_answer}
    
    Rubric Item:
    {rubric_item}

    Ideal Answer (reference answer for this evaluation):
    {ideal_answer}

    Given this information, reconsider if the current score of {previous_score} accurately reflects the student's answer. Provide detailed reasoning and a refined score out of {max_score}.

    Respond strictly in this JSON format:
    {{
        "thought_process": "Explain your reasoning clearly here.",
        "refined_score": "Final refined numerical score out of {max_score}"
    }}
""",
    input_variables=["previous_score", "question", "student_answer", "ideal_answer", "rubric_item", "max_score"],
)

def run_iterative_refinement(previous_score, question, student_answer, ideal_answer, rubric_item, max_score):
    history = []
    score = previous_score

    print("\n[DEBUG] Starting Iterative Refinement Loop")
    print(f"[DEBUG] Initial Score: {score}\n")

    for i in range(3):
        print(f"[DEBUG] Iteration {i+1} ------------------------------")
        print(f"[DEBUG] Current Score: {score}")
        print(f"[DEBUG] Question: {question}")
        print(f"[DEBUG] Student Answer: {student_answer}")
        print(f"[DEBUG] Ideal Answer: {ideal_answer}")
        print(f"[DEBUG] Rubric Item: {rubric_item}")
        print(f"[DEBUG] Max Score: {max_score}")
        print(f"[DEBUG] Previous Score: {previous_score}")
        

        prompt = ITERATIVE_PROMPT_TEMPLATE.format(
            previous_score=score,
            question=question,
            student_answer=student_answer,
            ideal_answer=ideal_answer,
            rubric_item=rubric_item,
            max_score=max_score
        )

        # print(f"[DEBUG] Prompt sent to LLM:\n{prompt.strip()}\n")

        try:
            raw_output = model.invoke(prompt)
            # print(f"[DEBUG] Raw Output from LLM (Iteration {i+1}):\n{raw_output.strip()}\n")

            messages = [{"role": "user", "content": prompt}]
            outcome = iterative_guard.parse(llm_output=raw_output, messages=messages)
            parsed = outcome.validated_output

            refined_score = float(parsed.get("refined_score", score))
            reason = parsed.get("thought_process", "No reasoning provided.")

            print(f"[DEBUG] Parsed Score: {refined_score}")
            print(f"[DEBUG] Reasoning: {reason}\n")

            score = refined_score
            history.append({
                "turn": i + 1,
                "refined_score": refined_score,
                "thought_process": reason
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to parse or process LLM output at Iteration {i+1}: {e}")
            history.append({
                "turn": i+1,
                "refined_score": score,
                "thought_process": f"Error: {e}"
            })

    print("\n[DEBUG] Iterative Refinement Complete")
    print(f"[DEBUG] Final Refined Score: {score}")
    print(f"[DEBUG] Full History:\n{history}\n")

    return score, history


def compute_thematic_similarity(student_answer: str, ideal_answer: str):
    """Computes thematic similarity between student answer and ideal answer using Spacy embeddings."""
    text_emb = sbert_model.encode([student_answer], convert_to_numpy=True)[0]
    q_emb = sbert_model.encode([ideal_answer], convert_to_numpy=True)[0]
    return float(cosine_similarity(text_emb.reshape(1, -1), q_emb.reshape(1, -1))[0][0])

def compute_tfidf_similarity(student_answer: str, ideal_answer: str):
    """Computes TF-IDF similarity between student answer and ideal answer."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([student_answer, ideal_answer])
    return float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])

PROCOT_RAIL = """
<rail version="0.1">

<output>
  <object>
    <property name="evaluation_method" type="string"
      description="Evaluation can be Clarification Dialogue, Target-Guided Dialogue, or Non-Collaborative Dialogue, Choose one based on the prompt and explain why and how"/>
    <property name="thought_process" type="string" 
      description="Reasoning before selecting an action."/>
    <property name="action_taken" type="string" 
      description="Chosen action based on evaluation."/>
    <property name="response" type="string" 
      description="Generated feedback with deductions or awards."/>
    <property name="final_adjusted_score" type="number" 
      description="Final adjusted score after refinements."/>
    
    <required name="evaluation_method"/>
    <required name="thought_process"/>
    <required name="action_taken"/>
    <required name="response"/>
    <required name="final_adjusted_score"/>
  </object>
</output>

</rail>
"""

guard = Guard.from_rail_string(PROCOT_RAIL)

def guarded_llm_invoke(prompt: str, debug_label: str) -> Optional[Dict[str, Any]]:
    raw_llm_output = model.invoke(prompt)
    # print(f"\n[RAW LLM OUTPUT - {debug_label}]")
    print(raw_llm_output)

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        # Instead of storing the entire ValidationOutcome, store just the validated_output dict
        outcome = guard.parse(llm_output=raw_llm_output, messages=messages)
        print(f"\n[VALIDATED JSON - {debug_label}]")
        print(outcome.validated_output)  # Show the dict part only
        return outcome.validated_output  # <--- Return the dict
    except Exception as e:
        print(f"\nGuardrails failed to parse JSON for {debug_label}: {e}")
        return None

def generate_structured_eval(
    dialogue_type: str,
    dialogue_desc: str,
    question: str,
    student_answer: str,
    ideal_answer: str,
    rubric: str,
    conversation_history: str,
    available_actions,
    max_marks
) -> Optional[Dict[str, Any]]:
    """
    Calls the LLM to produce a valid ProCoTOutput JSON object,
    enforced by Guardrails.
    """

    # If using these similarities only for target-guided, you can conditionally compute them:
    thematic_sim = compute_thematic_similarity(student_answer, ideal_answer) if dialogue_type == "Target-Guided Dialogue" else "N/A"
    tfidf_sim = compute_tfidf_similarity(student_answer, ideal_answer) if dialogue_type == "Target-Guided Dialogue" else "N/A"

    prompt_template = PromptTemplate(
        template="""
        You are a professor evaluating a student's answer. 
        Your task is to fairly grade the student's answer and see if the given rubric is satisfied in by the students answer or not with reference to the ideal answer provided.

        Context and Role:
        - You are responsible for grading fairly and consistently based on the rubric provided.
        - Assign a final_adjusted_score between 0 and {max_score} as shown in the rubric, where {max_score} means full credit and 0 means no credit.
        - No assumptions should be made — your evaluation should strictly follow the rubric.
        - Your evaluation method is {dialogue_type}, described below:

        Evaluation Approach:
        {dialogue_desc}

        Evaluation Criteria:
        The following inputs are provided for you to assess the student's response:
        - Question: <question>{question}</question>
        - Student Answer: <student_answer>{student_answer}</student_answer>
        - Ideal Answer: <ideal_answer>{ideal_answer}</ideal_answer>
        - Rubric: <rubric>{rubric}</rubric>
        - Thematic Similarity: {thematic_sim}
        - TF-IDF Similarity: {tfidf_sim}

        Evaluation Framework (Proactive Chain of Thought)
        You must strictly follow the ProCoT framework to ensure structured grading.
        - C (Conversation History): "{conversation_history}"
        - A (Available Actions): {available_actions}

        Scoring Guidelines
        - Any addition or deduction of marks must be explicitly based on whether the rubric is satisfied.
        - Do not assume or add any external information—only infer from the provided inputs.

        Response Format (Strict JSON) :
        ```json
        {{
            "evaluation_method": "How are you choosing to evaluate this answer? Explain the method.",
            "thought_process": "Your reasoning before selecting an action.",
            "action_taken": "Chosen action based on evaluation.",
            "response": "Generated feedback with deductions or awards.",
            "final_adjusted_score": Your score out of {max_score} based on the rubric.
        }}
        ```
        No extra text outside JSON.
        """,
        input_variables=[
            "dialogue_type",
            "dialogue_desc",
            "question",
            "student_answer",
            "ideal_answer",
            "rubric",
            "conversation_history",
            "available_actions",
            "thematic_sim",
            "tfidf_sim",
            "max_score"
        ],
    )

    formatted_prompt = prompt_template.format(
        dialogue_type=dialogue_type,
        dialogue_desc=dialogue_desc,
        question=question,
        student_answer=student_answer,
        ideal_answer=ideal_answer,
        rubric=rubric,
        conversation_history=conversation_history,
        available_actions=available_actions,
        thematic_sim=thematic_sim,
        tfidf_sim=tfidf_sim,
        max_score=max_marks
    )

    validated_json = guarded_llm_invoke(formatted_prompt, debug_label=dialogue_type)
    return validated_json


from typing import Union, List, Dict, Any

def evaluate_answer_by_rubric_items(
    question: str,
    ideal_answer: Union[List[str], Dict[str, str]],
    student_answer: Union[List[str], Dict[str, str]],
    rubric_items: List[Dict[str, str]]
) -> Dict[str, Any]:
    print("input : question", question)
    print("\n")
    print("input : ideal_answer", ideal_answer)
    print("\n")
    print("input : student_answer", student_answer)
    print("\n")
    print("input : rubric_items", rubric_items)
    print("\n")
    
    print("\n[DEBUG] Starting Per-Rubric-Item Evaluation")

    conversation_history = ""
    clarification_actions = ["Deduct marks", "Add marks"]
    target_guided_actions = ["Deduct marks", "Add marks"]

    clarification_desc = """
    - Identify missing, unclear, or ambiguous details in the student's answer.
    - Deduct marks based on missing information.
    - Explain why marks were deducted.
    """

    target_guided_desc = """
    - Determine how many transformations (steps or turns) are needed to thematically convert the student's answer into the ideal answer.
    - Deduct marks based on the necessary transformations.
    - Thematic and TF-IDF similarity are provided.
    """

    results_by_item = []
    feedbacks_by_item = []
    total_score = 0.0

    n_items = len(rubric_items)
    # If passed lists, ensure lengths match
    if isinstance(ideal_answer, list) or isinstance(student_answer, list):
        assert isinstance(ideal_answer, list) and isinstance(student_answer, list), \
            "If one of ideal_answer/student_answer is a list, both must be lists"
        assert len(ideal_answer) == n_items == len(student_answer), \
            "ideal_answer and student_answer lists must have same length as rubric_items"

    for idx, rubric_def in enumerate(rubric_items):
        rubric_label = rubric_def['rubric']
        print("rubric_label = ", rubric_label)
        
        # max_marks = rubric_def['marks']       

        # pick the correct segment—dict lookup by rubric_label, or list by idx
        if isinstance(ideal_answer, dict):
            ideal_seg = ideal_answer.get(rubric_label, "")
        else:
            ideal_seg = ideal_answer[idx]

        if isinstance(student_answer, dict):
            student_seg = student_answer.get(rubric_label, "")
        else:
            student_seg = student_answer[idx]

        max_marks = ideal_answer[rubric_label]["marks"]
        print("max_marks = ", max_marks)    
        rubric_text  = f"{rubric_label} ({rubric_def['marks']} Marks)"
        print(f"\n[DEBUG] Evaluating Rubric Item #{idx+1}: {rubric_text}")
        
        print(f"[DEBUG] Original student segment:\n{student_seg}")

        # Filter only that segment
        filtered_seg = filter_irrelevant_content(student_seg, question)
        print(f"[DEBUG] Filtered student segment:\n{filtered_seg}")
        
        
        print("\n[DEBUG] Starting Clarification Dialogue Loop with :")
        print(f"[DEBUG] Ideal Segment: {ideal_seg}")
        print(f"[DEBUG] Filtered Segment: {filtered_seg}")
        print(f"[DEBUG] Rubric Text: {rubric_text}")
        # Clarification Dialogue
        clar_raw = generate_structured_eval(
            dialogue_type="Clarification Dialogue",
            dialogue_desc=clarification_desc,
            question=question,
            student_answer=filtered_seg,
            ideal_answer=ideal_seg,
            rubric=rubric_text,
            conversation_history=conversation_history,
            available_actions=clarification_actions,
            max_marks=max_marks
        )
        clar_score = float(clar_raw.get("final_adjusted_score", 0.0))
        clar_score_refined, _ = run_iterative_refinement(
            clar_score, question, filtered_seg, ideal_seg, rubric_text, max_marks
        )

        print("\n[DEBUG] Starting Target Guided Dialogue Loop with :")
        print(f"[DEBUG] Ideal Segment: {ideal_seg}")
        print(f"[DEBUG] Filtered Segment: {filtered_seg}")
        print(f"[DEBUG] Rubric Text: {rubric_text}")
        # Target-Guided Dialogue
        target_raw = generate_structured_eval(
            dialogue_type="Target-Guided Dialogue",
            dialogue_desc=target_guided_desc,
            question=question,
            student_answer=filtered_seg,
            ideal_answer=ideal_seg,
            rubric=rubric_text,
            conversation_history=conversation_history,
            available_actions=target_guided_actions,
            max_marks=max_marks
        )
        tar_score = float(target_raw.get("final_adjusted_score", 0.0))
        tar_score_refined, _ = run_iterative_refinement(
            tar_score, question, filtered_seg, ideal_seg, rubric_text, max_marks
        )

        item_score = (clar_score_refined + tar_score_refined) / 2.0
        total_score += item_score

        print(f"[DEBUG] Scores for Item #{idx+1}: Clar={clar_score_refined}, Target={tar_score_refined}, Avg={item_score:.2f}")

        results_by_item.append({
            "rubric_item": rubric_text,
            "clarification_score": clar_score_refined,
            "target_guided_score": tar_score_refined,
            "item_score": item_score,
            "clarification_json": clar_raw,
            "target_guided_json": target_raw
        })
        feedbacks_by_item.append({
            "rubric_item": rubric_text,
            "clarification_feedback": clar_raw.get("response", "N/A"),
            "target_guided_feedback": target_raw.get("response", "N/A")
        })

    print("\n[DEBUG] Evaluation Completed")
    print(f"[DEBUG] Total Score: {total_score:.2f}")

    return {
        "total_score": total_score,
        "scores_by_item": results_by_item,
        "feedback": feedbacks_by_item
    }




if __name__ == '__main__':
    # --- 1) Initialize clustering and load data
    clustering = StudentAnswerClustering(
        ideal_answers_path='qnia.json',
        student_answers_path='student_answers.json',
        threshold=0.7
    )
    clustering.load_data()
    clustering.cluster_answers()

    # --- 2) Load entire qnia.json to get question + rubrics
    with open('qnia.json', 'r') as f:
        qnia = json.load(f)

    question = qnia.get('question', '')
    rubric_items = qnia.get('rubrics', [])
    if not rubric_items:
        raise KeyError("qnia.json must include a top-level 'rubric_items' list")

    print(f"\nQuestion:\n{question}\n")
    print("Rubric Items:")
    for idx, item in enumerate(rubric_items, 1):
        print(f"{idx}. {item}")

    # --- 3) Pick a student to inspect
    student_id = 3
    lookup = clustering.get_clustered_ideal(student_id)

    if lookup.get('message', '').startswith('Unclustered'):
        print(f"\nStudent {student_id} is unclustered; manual grading needed.")
    else:
        # --- 4) Show each side's rubric breakdown
        ideal_splits = lookup['ideal_split_by_rubric']
        student_splits = lookup['student_split_by_rubric']

        print("\nIdeal Answer Rubric Breakdown:")
        for criterion, marks in ideal_splits.items():
            print(f"- {criterion}: {marks}")

        print("\nStudent Answer Rubric Breakdown:")
        for criterion, marks in student_splits.items():
            print(f"- {criterion}: {marks}")

        result = evaluate_answer_by_rubric_items(
            question=question,
            student_answer=student_splits,
            ideal_answer=ideal_splits,
            rubric_items=rubric_items
        )

        print("\n--- Evaluation Result ---")
        print(result)


