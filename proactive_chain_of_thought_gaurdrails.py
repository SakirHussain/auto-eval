from typing import Optional, Dict, Any, List, Union
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

from langchain_ollama import OllamaLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Local imports
import config
from prompts import ITERATIVE_PROMPT_TEMPLATE, PROCOT_PROMPT_TEMPLATE
from student_answer_noncollab_filtering import filter_irrelevant_content
from answer_clustering import StudentAnswerClustering

# --- Pydantic Models for Structured Output ---
class IterativeRefinementOutput(BaseModel):
    thought_process: str = Field(description="Reasoning behind the refined score.")
    refined_score: float = Field(description="Final refined score after reconsideration.")

class ProCoTOutput(BaseModel):
    evaluation_method: str = Field(description="Evaluation can be Clarification Dialogue, Target-Guided Dialogue, or Non-Collaborative Dialogue, Choose one based on the prompt and explain why and how")
    thought_process: str = Field(description="Reasoning before selecting an action.")
    action_taken: str = Field(description="Chosen action based on evaluation.")
    response: str = Field(description="Generated feedback with deductions or awards.")
    final_adjusted_score: float = Field(description="Final adjusted score after refinements.")

# --- Model and Parser Initialization ---
sbert_model = SentenceTransformer(config.EMBEDDING_MODEL)
model = OllamaLLM(model=config.OLLAMA_MODEL, temperature=0.1)

iterative_parser = PydanticOutputParser(pydantic_object=IterativeRefinementOutput)
procot_parser = PydanticOutputParser(pydantic_object=ProCoTOutput)

# Inject format instructions into prompts
ITERATIVE_PROMPT_TEMPLATE.partial_variables["format_instructions"] = iterative_parser.get_format_instructions()
PROCOT_PROMPT_TEMPLATE.partial_variables["format_instructions"] = procot_parser.get_format_instructions()

# --- Core Functions ---
def run_iterative_refinement(previous_score, question, student_answer, ideal_answer, rubric_item, max_score):
    history = []
    score = previous_score

    # Setup conversation chain with memory
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=model, memory=memory, verbose=True)

    print("\n[DEBUG] Starting Iterative Refinement Loop with Memory")

    for i in range(config.ITERATIVE_REFINEMENT_TURNS):
        print(f"[DEBUG] Iteration {i+1} ------------------------------")
        
        prompt_input = ITERATIVE_PROMPT_TEMPLATE.format(
            previous_score=score,
            question=question,
            student_answer=student_answer,
            ideal_answer=ideal_answer,
            rubric_item=rubric_item,
            max_score=max_score
        )

        try:
            # Use the conversation chain to get the raw output
            raw_output = conversation.predict(input=prompt_input)
            parsed_output = iterative_parser.parse(raw_output)

            refined_score = parsed_output.refined_score
            reason = parsed_output.thought_process

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
    return score, history

def compute_thematic_similarity(student_answer: str, ideal_answer: str):
    text_emb = sbert_model.encode([student_answer], convert_to_numpy=True)[0]
    q_emb = sbert_model.encode([ideal_answer], convert_to_numpy=True)[0]
    return float(cosine_similarity(text_emb.reshape(1, -1), q_emb.reshape(1, -1))[0][0])

def compute_tfidf_similarity(student_answer: str, ideal_answer: str):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([student_answer, ideal_answer])
    return float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])

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
) -> Optional[ProCoTOutput]:
    thematic_sim = compute_thematic_similarity(student_answer, ideal_answer) if dialogue_type == "Target-Guided Dialogue" else "N/A"
    tfidf_sim = compute_tfidf_similarity(student_answer, ideal_answer) if dialogue_type == "Target-Guided Dialogue" else "N/A"

    chain = PROCOT_PROMPT_TEMPLATE | model | procot_parser

    try:
        validated_output = chain.invoke({
            "dialogue_type": dialogue_type,
            "dialogue_desc": dialogue_desc,
            "question": question,
            "student_answer": student_answer,
            "ideal_answer": ideal_answer,
            "rubric": rubric,
            "conversation_history": conversation_history,
            "available_actions": available_actions,
            "thematic_sim": thematic_sim,
            "tfidf_sim": tfidf_sim,
            "max_score": max_marks
        })
        print(f"\n[VALIDATED JSON - {dialogue_type}]")
        print(validated_output)
        return validated_output
    except Exception as e:
        print(f"\nLangChain Pydantic Parser failed for {dialogue_type}: {e}")
        return None

def evaluate_answer_by_rubric_items(
    question: str,
    ideal_answer: Union[List[str], Dict[str, str]],
    student_answer: Union[List[str], Dict[str, str]],
    rubric_items: List[Dict[str, str]]
) -> Dict[str, Any]:
    # ... (rest of the function remains the same, just the call to generate_structured_eval will now return a Pydantic object)
    # ... (and the call to run_iterative_refinement is also updated)
    pass # Placeholder for the rest of the function

# The rest of the file (including the __main__ block) needs to be updated to use the new function signatures and config values.


