import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field
from typing import List

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_md")

def compute_thematic_similarity(student_answer: str, ideal_answer: str):
    """Computes semantic similarity between the student's answer and the ideal answer using word embeddings."""
    student_doc = nlp(student_answer)
    ideal_doc = nlp(ideal_answer)
    return student_doc.similarity(ideal_doc)  # Returns a similarity score (1 = identical, 0 = completely different)

def tfidf_similarity(student_answer: str, ideal_answer: str):
    """Computes TF-IDF similarity to check if the keyphrases in the student answer align with the ideal answer."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([student_answer, ideal_answer])
    return cosine_similarity(vectors)[0, 1]  # Returns similarity score (closer to 1 = better match)

# Initialize LLM (DeepSeek 7B hosted on Ollama)
model = OllamaLLM(model="deepseek-r1:7b", temperature=0.1)

# Define Pydantic models for structured outputs
class ClarificationOutput(BaseModel):
    thought_process: str = Field(..., description="Analysis of missing details or ambiguities.")
    deductions: int = Field(..., description="Total marks deducted for missing information.")
    feedback: List[str] = Field(..., description="Specific points that were missing.")

class TargetGuidedOutput(BaseModel):
    thought_process: str = Field(..., description="Analysis of how far the student answer is from the ideal answer.")
    steps_to_fix: int = Field(..., description="Number of transformations needed to reach the ideal answer.")
    deductions: int = Field(..., description="Total marks deducted based on step distance.")

class NonCollaborativeOutput(BaseModel):
    thought_process: str = Field(..., description="Analysis of irrelevant or vague content.")
    deductions: int = Field(..., description="Total marks deducted for off-topic or vague answers.")
    issues_detected: List[str] = Field(..., description="List of non-collaborative issues in the student's answer.")

class FinalEvaluation(BaseModel):
    total_score: int = Field(..., description="Final score after all deductions.")
    total_deductions: int = Field(..., description="Total marks deducted across all dialogue types.")
    feedback: List[str] = Field(..., description="Consolidated feedback for the student.")

# Safe parsing function with retry handling
def safe_parse(parser, llm_response):
    """Safely parses the LLM response, removing <think> tags and handling errors."""
    
    # Step 1: Remove <think> tags
    llm_response_cleaned = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
    print(f"\n--- Cleaned LLM Response ---\n{llm_response_cleaned}")

    # Step 2: Try parsing the cleaned response
    try:
        return parser.parse(llm_response_cleaned)
    except OutputParserException as e:
        print("\n--- Parsing Failed ---")
        print(f"Error: {e}\nRaw Cleaned LLM Response:\n{llm_response_cleaned}")
        print("\nRetrying with stricter format enforcement...")
        return None  # Indicating failure

# Function to query LLM with structured output parsing
def query_llm(prompt, output_model):
    """Send a structured prompt to the LLM and parse its response."""
    parser = PydanticOutputParser(pydantic_object=output_model)
    formatted_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"
    
    # Get LLM response
    llm_response = model.invoke(formatted_prompt)
    print(f"\n--- Raw LLM Response ---\n{llm_response}")

    return safe_parse(parser, llm_response)

# Step 1: Clarification Dialogue Evaluation
def evaluate_clarification(question, student_answer, ideal_answer, rubric):
    prompt = f"""
    **Clarification Dialogue Phase**
    - Identify missing, unclear, or ambiguous details in the student's answer.
    - Deduct marks based on missing information.
    - Explain why marks were deducted.

    <question>
    {question}
    </question>

    <student_answer>
    {student_answer}
    </student_answer>

    <ideal_answer>
    {ideal_answer}
    </ideal_answer>

    <rubric>
    {rubric}
    </rubric>
    """
    return query_llm(prompt, ClarificationOutput)

# Step 2: Target-Guided Dialogue Evaluation
def evaluate_target_guided(question, student_answer, ideal_answer):
    
    # Compute thematic similarity (Conceptual understanding)
    thematic_score = compute_thematic_similarity(student_answer, ideal_answer)
    tfidf_score = tfidf_similarity(student_answer, ideal_answer)
    
    print(f"\n--- Thematic Similarity: {thematic_score:.3f} | TF-IDF Similarity: {tfidf_score:.3f} ---")
    
    if thematic_score >= 0.85:
        thought_process = "The student's answer is thematically correct and aligns with the key concept of Newton's First Law."
        steps_to_fix = 0
        deductions = 0
    elif 0.7 <= thematic_score < 0.85:
        thought_process = "The student's answer is mostly correct but contains slight conceptual deviations or extra details."
        steps_to_fix = 1
        deductions = 1  # Minor deduction
    elif 0.5 <= thematic_score < 0.7:
        thought_process = "The student's answer conveys some correct ideas but lacks clarity or precise explanation."
        steps_to_fix = 2
        deductions = 3  # Moderate deduction
    else:
        thought_process = "The student's answer does not correctly reflect the intended concept of Newton's First Law."
        steps_to_fix = 3
        deductions = 5  # Major deduction
        
    return {
        "thought_process": thought_process,
        "steps_to_fix": steps_to_fix,
        "deductions": deductions
    }
    
    
    # prompt = f"""
    # **Target-Guided Dialogue Phase**
    # - Determine how many transformations (steps or turns) are needed to thematically convert the student’s answer into the ideal answer.
    # - Deduct marks based on the number of necessary transformations. To further assist you with the evaluation, the thematic similarity score and TF-IDF similarity score are provided.

    # <thematic_similarity>
    # {thematic_score:.3f}
    # </thematic_similarity>
    
    # <tfidf_similarity>
    # {tfidf_score:.3f}
    # </tfidf_similarity>
    
    # <question>
    # {question}
    # </question>

    # <student_answer>
    # {student_answer}
    # </student_answer>

    # <ideal_answer>
    # {ideal_answer}
    # </ideal_answer>
    # """
    # return query_llm(prompt, TargetGuidedOutput)

# Step 3: Non-Collaborative Dialogue Evaluation
def evaluate_non_collaborative(question, student_answer):
    prompt = f"""
    **Non-Collaborative Dialogue Phase**
    - Detect if the student's answer is off-topic, vague, or irrelevant.
    - Deduct marks accordingly and list the detected issues.

    <question>
    {question}
    </question>

    <student_answer>
    {student_answer}
    </student_answer>
    """
    return query_llm(prompt, NonCollaborativeOutput)

# Master function that runs all three phases sequentially
def evaluate_answer(question, student_answer, ideal_answer, rubric, max_score=10):
    print("Running Clarification Dialogue Evaluation...")
    clarification_result = evaluate_clarification(question, student_answer, ideal_answer, rubric)
    
    print("Running Target-Guided Dialogue Evaluation...")
    target_guided_result = evaluate_target_guided(question, student_answer, ideal_answer)
    print("--- Target-Guided Dialogue Result ---\n", target_guided_result)
    
    print("\nRunning Non-Collaborative Dialogue Evaluation...")
    non_collab_result = evaluate_non_collaborative(question, student_answer)

    # Calculate total deductions
    total_deductions = (
        (clarification_result.deductions if clarification_result else 0) +
        (target_guided_result["deductions"] if target_guided_result else 0) +
        (non_collab_result.deductions if non_collab_result else 0)
    )

    # Ensure score is not negative
    final_score = max(0, max_score - total_deductions)

    # Consolidate feedback
    feedback = []
    if clarification_result:
        feedback.extend(clarification_result.feedback)
    if target_guided_result:
        feedback.append(f"-{target_guided_result['deductions']} points: {target_guided_result['thought_process']}")
    if non_collab_result:
        feedback.append(f"-{non_collab_result.deductions} points: {non_collab_result.thought_process}")

    # Print final evaluation summary
    print("\n--- Final Evaluation Summary ---")
    print(f"Total Score: {final_score}/{max_score}")
    print(f"Total Deductions: {total_deductions}")
    print("Feedback:", feedback)

    return FinalEvaluation(
        total_score=final_score,
        total_deductions=total_deductions,
        feedback=feedback
    ).dict()

# Example Usage
question = "Compare and contrast perception model and sensor model"
student_answer = '''
The Perception Model is designed to convert sensor data into an understanding of the environmental state. It interprets sensor inputs to infer meaning by handling transformations necessary for invariant object recognition. This model focuses on processing data through various representations, such as feature space and predicates, to achieve recognition invariance. By transforming raw sensor inputs into meaningful insights, the perception model enables intelligent decision-making in various applications.

In contrast, the Sensor Model primarily captures and records raw sensor data without interpretation. Its main functionality is to collect and store sensor inputs directly, preserving them for further processing or storage. Unlike the perception model, which processes data to extract meaning, the sensor model serves as the foundational layer, ensuring that raw data is available for subsequent analysis and interpretation
'''
ideal_answer = ''' 
The Perception Model determines the state of the system based on sensor inputs. It processes raw sensory data (e.g., camera images, LiDAR readings, or touch sensors) and converts them into meaningful information about the robot’s environment and internal state. For Example - A self-driving car uses camera and LiDAR data to identify objects such as pedestrians and vehicles. The perception model processes these inputs to estimate the car’s surroundings and potential obstacles. The Sensor Model is the inverse of the Perception Model. Given a known system state, it predicts what the sensor readings should be. This is useful in localization, state estimation, and sensor fusion techniques. For Example - In robot localization, if the robot's position in a mapped environment is known, the sensor model can predict what the LiDAR or camera should detect from that location. 
'''

rubric = '''
Perception Model
When given a sensor input, what is the state of the system → Perception Model (1 mark)
Example (1.5 marks)
Sensor Model
Inverse of Perception model. Given state of system what is the sensor input (1 mark)
Example (1.5 marks)
'''

result = evaluate_answer(question, student_answer, ideal_answer, rubric)
print(result)
