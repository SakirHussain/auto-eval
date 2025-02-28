import re
import spacy
import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from guardrails import Guard, OnFailAction

# Load SpaCy NLP model
print("[INFO] Loading SpaCy NLP model...")
nlp = spacy.load("en_core_web_md")
print("[INFO] SpaCy NLP model loaded.")

# Initialize LLM
print("[INFO] Initializing LLM model...")
model = OllamaLLM(model="deepseek-r1:7b", temperature=0.40)
print("[INFO] LLM model initialized.")

# Guardrails Schema
rails_schema = """
<rails version='0.1'>
    <output>
        <json>
            {
                "thought_process": "string",
                "action_taken": "string",
                "response": "string",
                "rubric_scores": {
                    "type": "object",
                    "patternProperties": {
                        ".*": { "type": "number", "minimum": 0, "maximum": 10 }
                    }
                },
                "final_adjusted_score": { "type": "number", "minimum": 0, "maximum": 10 }
            }
        </json>
    </output>
</rails>
"""

print("[INFO] Initializing Guardrails schema...")
procot_guard = Guard.from_rail_string(rails_schema)
print("[INFO] Guardrails schema initialized.")

# Class for structured output
class ProCoTOutput(BaseModel):
    thought_process: str = Field(..., description="Reasoning before selecting an action.")
    action_taken: str = Field(..., description="Chosen action based on evaluation.")
    response: str = Field(..., description="Generated feedback with deductions or awards.")
    rubric_scores: dict = Field(..., description="Score breakdown for each rubric component.")
    final_adjusted_score: float = Field(..., description="Final adjusted score after refinements.")

# Compute Thematic & TF-IDF Similarity
def compute_similarities(student_answer, ideal_answer):
    print("[INFO] Computing thematic and TF-IDF similarities...")
    student_doc = nlp(student_answer)
    ideal_doc = nlp(ideal_answer)
    thematic_sim = student_doc.similarity(ideal_doc)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([student_answer, ideal_answer])
    tfidf_sim = cosine_similarity(vectors)[0, 1]
    
    print("[INFO] Similarities computed.")
    return thematic_sim, tfidf_sim

# Safe LLM Parsing with Guardrails
def safe_parse(llm_response):
    print("[INFO] Validating and cleaning LLM response with Guardrails...")
    llm_response_cleaned = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
    llm_response_cleaned = re.sub(r'```json', '', llm_response_cleaned).strip()
    llm_response_cleaned = re.sub(r'```', '', llm_response_cleaned).strip()
    
    try:
        validated_result = procot_guard.validate(llm_response_cleaned)
        if hasattr(validated_result, 'output') and isinstance(validated_result.output, dict):
            print("[INFO] Guardrails validation successful.")
            return validated_result.output
        else:
            print("[ERROR] Guardrails validation failed. Falling back to raw JSON parsing...")
            try:
                validated_output = json.loads(llm_response_cleaned)
                print("[INFO] Successfully parsed raw JSON from LLM response.")
                return validated_output
            except json.JSONDecodeError:
                print("[ERROR] Both Guardrails validation and raw JSON parsing failed. Defaulting to empty response.")
                return {"rubric_component": "Unknown", "score": 0.0, "justification": "Validation failed, defaulting to 0."}
        if validated_result:
            print("[INFO] Guardrails validation successful.")
            return validated_result.output
        if hasattr(validated_result, 'validated_output') and isinstance(validated_result.validated_output, dict):
            validated_output = validated_result.validated_output
            print("[INFO] Guardrails validation successful.")
        else:
            print("[ERROR] Guardrails validation failed. Falling back to raw JSON parsing...")
            try:
                validated_output = json.loads(llm_response_cleaned)
                print("[INFO] Successfully parsed raw JSON from LLM response.")
            except json.JSONDecodeError:
                print("[ERROR] Both Guardrails validation and raw JSON parsing failed. Defaulting to empty response.")
                validated_output = {"rubric_component": "Unknown", "score": 0.0, "justification": "Validation failed, defaulting to 0."}
        if validated_output and isinstance(validated_output, dict):
            print("[INFO] Guardrails validation successful.")
            return validated_output
        else:
            print("[ERROR] Guardrails validation failed. Falling back to raw JSON parsing...")
            try:
                validated_result = json.loads(llm_response_cleaned)
                print("[INFO] Successfully parsed raw JSON from LLM response.")
            except json.JSONDecodeError:
                print("[ERROR] Both Guardrails validation and raw JSON parsing failed. Defaulting to empty response.")
                return {"rubric_component": "Unknown", "score": 0.0, "justification": "Validation failed, defaulting to 0."}
        
        if validated_output:
            print("[INFO] LLM response validated successfully.")
            return validated_output
        else:
            raise ValueError("Validation failed, empty output.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[ERROR] Guardrails Validation Failed: {e}")
        return {"error": "Invalid JSON format"}

# Evaluate Dialogue
def evaluate_dialogue(dialogue_type, dialogue_desc, question, student_answer, ideal_answer, rubric):
    print(f"[INFO] Evaluating dialogue: {dialogue_type}")
    thematic_sim, tfidf_sim = compute_similarities(student_answer, ideal_answer)
    rubric_components = rubric.split("\n")
    rubric_evaluations = {}
    
    for component in rubric_components:
        prompt = PromptTemplate(
            template="""
            You are a professor grading a student's response based on a strict rubric.
            - **Rubric Component**: {rubric_component}
            - **Max Marks**: 10
            - **Question**: {question}
            - **Student Answer**: {student_answer}
            - **Ideal Answer**: {ideal_answer}
            - **Thematic Similarity**: {thematic_sim:.2f}
            - **TF-IDF Similarity**: {tfidf_sim:.2f}
            
            STRICT RULES:
            - If the rubric component is NOT covered, marks MUST be deducted.
            - Do NOT assume information beyond the provided text.
            
            Output JSON format:
            ```json
            {{
                "rubric_component": "{rubric_component}",
                "score": 0.0,
                "justification": "Explanation for awarded marks or deductions."
            }}
            ```
            """,
            input_variables=["rubric_component", "question", "student_answer", "ideal_answer", "thematic_sim", "tfidf_sim"]
        )
        
        print(f"[INFO] Sending prompt to LLM for rubric component: {component}")
        llm_response = model.invoke(prompt.format(
            rubric_component=component,
            question=question,
            student_answer=student_answer,
            ideal_answer=ideal_answer,
            thematic_sim=thematic_sim,
            tfidf_sim=tfidf_sim
        ))
        print("[INFO] Received response from LLM.")
        validated_result = safe_parse(llm_response)
        rubric_evaluations[component] = validated_result if validated_result else {"rubric_component": component, "score": 0.0, "justification": "Validation failed, defaulting to 0."}
    
    final_score = sum(float(item.get("score", 0.0)) for item in rubric_evaluations.values()) / len(rubric_evaluations) if rubric_evaluations else 0.0
    
    final_output = {"rubric_scores": rubric_evaluations, "final_adjusted_score": final_score}
    print("[INFO] Evaluation completed. Displaying final output...")
    print(json.dumps(final_output, indent=4))
    return final_output

# Run Evaluation
if __name__ == "__main__":
    question = """
    Compare and contrast perception model and sensor model.
    """
    student_answer = """
    The Perception Model is designed to convert sensor data into an understanding of the environmental state. It interprets sensor inputs to infer meaning by handling transformations necessary for invariant object recognition. This model focuses on processing data through various representations, such as feature space and predicates, to achieve recognition invariance. By transforming raw sensor inputs into meaningful insights, the perception model enables intelligent decision-making in various applications.

    In contrast, the Sensor Model primarily captures and records raw sensor data without interpretation. Its main functionality is to collect and store sensor inputs directly, preserving them for further processing or storage. Unlike the perception model, which processes data to extract meaning, the sensor model serves as the foundational layer, ensuring that raw data is available for subsequent analysis and interpretation.
    """
    ideal_answer ="""
    The Perception Model determines the state of the system based on sensor inputs. It processes raw sensory data (e.g., camera images, LiDAR readings, or touch sensors) and converts them into meaningful information about the robot’s environment and internal state. For Example - A self-driving car uses camera and LiDAR data to identify objects such as pedestrians and vehicles. The perception model processes these inputs to estimate the car’s surroundings and potential obstacles. The Sensor Model is the inverse of the Perception Model. Given a known system state, it predicts what the sensor readings should be. This is useful in localization, state estimation, and sensor fusion techniques. For Example - In robot localization, if the robot's position in a mapped environment is known, the sensor model can predict what the LiDAR or camera should detect from that location.
    """
    rubric = """Explanation of Perception Model (2 marks)\nExplanation of Sensor Model (2 marks)\nReal world Example of Perception Model (3 marks)\nReal world Example of Sensor Model (3 marks)
    """
    
    evaluate_dialogue("Clarification", "Check if response is missing details", question, student_answer, ideal_answer, rubric)
