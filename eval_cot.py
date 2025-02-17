from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field
from typing import List

# Initialize Model
model = OllamaLLM(model="deepseek-r1:7b")

# Define Evaluation Schema with Pydantic
class AnswerEvaluationSchema(BaseModel):
    accuracy: int = Field(..., ge=0, le=10, description="Score for factual correctness (0-10).")
    reasoning_flow: int = Field(..., ge=0, le=10, description="Score for logical reasoning structure (0-10).")
    completeness: int = Field(..., ge=0, le=10, description="Score for covering all necessary details (0-10).")
    relevance: int = Field(..., ge=0, le=10, description="Score for staying focused on the question (0-10).")
    clarity_expression: int = Field(..., ge=0, le=10, description="Score for readability and coherence (0-10).")
    overall_score: int = Field(..., ge=0, le=55, description="Final cumulative score based on rubric (0-55).")
    feedback: List[str] = Field(..., description="Detailed feedback on strengths and weaknesses of the answer.")

# Create Output Parser
parser = PydanticOutputParser(pydantic_object=AnswerEvaluationSchema)

# Create Prompt with the Question Included
prompt = PromptTemplate(
    template="""
    You are evaluating a student's descriptive answer based on the following grading rubric:

    **Question Asked:** "{question}"

    **Evaluation Rubric:**
    - **Accuracy (0-10):** Does the answer correctly reference the relevant information related to the question?
    - **Reasoning Flow (0-10):** Is the reasoning logically structured and coherent?
    - **Completeness (0-10):** Does the answer cover all required aspects of the question?
    - **Relevance (0-10):** Does the answer stay focused on answering the question, without unnecessary information?
    - **Clarity & Expression (0-10):** Is the response well-expressed and easy to understand?
    - **Perception Model (0-2.5):** Does the answer explain how a sensor input maps to the system state? Include an example.
    - **Sensor Model (0-2.5):** Does the answer explain the inverse case, mapping system state back to sensor input? Include an example.

    **Scoring Instructions:**
     - Each category is scored from **0 to 10**.
    - The **overall score must be the sum of all five categories** (**0-55 max**). **Do NOT exceed 55.**
    - Provide **detailed feedback** explaining the score given in each category.

    **Format Your Response as JSON:**
    {format_instructions}

    **Student Answer:**  
    {student_answer}
    """,
    input_variables=["question", "student_answer"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Function to Evaluate Student Answer
def evaluate_answer_with_rubric(question: str, student_answer: str):
    """Evaluates a descriptive answer using structured rubrics with the question included."""
    
    llm_response = model.invoke(prompt.format(question=question, student_answer=student_answer))
    return parser.parse(llm_response)

# Example Usage
question = "Compare and contrast perception model and sensor model."
student_answer = """
The Perception Model determines the state of the system based on sensor inputs. It processes raw sensory data (e.g., camera images, LiDAR readings, or touch sensors) and converts them into meaningful information about the robot’s environment and internal state. For Example - A self-driving car uses camera and LiDAR data to identify objects such as pedestrians and vehicles. The perception model processes these inputs to estimate the car’s surroundings and potential obstacles. The Sensor Model is the inverse of the Perception Model. Given a known system state, it predicts what the sensor readings should be. This is useful in localization, state estimation, and sensor fusion techniques. For Example - In robot localization, if the robot's position in a mapped environment is known, the sensor model can predict what the LiDAR or camera should detect from that location.
"""
format_instructions = "Perception Model - When given a sensor input, what is the state of the system → Perception Model (1 mark). Example (1.5 marks). Sensor Model - Inverse of Perception model. Given state of system what is the sensor input (1 mark). Example (1.5 marks)"

evaluation_result = evaluate_answer_with_rubric(question, student_answer)

# Print the structured evaluation
import json
# print(json.dumps(evaluation_result, indent=2))
print(json.dumps(evaluation_result.dict(), indent=2))
