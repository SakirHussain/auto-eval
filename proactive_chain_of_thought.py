import re
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize DeepSeek R1 model
model = OllamaLLM(model="deepseek-r1:7b", temperature=0.45)

# Load NLP Model
nlp = spacy.load("en_core_web_md")


# Define structured Pydantic output
class ProCoTOutput(BaseModel):
    thought_process: str = Field(..., description="Reasoning before selecting an action.")
    action_taken: str = Field(..., description="Chosen action based on evaluation.")
    response: str = Field(..., description="Generated feedback with deductions or awards.")
    final_adjusted_score: float = Field(..., description="Final adjusted score after refinements.")


# Function to clean and parse LLM responses
def safe_parse(parser, llm_response):
    """Safely parses the LLM response, ensuring strict JSON format compliance."""
    
    # Remove <think> tags
    llm_response_cleaned = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
    print(f"\n--- Cleaned LLM Response ---\n{llm_response_cleaned}")

    # # Validate response format before parsing
    # if not llm_response_cleaned.startswith("{") or not llm_response_cleaned.endswith("}"):
    #     print("\n--- ERROR: LLM Response is not valid JSON! Retrying... ---")
    #     return None  # Retry instead of crashing

    # Try parsing the cleaned response
    try:
        return parser.parse(llm_response_cleaned)
    except OutputParserException as e:
        print("\n--- Parsing Failed ---")
        print(f"Error: {e}\nRaw Cleaned LLM Response:\n{llm_response_cleaned}")
        print("\nRetrying with stricter format enforcement...")
        return None  # Indicating failure


# Function to generate and parse LLM responses using ProCoT
def evaluate_dialogue(dialogue_type, dialgoue_desc ,question, student_answer, ideal_answer, rubric, conversation_history, available_actions):
    """Evaluates student answers using ProCoT with a structured prompt and output parser."""
    
    parser = PydanticOutputParser(pydantic_object=ProCoTOutput)
    
    prompt = PromptTemplate(
        template="""
        You are a professor evaluating a student's answer. You need to fairly evaluete the student's answer based on the rubric. 
        The max marks possible for this question is 5. You need to decide how many marks to award or deduct based on the rubric provided.
        
        The question, student answer, ideal answer and rubric are provided within the respective <question>, <student_answer>, <ideal_answer> and <rubric> tags.
        
        Your task is to analyze the response using the {dialogue_type} evaluation approach.        
        Description of evaluation approach : 
        {dialogue_desc}

        Based on these given input, choose an action from the available actions.
        Given:
        - D (Task Background): "You are a teacher grading a student's answer based on the rubric."
        - C (Conversation History): "{conversation_history}"
        - A (Available Actions): {available_actions}
                
        Any addtion or deduction of marks must be based on whether the rubrics is satisfied or not. No information must be assumed or added. Only infer from that which is provided
        
        Ensure your response is a JSON object exactly in this format:
        '''json
            "thought_process": "Reasoning before selecting an action.",
            "action_taken": "Chosen action based on evaluation.",
            "response": "Generated feedback with deductions or awards.",
            "final_adjusted_score": 0.0
        '''
        
        {format_instructions}

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
        """,
        input_variables=["dialogue_type","dialgoue_desc", "question", "student_answer", "ideal_answer", "rubric", "conversation_history", "available_actions"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    print(f"\n--- Running {dialogue_type} Evaluation ---")
    formatted_prompt = prompt.format(
        dialogue_type=dialogue_type,
        dialogue_desc=dialgoue_desc,
        question=question,
        student_answer=student_answer,
        ideal_answer=ideal_answer,
        rubric=rubric,
        conversation_history=conversation_history,
        available_actions=available_actions,
    )

    print(f"\n--- Prompt ---\n")
    print(formatted_prompt)
    
    llm_response = model.invoke(formatted_prompt)
    print(f"\n--- Raw LLM Response ---\n{llm_response}")

    result = safe_parse(parser, llm_response)

    if not result:
        print("\n--- ERROR: Invalid response format. Skipping turn. ---")

    print(f"\n--- Evaluation Result ---")
    print(result)

    # Update conversation history for next turn
    # conversation_history.append({
    #     "thought_process": result.thought_process,
    #     "action_taken": result.action_taken,
    #     "response": result.response
    # })

    # Adjust score dynamically
    # score_adjustment = 0
    # if "Award" in result.action_taken:
    #     score_adjustment = float(result.action_taken.split("+")[-1])
    # elif "Deduct" in result.action_taken:
    #     score_adjustment = float(result.action_taken.split("-")[-1]) * -1

    # Apply final adjusted score refinement in the last turn
    return result


# Master function that runs all three evaluation phases sequentially
def evaluate_answer(question, student_answer, ideal_answer, rubric):
    conversation_history = []

    clarification_actions = ["Deduct marks", "Add marks"]
    target_guided_actions = ["Deduct marks", "Add marks"]
    non_collab_actions = ["Deduct marks", "Add marks"]
    
    clarification_desc = '''
    - Identify missing, unclear, or ambiguous details in the student's answer.
    - Deduct marks based on missing information.
    - Explain why marks were deducted.
    '''
    
    target_guided_desc = '''
    - Determine how many transformations (steps or turns) are needed to thematically convert the student's answer into the ideal answer.
    - Deduct marks based on the number of necessary transformations. To further assist you with the evaluation, the thematic similarity score and TF-IDF similarity score are provided.
    '''
    
    non_collab_desc = '''
     - Detect if the student's answer is off-topic, vague, or irrelevant.
    - Deduct marks accordingly and list the detected issues.
    '''

    clarification_result = evaluate_dialogue("Clarification Dialogue", clarification_desc, question, student_answer, ideal_answer, rubric, conversation_history, clarification_actions)
    print("\n--- Clarification Dialogue Completed ---")
    print(clarification_result)
    
    target_guided_result = evaluate_dialogue("Target-Guided Dialogue", target_guided_desc, question, student_answer, ideal_answer, rubric, conversation_history, target_guided_actions)
    print("\n--- Target-Guided Dialogue Completed ---")
    print(target_guided_result)
    
    non_collab_result = evaluate_dialogue("Non-Collaborative Dialogue", non_collab_desc,question, student_answer, ideal_answer, rubric, conversation_history, non_collab_actions)
    print("\n--- Non-Collaborative Dialogue Completed ---")
    print(non_collab_result)

    # Ensure no NoneType errors
    clarification_score = clarification_result.final_adjusted_score if clarification_result else 0
    target_guided_score = target_guided_result.final_adjusted_score if target_guided_result else 0
    non_collab_score = non_collab_result.final_adjusted_score if non_collab_result else 0

    total_adjusted_score = (clarification_score + target_guided_score + non_collab_score)/3.0

    print("\n--- Final Evaluation Summary ---")
    print(f"Total Score: {total_adjusted_score}")

    return {
        "total_score": total_adjusted_score,
        "feedback": [
            clarification_result.response if clarification_result else "Clarification dialogue failed.",
            target_guided_result.response if target_guided_result else "Target-guided dialogue failed.",
            non_collab_result.response if non_collab_result else "Non-collaborative dialogue failed."
        ]
    }


#example
question = "Compare and contrast perception model and sensor model."

student_answer = '''
The Perception Model is designed to convert sensor data into an understanding of the environmental state. It interprets sensor inputs to infer meaning by handling transformations necessary for invariant object recognition. This model focuses on processing data through various representations, such as feature space and predicates, to achieve recognition invariance. By transforming raw sensor inputs into meaningful insights, the perception model enables intelligent decision-making in various applications.

In contrast, the Sensor Model primarily captures and records raw sensor data without interpretation. Its main functionality is to collect and store sensor inputs directly, preserving them for further processing or storage. Unlike the perception model, which processes data to extract meaning, the sensor model serves as the foundational layer, ensuring that raw data is available for subsequent analysis and interpretation.
'''

ideal_answer = '''
The Perception Model determines the state of the system based on sensor inputs. It processes raw sensory data (e.g., camera images, LiDAR readings, or touch sensors) and converts them into meaningful information about the robot’s environment and internal state. For Example - A self-driving car uses camera and LiDAR data to identify objects such as pedestrians and vehicles. The perception model processes these inputs to estimate the car’s surroundings and potential obstacles. The Sensor Model is the inverse of the Perception Model. Given a known system state, it predicts what the sensor readings should be. This is useful in localization, state estimation, and sensor fusion techniques. For Example - In robot localization, if the robot's position in a mapped environment is known, the sensor model can predict what the LiDAR or camera should detect from that location. 
'''

rubric = '''
Perception Model : When given a sensor input, what is the state of the system → Perception Model (1 mark)
Example (1.5 marks)
Sensor Model : Inverse of Perception model. Given state of system what is the sensor input (1 mark)
Example (1.5 marks)
'''

evaluation_result = evaluate_answer(question, student_answer, ideal_answer, rubric)