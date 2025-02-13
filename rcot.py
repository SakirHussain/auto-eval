from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import re

from few_shot_examples import FewShotExamples


# Initialize the DeepSeek R1 model
model = OllamaLLM(model="deepseek-r1:7b")


def safe_parse(parser, llm_response):
    """Safely parses the LLM response with retry on failure, removing <think> tags if present."""
    
    # Step 1: Remove <think> tags and content inside them
    llm_response_cleaned = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
    print(f"\n--- Cleaned LLM Response ---\n{llm_response_cleaned}")

    # Step 2: Try parsing the cleaned response
    try:
        return parser.parse(llm_response_cleaned)
    except OutputParserException as e:
        print("\n--- Parsing Failed ---")
        print(f"Error: {e}\nRaw Cleaned LLM Response:\n{llm_response_cleaned}")

        # Retry with additional instructions
        print("\nRetrying with additional format instructions...")
        prompt_retry = """
        Ensure the response strictly follows the JSON format below without any additional text:
        {
            "problem": "Your problem description here."
        }
        """
        print(prompt_retry)

        # Return None to indicate failure for further handling
        return None


# Define Pydantic models for structured output
class ProblemReconstructionSchema(BaseModel):
    problem: str = Field(..., description="The reconstructed problem description.")

class ConditionDecompositionSchema(BaseModel):
    conditions: list[str] = Field(..., description="A list of conditions extracted from the problem.")

class ConditionComparisonSchema(BaseModel):
    condition: str = Field(..., description="The condition being checked.")
    deducible: bool = Field(..., description="Whether the condition is deducible from the other conditions.")
    explanation: str = Field(..., description="An explanation for why the condition is or isn't deducible.")

def reconstruct_problem(response: str) -> str:
    """Reconstructs the problem using a structured output parser."""
    parser = PydanticOutputParser(pydantic_object=ProblemReconstructionSchema)
    
    few_shot_examples = FewShotExamples.problem_reconstruction_examples()  # Get examples

    # Update the prompt to be more explicit
    prompt = PromptTemplate(
        template="""
        USER:
        Give the concrete prompt (problem) that can generate this answer, specified by the 'Answer' field.
        The problem should contain all basic and necessary information and correspond to the answer.
        The problem can only ask for one result. 
        
        No information must be assumed or added. Only infer from that which is provided
        
        Here are some examples : {few_shot_examples}

        Ensure your response is a JSON object exactly in this format:
        {{
            "problem": "Your problem description here."
        }}

        {format_instructions}

        Answer: {response}
        """,
        input_variables=["response"],
        partial_variables={"format_instructions": parser.get_format_instructions(),
                           "few_shot_examples": few_shot_examples}
    )

    # Run the model and parse the response safely
    llm_response = model(prompt.format(response=response))
    print(f"\n--- LLM Response ---\n{llm_response}")
    
    result = safe_parse(parser, llm_response)
    return result.problem if result else "Error in reconstruction."


def decompose_conditions(query: str) -> list[str]:
    """Extracts conditions using a structured output parser."""
    parser = PydanticOutputParser(pydantic_object=ConditionDecompositionSchema)
    
    few_shot_examples = FewShotExamples.condition_extraction_examples()  # Get examples


    prompt = PromptTemplate(
        template="""
        Please list the conditions of the problem, as specifed by the 'Problem' field. There may be multiple conditions.
        Do not list conditions not related to calculations, but list all necessary conditions.
        The format should be a list of conditions with one condition per item.
        
        Here are some examples : {few_shot_examples}
        
        Ensure your response is a JSON object exactly in this format:
        {{
            "conditions": [
                "Condition 1",
                "Condition 2",
                ...
            ]
        }}

        {format_instructions}

        Problem: {query}
        """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions(),
                           "few_shot_examples": few_shot_examples}
    )

    llm_response = model(prompt.format(query=query))
    print(f"\n--- LLM Response ---\n{llm_response}")
    
    result = safe_parse(parser, llm_response)
    return result.conditions if result else []


def compare_condition(condition: str, condition_list: list[str]) -> dict:
    """Compares a condition against known conditions using a structured output parser."""
    parser = PydanticOutputParser(pydantic_object=ConditionComparisonSchema)
    
    few_shot_examples = FewShotExamples.condition_comparison_examples()  # Get examples

    prompt = PromptTemplate(
        template="""
        Given a candidate condition: '{condition}'
        
        Here is a condition list: '{condition_list}'
        
        From a mathematical point of view, can this candidate condition be deduced from the condition list?
        Please illustrate your reason and answer True or False.
        
        No information must be assumed or added. Only infer from that which is provided
        
        Here are some examples : {few_shot_examples}

        Ensure your response is a JSON object exactly in this format:
        {{
            "condition": "{condition}",
            "deducible": true or false,
            "explanation": "Provide a brief explanation."
        }}

        {format_instructions}
        """,
        input_variables=["condition", "condition_list"],
        partial_variables={"format_instructions": parser.get_format_instructions(),
                           "few_shot_examples": few_shot_examples}
    )

    llm_response = model(prompt.format(condition=condition, condition_list="\n".join(condition_list)))
    result = safe_parse(parser, llm_response)
    return {
        "condition": result.condition if result else condition,
        "deducible": result.deducible if result else False,
        "explanation": result.explanation if result else "Error in comparison."
    }


def fine_grained_comparison(original_query: str, student_response: str):
    """Performs fine-grained comparison between the original and reconstructed problems."""
    # Step 1: Reconstruct problem and extract conditions
    reconstructed_problem = reconstruct_problem(student_response)
    print(f"\n--- Reconstructed Problem ---\n{reconstructed_problem}")
    
    original_conditions = decompose_conditions(original_query)
    print("\n--- Original Conditions ---")
    print(original_conditions)
    
    reconstructed_conditions = decompose_conditions(reconstructed_problem)
    print("\n--- Reconstructed Conditions ---")
    print(reconstructed_conditions)

    # Step 2: Identify overlooked and hallucinated conditions
    overlooked_conditions = [cond for cond in original_conditions if cond not in reconstructed_conditions]
    hallucinated_conditions = [cond for cond in reconstructed_conditions if cond not in original_conditions]

    # Step 3: Perform condition comparisons
    comparison_results = [
        compare_condition(cond, reconstructed_conditions) for cond in original_conditions
    ]

    # Step 4: Output results
    print("\n--- Comparison Results ---")
    print("Overlooked Conditions:", overlooked_conditions)
    print("Hallucinated Conditions:", hallucinated_conditions)
    print("Condition Comparison Results:", comparison_results)

    # Step 5: Score Calculation
    score = max(0, (len(original_conditions) - len(overlooked_conditions)) / len(original_conditions) * 100)
    print(f"\nScore: {score:.2f}%")
    return {"overlooked": overlooked_conditions, "hallucinated": hallucinated_conditions, "score": score}



def evaluate_questions(questions, student_answers):
    """Evaluates student answers using structured output handling."""
    for question, answer in zip(questions, student_answers):
        print(f"\n--- Evaluating Question ---\n{question}")
        feedback = fine_grained_comparison(question, answer)
        print("Feedback:", feedback)


if __name__ == "__main__":
    example_questions = [
        "90 people are divided into groups of 9. How many groups are there?",
        "What is the sum of 5 and 7?"
    ]
    example_answers = [
        "There are 10 groups because 90 divided by 9 equals 10.",
        "The sum is 12."
    ]

    evaluate_questions(example_questions, example_answers)
