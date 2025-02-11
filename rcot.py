from langchain_ollama import OllamaLLM
from jsonformer import Jsonformer
from pydantic import BaseModel, Field

# Step 1: Initialize the DeepSeek model from Ollama
model = OllamaLLM(model="deepseek-r1:7b")

# Step 2: Define schema models for structured output
class ProblemReconstructionSchema(BaseModel):
    problem: str = Field(..., description="The reconstructed problem description.")

class ConditionDecompositionSchema(BaseModel):
    conditions: list[str] = Field(..., description="A list of conditions extracted from the problem.")

class ConditionComparisonSchema(BaseModel):
    condition: str = Field(..., description="The condition being checked.")
    deducible: bool = Field(..., description="Whether the condition is deducible from the other conditions.")
    explanation: str = Field(..., description="An explanation for why the condition is or isn't deducible.")

# Helper function to invoke Jsonformer with DeepSeek
def run_jsonformer(schema, prompt):
    """Runs Jsonformer on the provided schema and prompt."""
    jsonformer = Jsonformer(
        model=model,  # DeepSeek model
        tokenizer=None,  # No separate tokenizer needed for Ollama
        json_schema=schema,
        prompt=prompt
    )
    return jsonformer()

# Step 3: Define functions for structured tasks

def reconstruct_problem(response: str) -> str:
    """Reconstructs the problem using Jsonformer."""
    json_schema = {
        "type": "object",
        "properties": {
            "problem": {"type": "string"}
        },
        "required": ["problem"]
    }

    prompt = f"Reconstruct the problem based on this answer: {response}"
    result = run_jsonformer(json_schema, prompt)
    return result.get("problem", "")

def decompose_conditions(query: str) -> list[str]:
    """Extracts conditions using Jsonformer."""
    json_schema = {
        "type": "object",
        "properties": {
            "conditions": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["conditions"]
    }

    prompt = f"Extract all conditions from the following problem: {query}"
    result = run_jsonformer(json_schema, prompt)
    return result.get("conditions", [])

def compare_condition(condition: str, condition_list: list[str]) -> dict:
    """Compares a condition against known conditions using Jsonformer."""
    json_schema = {
        "type": "object",
        "properties": {
            "condition": {"type": "string"},
            "deducible": {"type": "boolean"},
            "explanation": {"type": "string"}
        },
        "required": ["condition", "deducible", "explanation"]
    }

    prompt = (
        f"Check if the condition '{condition}' can be deduced from the following conditions:\n"
        f"{condition_list}\nProvide the response in JSON format."
    )
    result = run_jsonformer(json_schema, prompt)
    return {
        "condition": result.get("condition"),
        "deducible": result.get("deducible"),
        "explanation": result.get("explanation")
    }

# Step 4: Implement fine-grained comparison

def fine_grained_comparison(original_query: str, student_response: str):
    """Performs fine-grained comparison between the original and reconstructed problems."""
    # Step 1: Reconstruct problem and extract conditions
    reconstructed_problem = reconstruct_problem(student_response)
    original_conditions = decompose_conditions(original_query)
    reconstructed_conditions = decompose_conditions(reconstructed_problem)

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

# Step 5: Main function to evaluate questions

def evaluate_questions(questions, student_answers):
    """Evaluates student answers using Jsonformer for structured output."""
    for question, answer in zip(questions, student_answers):
        print(f"\n--- Evaluating Question ---\n{question}")
        feedback = fine_grained_comparison(question, answer)
        print("Feedback:", feedback)

# Example usage
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
