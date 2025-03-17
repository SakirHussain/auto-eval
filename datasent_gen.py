import json
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from guardrails import Guard
from rake_nltk import Rake
import nltk

# Ensure NLTK stopwords are downloaded (if not already)
nltk.download('stopwords', quiet=True)

# Define the Guardrails rail string to enforce a list output.
RAIL_STRING = """
<rail version="0.1">
<output>
  <list name="rubrics" description="List of rubric criteria for grading student answers">
    <string description="A clear, concise rubric criterion statement ending with (1 Mark)" />
  </list>
</output>
</rail>
"""

# Initialize Guardrails with the rail string.
guard = Guard.from_rail_string(RAIL_STRING)

# Initialize the Mistral-7B model on Ollama.
model = OllamaLLM(model="mistral-7b", temperature=0.4)

def extract_key_phrases(text: str) -> list:
    """
    Use RAKE to extract key phrases from the text.
    Returns a list of key phrases.
    """
    rake_extractor = Rake()
    rake_extractor.extract_keywords_from_text(text)
    key_phrases = rake_extractor.get_ranked_phrases()
    # Return top 5 key phrases for brevity (adjust as needed)
    return key_phrases[:5]

def generate_rubric(question: str, ideal_answer: str):
    """
    Generate a list of rubric items for a given question and ideal answer.
    Steps:
      1. Extract key phrases from the ideal answer.
      2. Create a structured prompt including these key phrases.
      3. Use Mistral-7B to generate rubric items.
      4. Validate the output with Guardrails.ai.
    """
    key_phrases = extract_key_phrases(ideal_answer)
    # Construct the key points string for the prompt.
    key_points_str = ", ".join(key_phrases)
    
    # Create the prompt.
    prompt = f"""
    You are an AI assistant tasked with creating clear, concise rubric criteria for grading student answers.
    Given the following question and its ideal answer, generate a list of rubric items.
    Each rubric item must be a single, clear sentence that describes what a student answer should demonstrate, and must end with "(1 Mark)".
    Use the key points extracted from the ideal answer: {key_points_str}.

    Question:
    {question}

    Ideal Answer:
    {ideal_answer}

    Generate Rubrics:
    """
    # Invoke the model to generate rubric items.
    response = model.invoke(prompt)
    # Print raw response for debugging purposes.
    print("Raw Model Response:\n", response)
    
    # Validate and parse the output using Guardrails.
    try:
        validated = guard.parse(response)
        rubric_items = validated.validated_output["rubrics"]
    except Exception as e:
        print("Guardrails validation failed:", e)
        rubric_items = []
    return rubric_items

def load_file_lines(file_path: str):
    """Load non-empty lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def main():
    # Load questions and answers from files.
    questions = load_file_lines("questions.txt")
    answers = load_file_lines("answers.txt")
    
    if len(questions) != len(answers):
        print("Warning: Number of questions and answers do not match!")
    
    rubrics_for_all = {}
    # Iterate over each question-answer pair.
    for q, a in zip(questions, answers):
        print(f"\nGenerating rubrics for Question: {q}")
        rubric_items = generate_rubric(q, a)
        rubrics_for_all[q] = rubric_items
        if rubric_items:
            print("Rubric Items:")
            for item in rubric_items:
                print(" -", item)
        else:
            print("No valid rubric items generated.")
        print("-" * 40)
    
    # Optionally, save the generated rubrics to a JSON file.
    with open("generated_rubrics.json", "w", encoding="utf-8") as out_f:
        json.dump(rubrics_for_all, out_f, indent=2)

if __name__ == "__main__":
    main()
