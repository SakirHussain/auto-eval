import json
import re
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field
from typing import Dict, Any
from langchain_community.document_loaders import PyPDFLoader

# Step 1: Initialize Model
model = OllamaLLM(model="deepseek-r1:7b", temperature=0.7)

# Step 2: Define Schema for Answer Generation with Dynamic Rubrics
class GeneratedAnswerSchema(BaseModel):
    answer: Dict[str, str] = Field(..., description="Generated structured answer with explanation categories.")

parser = PydanticOutputParser(pydantic_object=GeneratedAnswerSchema)

# Step 3: Define Utility Functions

def clean_text(text):
    """Custom function to clean and normalize text."""
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def remove_think_tags(text):
    """Removes <think> tags and their content."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def sanitize_json_response(response):
    """Cleans the LLM response before JSON parsing to remove artifacts and formatting issues."""
    response = response.strip()  # Remove leading/trailing spaces
    response = re.sub(r"```json\s*", "", response)  # Remove Markdown JSON formatting
    response = re.sub(r"```", "", response)  # Remove stray backticks
    response = re.sub(r"\n\s*", " ", response)  # Remove excessive newlines & spaces
    response = re.sub(r"\*\*(.*?)\*\*", r"\1", response)  # Remove Markdown bold (**text**)
    response = re.sub(r"\\n", " ", response)  # Ensure newlines are safely handled
    response = response.replace("\n", " ")  # Remove remaining unintended newlines
    return response

def flatten_dict(d, parent_key='', sep=' - '):
    """Recursively flattens a nested dictionary into a single-level dictionary with concatenated keys."""
    flattened = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            flattened.update(flatten_dict(v, new_key, sep))  # Recursively flatten dicts
        elif isinstance(v, list):
            flattened[new_key] = ", ".join([str(item) for item in v])  # Convert lists to strings
        elif isinstance(v, bool):  
            flattened[new_key] = str(v)  # Convert boolean (True/False) to string
        else:
            flattened[new_key] = str(v)  # Convert everything else to string

    return flattened

def load_and_clean_corpus(file_path):
    """Load and clean documents from a PDF file."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    cleaned_documents = [clean_text(doc.page_content) for doc in documents]
    return cleaned_documents

def create_vector_store(corpus_path):
    """Create a FAISS vector store with cleaned documents."""
    documents = load_and_clean_corpus(corpus_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", ".", " "])
    docs = text_splitter.create_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def retrieve_relevant_docs(question, vector_store):
    """Retrieve the top relevant documents from the vector store."""
    relevant_docs = vector_store.similarity_search(question, k=5)
    return relevant_docs[:3]

# Step 4: Define the RAG Chain with CoT for Answer Generation with Rubric Enforcement
def rag_generate(question, rubric, corpus_path):
    """Execute RAG with dynamic reasoning, rubric-based constraints, and CoT for answer generation."""
    vector_store = create_vector_store(corpus_path)
    relevant_docs = retrieve_relevant_docs(question, vector_store)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt_template = PromptTemplate(
        template="""
        You are an AI assistant generating an answer based on retrieved knowledge. Use a Chain-of-Thought (CoT) approach to reason through the question and extract key insights.

        Ensure the answer adheres to the following rubric, covering all required topics with the given weightage:

        <rubric>
        {rubric}
        </rubric>

        <question>
        {question}
        </question>

        <context>
        {context}
        </context>

        Generate a structured answer step by step, ensuring factual accuracy and coherence while meeting rubric criteria.
        **STRICT REQUIREMENT**: Your response MUST be valid JSON ONLY. Do NOT include any explanations, formatting, or extra text outside of JSON.
        Output must be in the following JSON format:
        {format_instructions}
        """,
        input_variables=["question", "context", "rubric"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt_template | model
    llm_response = chain.invoke({"question": question, "context": context, "rubric": rubric})

    # Debugging: Print raw response before processing
    print(f"\n--- Raw LLM Response ---\n{llm_response}")

    # Remove <think> tags and sanitize JSON formatting
    cleaned_response = remove_think_tags(llm_response)
    cleaned_response = sanitize_json_response(cleaned_response)

    # Debugging: Print cleaned response before JSON parsing
    print(f"\n--- Cleaned JSON Response ---\n{cleaned_response}")

    # Ensure valid JSON parsing with full flattening of nested dictionaries and lists
    try:
        parsed_response = json.loads(cleaned_response)
        print("\n--- Successfully Parsed JSON ---")  # Debugging

        # Debugging: Check if `additionalProperties` exists
        if "additionalProperties" in parsed_response["answer"]:
            print("\n--- Detected `additionalProperties` ---")
            print(parsed_response["answer"]["additionalProperties"])

        # Remove problematic `additionalProperties` key if it's not a valid dictionary
        if "additionalProperties" in parsed_response["answer"]:
            if not isinstance(parsed_response["answer"]["additionalProperties"], dict):
                del parsed_response["answer"]["additionalProperties"]

        # Flatten remaining dictionary
        # Only flatten if `answer` is a dictionary
        if isinstance(parsed_response["answer"], dict):
            parsed_response["answer"] = flatten_dict(parsed_response["answer"])


        # Parse with Pydantic
        return parser.parse(json.dumps(parsed_response))  # Convert back to JSON string before Pydantic parsing

    except (OutputParserException, ValueError, json.JSONDecodeError) as e:
        print("\n--- JSON Parsing Failed ---")
        print(f"Error: {e}\nRaw Response:\n{cleaned_response}")
        return {"error": "Failed to parse valid JSON response."}
        
# Accept input from JSON file
if __name__ == "__main__":
    with open("rubrics.json", "r") as file:
        data = json.load(file)
    
    question = data["question"]
    rubric = data["rubric"]  # Dynamically accepts rubric with topics and weightage
    corpus_path = data["corpus_path"]
    
    generated_answer = rag_generate(question, rubric, corpus_path)
    print(generated_answer)
