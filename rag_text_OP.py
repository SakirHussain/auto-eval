import re
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.exceptions import OutputParserException
from langchain_community.document_loaders import PyPDFLoader

# Step 1: Initialize Model
model = OllamaLLM(model="deepseek-r1:7b", temperature=0.7)

# Step 2: Define Utility Functions

def clean_text(text):
    """Custom function to clean and normalize text."""
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def remove_think_tags(text):
    """Removes <think> tags and their content."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

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

# Step 3: Define the RAG Chain with CoT for Answer Generation in Paragraph Format
def rag_generate(question, rubric, corpus_path):
    """Execute RAG with dynamic reasoning, rubric-based constraints, and CoT for answer generation in paragraph format."""
    vector_store = create_vector_store(corpus_path)
    relevant_docs = retrieve_relevant_docs(question, vector_store)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt_template = PromptTemplate(
        template="""
        You are an AI assistant generating a detailed, structured answer based on retrieved knowledge. 
        Use a Chain-of-Thought (CoT) approach to reason through the question and extract key insights.
        
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
        
        Generate a detailed and descriptive answer in paragraph format. 
        The answer should include all required elements from the rubric and must be well-structured.
        The output should ONLY contain the generated answer as a full text paragraph without any additional formatting, labels, or explanations.
        """,
        input_variables=["question", "context", "rubric"]
    )
    
    chain = prompt_template | model
    llm_response = chain.invoke({"question": question, "context": context, "rubric": rubric})
    
    # Remove <think> tags before returning the final response
    cleaned_response = remove_think_tags(llm_response)
    return cleaned_response

# Accept input from JSON file
# if __name__ == "__main__":
#     with open("rubrics.json", "r") as file:
#         data = json.load(file)
    
#     question = data["question"]
#     rubric = data["rubric"]  # Dynamically accepts rubric with topics and weightage
#     corpus_path = data["corpus_path"]
    
#     generated_answer = rag_generate(question, rubric, corpus_path)
#     print(generated_answer)
