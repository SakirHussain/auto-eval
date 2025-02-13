from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# Import additional modules for PDF and text processing
import fitz  # PyMuPDF
import re

# Import prompt templates and questions
from prompt_templates import BasicPromptTemplate, DetailedExplanationPromptTemplate
from questions import BasicQuestions, AdvancedQuestions

# Step 1: Initialize the LLM model
model = OllamaLLM(model="deepseek-r1:7b")

# Step 2: Define text extraction, cleaning, and chunking functions

def clean_text(text):
    """Custom function to clean and normalize extracted text."""
    # Remove page numbers like "Page X of Y"
    text = re.sub(r"Page \d+ of \d+", "", text)
    # Remove excessive whitespace and newlines
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Remove unwanted special characters (optional)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Removes non-ASCII characters
    return text

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file while ignoring diagrams and images."""
    doc = fitz.open(pdf_path)
    extracted_text = []

    for page in doc:
        text = page.get_text("text")  # Extract text only
        cleaned_text = clean_text(text)
        if cleaned_text.strip():  # Ensure empty sections are ignored
            extracted_text.append(cleaned_text)

    return extracted_text

def load_and_clean_corpus(file_path):
    """Loads a PDF file, extracts text while filtering out diagrams/images, and cleans the content."""
    documents = extract_text_from_pdf(file_path)

    # Debug: Print cleaned documents to verify
    print("\n--- Cleaned Documents (Filtered) ---")
    for i, doc in enumerate(documents[:3]):  # Show first 3 sections for debugging
        print(f"Document {i+1}:\n{doc[:500]}\n--- End ---\n")

    return documents

def create_vector_store(corpus_path):
    """Create a FAISS vector store with cleaned and semantically chunked documents."""
    documents = load_and_clean_corpus(corpus_path)

    # Create semantic chunks with logical separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", ".", " "]
    )
    docs = text_splitter.create_documents(documents)

    # Add metadata for tracking sections (optional)
    for i, doc in enumerate(docs):
        doc.metadata = {"chunk_id": i + 1}

    # Debug: Print each chunk to verify splitting
    print("\n--- Document Chunks ---")
    for doc in docs[:3]:  # Show first 3 chunks for debugging
        print(f"Chunk ID: {doc.metadata['chunk_id']}\nContent: {doc.page_content[:500]}\n")

    # Use HuggingFace embeddings for better retrieval
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create and return the vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def retrieve_relevant_docs(question, vector_store):
    """Retrieve the top relevant documents from the vector store."""
    relevant_docs = vector_store.similarity_search(question, k=5)

    # Apply filtering based on content length to avoid short or irrelevant chunks
    filtered_docs = [doc for doc in relevant_docs if len(doc.page_content) > 100]

    # Debug: Print retrieved chunks
    print("\n--- Retrieved Chunks ---")
    for i, doc in enumerate(filtered_docs[:3]):  # Show first 3 relevant chunks
        print(f"Retrieved Chunk {i+1}: {doc.page_content[:500]}\n")

    return filtered_docs[:3]

# Step 3: Define the RAG chain and process

def rag_chain(prompt_class, questions_class, corpus_path):
    # Load the selected prompt and questions
    prompt = prompt_class.get_template()
    questions = questions_class.get_questions()
    vector_store = create_vector_store(corpus_path)

    # Create the LLM chain
    chain = LLMChain(llm=model, prompt=prompt)

    # Process each question
    for question in questions:
        relevant_docs = retrieve_relevant_docs(question, vector_store)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        print(f"Question: {question}")
        print("CONTEXT: ", context)

        # Generate response with the retrieved context
        response = chain.run({"question": question, "context": context})
        print(f"Response: {response}\n")

# Step 4: Run the RAG chain with enhanced document handling
if __name__ == "__main__":
    rag_chain(
        prompt_class=BasicPromptTemplate,
        questions_class=BasicQuestions,
        corpus_path="corpus.pdf"  # Handles PDF input
    )
