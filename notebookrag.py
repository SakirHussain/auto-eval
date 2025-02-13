from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# Import additional modules for PDF and text processing
from langchain_community.document_loaders import PyPDFLoader
import re

# Import prompt templates and questions
from prompt_templates import BasicPromptTemplate, DetailedExplanationPromptTemplate
from questions import BasicQuestions, AdvancedQuestions

# Step 1: Initialize the model
model = OllamaLLM(model="deepseek-r1:7b")

# Step 2: Define utility functions for document cleaning and chunking

def clean_text(text):
    """Custom function to clean and normalize text."""
    # Remove page numbers like "Page X of Y"
    text = re.sub(r"Page \d+ of \d+", "", text)
    # Remove excessive whitespace and newlines
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Additional cleaning can be added here if needed
    return text

def load_and_clean_corpus(file_path):
    """Load and clean documents from a PDF file."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Apply cleaning to each document's page content
    cleaned_documents = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        cleaned_documents.append(cleaned_content)

    # Debug: Print cleaned documents to verify
    print("\n--- Cleaned Documents ---")
    for doc in cleaned_documents:
        print(doc[:500], "\n--- Document End ---\n")

    return cleaned_documents

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
    for doc in docs:
        print(f"Chunk ID: {doc.metadata['chunk_id']}\nContent: {doc.page_content}\n")

    # Use HuggingFace embeddings for flexibility
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create and return the vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def retrieve_relevant_docs(question, vector_store):
    """Retrieve the top relevant documents from the vector store."""
    relevant_docs = vector_store.similarity_search(question, k=5)

    # Apply filtering based on relevance if needed
    filtered_docs = [doc for doc in relevant_docs if len(doc.page_content) > 100]

    # Debug: Print retrieved chunks
    print("\n--- Retrieved Chunks ---")
    for doc in filtered_docs:
        print(f"Retrieved Chunk: {doc.page_content}\n")

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
        corpus_path=r"C:\Users\KIRTI\Downloads\chap 1 cog.pdf"
        )