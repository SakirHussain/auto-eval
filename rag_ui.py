import streamlit as st
import json
import os
import tempfile
import re
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ---- Initialize LLM ----
model = OllamaLLM(model="deepseek-r1:7b", temperature=0.7)

# ---- Utility Functions ----
def clean_text(text):
    """Removes unnecessary formatting from text."""
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def remove_think_tags(text):
    """Removes <think> tags and their content."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def load_and_clean_corpus(file_path):
    """Loads and cleans PDF content."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return [clean_text(doc.page_content) for doc in documents]

def create_vector_store(corpus_path):
    """Creates a FAISS vector store from the PDF corpus."""
    documents = load_and_clean_corpus(corpus_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", ".", " "])
    docs = text_splitter.create_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def retrieve_relevant_docs(question, vector_store):
    """Retrieves relevant document chunks for the given question."""
    relevant_docs = vector_store.similarity_search(question, k=5)
    return relevant_docs[:3]

def rag_generate(question, rubric, corpus_path):
    """Executes RAG with CoT and rubric enforcement for paragraph-based answer generation."""
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

# ---- Streamlit UI ----
st.title("📚 RAG + Chain-of-Thought Answer Generator")

st.sidebar.header("Upload PDF Corpus")
uploaded_file = st.sidebar.file_uploader("Upload your knowledge corpus (PDF)", type=["pdf"])

st.sidebar.header("Define Question & Rubric")
question = st.sidebar.text_area("Enter your question:")
rubric = st.sidebar.text_area("Enter the grading rubric:")

if st.sidebar.button("Generate Answer"):
    if uploaded_file and question and rubric:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        with st.spinner("Processing... ⏳"):
            result = rag_generate(question, rubric, temp_path)
        
        st.subheader("Generated Answer:")
        st.write(result)
    else:
        st.sidebar.error("Please upload a PDF, enter a question, and provide a rubric.")
