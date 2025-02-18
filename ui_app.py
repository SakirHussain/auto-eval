import streamlit as st
import json
import tempfile
import re
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# ---- Initialize LLM ----
model = OllamaLLM(model="deepseek-r1:7b", temperature=0.7)

# ---- Load NLP Model for Thematic Similarity ----
nlp = spacy.load("en_core_web_md")

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

def compute_thematic_similarity(student_answer, ideal_answer):
    """Computes thematic similarity between student answer and ideal answer."""
    student_doc = nlp(student_answer)
    ideal_doc = nlp(ideal_answer)
    return student_doc.similarity(ideal_doc)

def compute_tfidf_similarity(student_answer, ideal_answer):
    """Computes TF-IDF similarity between student answer and ideal answer."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([student_answer, ideal_answer])
    return cosine_similarity(vectors)[0, 1]

class ProCoTOutput(BaseModel):
    thought_process: str = Field(..., description="Reasoning before selecting an action.")
    action_taken: str = Field(..., description="Chosen action based on evaluation.")
    response: str = Field(..., description="Generated feedback with deductions or awards.")
    final_adjusted_score: float = Field(..., description="Final adjusted score after refinements.")

def evaluate_answer(question, student_answer, ideal_answer, rubric):
    """Evaluates the student answer using ProCoT-based structured evaluation."""
    thematic_sim = compute_thematic_similarity(student_answer, ideal_answer)
    tfidf_sim = compute_tfidf_similarity(student_answer, ideal_answer)

    parser = PydanticOutputParser(pydantic_object=ProCoTOutput)
    
    prompt_template = PromptTemplate(
        template="""
        You are a professor evaluating a student's answer. Evaluate the response using the rubric while ensuring fairness.

        Evaluation Criteria:
        - Question: {question}
        - Student Answer: {student_answer}
        - Ideal Answer: {ideal_answer}
        - Rubric: {rubric}
        - Thematic Similarity: {thematic_sim}
        - TF-IDF Similarity: {tfidf_sim}

        Generate a structured evaluation using the ProCoT framework.

        Response Format (Strict JSON):
        {format_instructions}
        """,
        input_variables=["question", "student_answer", "ideal_answer", "rubric", "thematic_sim", "tfidf_sim"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    llm_response = model.invoke(prompt_template.format(
        question=question,
        student_answer=student_answer,
        ideal_answer=ideal_answer,
        rubric=rubric,
        thematic_sim=thematic_sim,
        tfidf_sim=tfidf_sim
    ))

    result = parser.parse(remove_think_tags(llm_response))
    return result

# ---- Streamlit UI ----
st.title("📚 RAG + ProCoT Evaluation System")

st.sidebar.header("Upload PDF Corpus")
uploaded_file = st.sidebar.file_uploader("Upload your knowledge corpus (PDF)", type=["pdf"])

st.sidebar.header("Define Question & Rubric")
question = st.sidebar.text_area("Enter your question:")
rubric = st.sidebar.text_area("Enter the grading rubric:")

if st.sidebar.button("Generate Ideal Answer"):
    if uploaded_file and question and rubric:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        with st.spinner("Processing Ideal Answer... ⏳"):
            ideal_answer = rag_generate(question, rubric, temp_path)
        
        st.subheader("Generated Ideal Answer:")
        st.write(ideal_answer)

        st.session_state["ideal_answer"] = ideal_answer  # Store for evaluation

if "ideal_answer" in st.session_state:
    st.sidebar.header("Student Answer Evaluation")
    student_answer = st.text_area("Enter Student Answer:")

    if st.sidebar.button("Evaluate Answer"):
        with st.spinner("Evaluating Student Answer... ⏳"):
            evaluation = evaluate_answer(question, student_answer, st.session_state["ideal_answer"], rubric)
        
        st.subheader("Evaluation Feedback:")
        st.json(evaluation)
