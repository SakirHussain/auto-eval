import streamlit as st
import tempfile
from proactive_chain_of_thought import evaluate_answer
from rag_text_OP import rag_generate

# # ---- Initialize LLM ----
# model = OllamaLLM(model="deepseek-r1:7b", temperature=0.7)

# # ---- Load NLP Model for Thematic Similarity ----
# nlp = spacy.load("en_core_web_md")
   
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
