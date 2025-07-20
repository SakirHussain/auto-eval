import streamlit as st
from proactive_chain_of_thought import evaluate_answer_by_rubric_items
from graphrag import rag_generate
from softner import predict_softened_score
import config

st.title("📚 Graph RAG + ProCoT Evaluation System")

# Sidebar: Define Question & Rubric
st.sidebar.header("Define Question & Rubric")
question = st.sidebar.text_area("Enter your question:")
rubric_text = st.sidebar.text_area("Enter the grading rubric (each line a rubric item):")

# Generate Ideal Answer
if st.sidebar.button("Generate Ideal Answer"):
    if question and rubric_text:
        # Convert the multiline rubric text into a list of rubric items.
        rubric_items = [item.strip() for item in rubric_text.split('\n') if item.strip()]
        with st.spinner("Processing Ideal Answer... ⏳"):
            ideal_answer = rag_generate(question, rubric_items)
        # Store the generated ideal answer in session state.
        st.session_state["ideal_answer"] = ideal_answer
    else:
        st.error("Please enter both a question and the grading rubric.")

# Display the generated ideal answer, if available.
if "ideal_answer" in st.session_state:
    st.subheader("Generated Ideal Answer:")
    st.write(st.session_state["ideal_answer"])

    # Student Answer Input and Evaluation
    st.sidebar.header("Student Answer Evaluation")
    student_answer = st.text_area("Enter Student Answer:")

    if st.sidebar.button("Evaluate Answer"):
        if not student_answer:
            st.error("Please enter a student answer.")
        else:
            # Re-convert rubric text into list for evaluation.
            rubric_items = [item.strip() for item in rubric_text.split('\n') if item.strip()]
            with st.spinner("Evaluating Student Answer... ⏳"):
                evaluation = evaluate_answer_by_rubric_items(
                    question,
                    student_answer,
                    st.session_state["ideal_answer"],
                    rubric_items
                )
            
            softened = predict_softened_score(
                evaluation["total_score"],      # ProCoT total
                student_answer,
                st.session_state["ideal_answer"],
            )
            
            st.subheader("Evaluation Feedback:")
            st.json(evaluation)
            
            st.markdown(
                f"""
            **ProCoT Total:** `{evaluation['total_score']:.2f}`  
            **Softened Score:** `{softened}`  
            """
            )
