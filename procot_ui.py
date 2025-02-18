import streamlit as st
import re
import spacy
import json
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# ---- Initialize Model ----
model = OllamaLLM(model="deepseek-r1:7b", temperature=0.45)

# ---- Load NLP Model for Thematic Similarity ----
nlp = spacy.load("en_core_web_md")

# ---- Utility Functions ----
def compute_thematic_similarity(student_answer: str, ideal_answer: str):
    """Computes thematic similarity between student answer and ideal answer."""
    return nlp(student_answer).similarity(nlp(ideal_answer))

def compute_tfidf_similarity(student_answer: str, ideal_answer: str):
    """Computes TF-IDF similarity between student answer and ideal answer."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([student_answer, ideal_answer])
    return cosine_similarity(vectors)[0, 1]

def remove_think_tags(text):
    """Removes <think> tags and their content."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ---- Pydantic Schema for Structured Output ----
class ProCoTOutput(BaseModel):
    thought_process: str = Field(..., description="Reasoning before selecting an action.")
    action_taken: str = Field(..., description="Chosen action based on evaluation.")
    response: str = Field(..., description="Generated feedback with deductions or awards.")
    final_adjusted_score: float = Field(..., description="Final adjusted score after refinements.")

def evaluate_dialogue(dialogue_type, dialogue_desc, question, student_answer, ideal_answer, rubric, conversation_history, available_actions):
    """Evaluates student answers using ProCoT with a structured prompt and output parser."""
    
    thematic_sim = compute_thematic_similarity(student_answer, ideal_answer) if dialogue_type == "Target-Guided Dialogue" else "Not Calculated"
    tfidf_sim = compute_tfidf_similarity(student_answer, ideal_answer) if dialogue_type == "Target-Guided Dialogue" else "Not Calculated"
    
    parser = PydanticOutputParser(pydantic_object=ProCoTOutput)
    
    prompt = PromptTemplate(
        template="""
        You are a professor evaluating a student's answer. Your task is to fairly evaluate the student's response based on the provided rubric while ensuring strict adherence to grading criteria.

        Context and Role:
        - You are responsible for grading fairly and consistently based on the rubric provided.  
        - Max Marks Possible: 10  
        - No assumptions should be made—your evaluation should strictly follow the rubric.  
        - Your evaluation method is {dialogue_type}, described below:  

        Evaluation Approach:
        {dialogue_desc}

        Evaluation Criteria:
        The following inputs are provided for you to assess the student's response:  
        - Question: <question>{question}</question>  
        - Student Answer: <student_answer>{student_answer}</student_answer>  
        - Ideal Answer: <ideal_answer>{ideal_answer}</ideal_answer>  
        - Rubric: <rubric>{rubric}</rubric>  
        - Thematic Similarity: {thematic_sim}
        - TF-IDF Similarity: {tfidf_sim}

        Evaluation Framework (Proactive Chain of Thought)
        - D (Task Background): "You are a teacher grading a student's answer based on the rubric."  
        - C (Conversation History): "{conversation_history}"  
        - A (Available Actions): {available_actions}  

        Scoring Guidelines
        - Any addition or deduction of marks must be explicitly based on whether the rubric is satisfied.  
        - No information must be assumed or added—only infer from the provided inputs.  

        Response Format (Strict JSON)
        ```json
        {{
            "thought_process": "Your reasoning before selecting an action.",
            "action_taken": "Chosen action based on evaluation.",
            "response": "Generated feedback with deductions or awards.",
            "final_adjusted_score": 0.0
        }}
        ```
        {format_instructions}
        """,
        input_variables=["dialogue_type", "dialogue_desc", "question", "student_answer", "ideal_answer", "rubric", "conversation_history", "available_actions", "thematic_sim", "tfidf_sim"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    llm_response = model.invoke(prompt.format(
        dialogue_type=dialogue_type,
        dialogue_desc=dialogue_desc,
        question=question,
        student_answer=student_answer,
        ideal_answer=ideal_answer,
        rubric=rubric,
        conversation_history=conversation_history,
        available_actions=available_actions,
        thematic_sim=thematic_sim,
        tfidf_sim=tfidf_sim
    ))

    return parser.parse(remove_think_tags(llm_response))

def evaluate_answer(question, student_answer, ideal_answer, rubric):
    """Runs all three ProCoT evaluation phases sequentially."""
    conversation_history = []

    dialogue_steps = [
        ("Clarification Dialogue", "Identify missing details in the student's answer.", ["Deduct marks", "Add marks"]),
        ("Target-Guided Dialogue", "Determine how many transformations are needed to convert the student's answer into the ideal answer.", ["Deduct marks", "Add marks"]),
        ("Non-Collaborative Dialogue", "Detect if the student's answer is off-topic or vague.", ["Deduct marks", "Add marks"])
    ]

    final_scores = []
    feedbacks = []

    for dialogue_type, dialogue_desc, actions in dialogue_steps:
        result = evaluate_dialogue(dialogue_type, dialogue_desc, question, student_answer, ideal_answer, rubric, conversation_history, actions)
        final_scores.append(result.final_adjusted_score)
        feedbacks.append(result.response)

    total_score = sum(final_scores) / len(final_scores)

    return {"total_score": total_score, "feedback": feedbacks}

# ---- Streamlit UI ----
st.title("📚 ProCoT Answer Evaluation")

st.sidebar.header("Input Evaluation Data")
question = st.sidebar.text_area("Enter the Question:")
ideal_answer = st.sidebar.text_area("Enter the Ideal Answer:")
rubric = st.sidebar.text_area("Enter the Rubric:")

st.header("Student Answer Input")
student_answer = st.text_area("Enter Student Answer:")

if st.button("Evaluate Answer"):
    if question and ideal_answer and rubric and student_answer:
        with st.spinner("Evaluating Student Answer... ⏳"):
            evaluation = evaluate_answer(question, student_answer, ideal_answer, rubric)

        st.subheader("Evaluation Feedback:")
        st.json(evaluation)
    else:
        st.error("Please enter the Question, Ideal Answer, Rubric, and Student Answer.")

