"""
Centralized store for all LLM prompt templates.
"""

from langchain.prompts import PromptTemplate

# --- RAG Prompt for Ideal Answer Generation ---
RAG_PROMPT_TEMPLATE = PromptTemplate(
    template="""
    You are a college professor tasked with generating a structured and comprehensive answer for a given question taking help from the retrieved knowledge.  
    The answer will be worth 10 marks in total so the answer MUST BE 200 WORDS OR MORE.

    Task Overview
    - Use Chain-of-Thought (CoT) reasoning to analyze the question step by step.
    - Extract key insights from the provided context.
    - Construct a well-structured, paragraph-based answer that directly satisfies the rubric criteria.

    <question>
    {question}
    </question>
    
    <rubric>
    {rubric_items}
    </rubric>

    Retrieved Context
    The following context has been retrieved from reliable sources.  
    Use this information to construct an accurate and detailed response to the given question:

    <context>
    {context}
    </context>

    Response Generation Guidelines
    - The response must be a fully detailed and structured answer.  
    - DO NOT include any explanations, formatting, labels, or extra text—only generate the answer.  
    - The output should be a cohesive, well-written paragraph addressing all rubric points. 
    """,
    input_variables=["context", "question", "rubric_items"]
)

# --- Iterative Refinement Prompt ---
ITERATIVE_PROMPT_TEMPLATE = PromptTemplate(
    template="""
    You are evaluating a student's answer based on an academic rubric. You previously scored this answer as {previous_score}/{max_score}, where the {max_score} is defined as full credit while 0 is defined as no credit for that particular rubric alone.
    The current rubric item being evaluated is also provided. You must strictly evaluate the student's answer only considering the rubric item.
    
    Here is the information provided:

    Question:
    {question}

    Student Answer:
    {student_answer}
    
    Rubric Item:
    {rubric_item}

    Ideal Answer (reference answer for this evaluation):
    {ideal_answer}

    Given this information, reconsider if the current score of {previous_score} accurately reflects the student's answer. Provide detailed reasoning and a refined score out of {max_score}.

    Respond strictly in this JSON format:
    {format_instructions}
""",
    input_variables=["previous_score", "question", "student_answer", "ideal_answer", "rubric_item", "max_score"],
    partial_variables={}
)

# --- ProCoT Evaluation Prompt ---
PROCOT_PROMPT_TEMPLATE = PromptTemplate(
    template="""
    You are a professor evaluating a student's answer. 
    Your task is to fairly grade the student's answer and see if the given rubric is satisfied in by the students answer or not with reference to the ideal answer provided.

    Context and Role:
    - You are responsible for grading fairly and consistently based on the rubric provided.
    - Assign a final_adjusted_score between 0 and {max_score} as shown in the rubric, where {max_score} means full credit and 0 means no credit.
    - No assumptions should be made — your evaluation should strictly follow the rubric.
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
    You must strictly follow the ProCoT framework to ensure structured grading.
    - C (Conversation History): "{conversation_history}"
    - A (Available Actions): {available_actions}

    Scoring Guidelines
    - Any addition or deduction of marks must be explicitly based on whether the rubric is satisfied.
    - Do not assume or add any external information—only infer from the provided inputs.

    Response Format (Strict JSON) :
    {format_instructions}
    No extra text outside JSON.
    """,
    input_variables=[
        "dialogue_type",
        "dialogue_desc",
        "question",
        "student_answer",
        "ideal_answer",
        "rubric",
        "conversation_history",
        "available_actions",
        "thematic_sim",
        "tfidf_sim",
        "max_score"
    ],
    partial_variables={}
)
