import json
import re
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM

def clarification_eval_tool(input_str: str) -> str:
    """
    A tool that:
     1) parses the single JSON string input to get question, student_answer, ideal_answer, rubric
     2) calls an LLM to evaluate the student's answer
     3) returns the LLM's output
    """
    print("\n[DEBUG] ClarificationEvaluator called with input:", input_str)

    # 1) Parse the JSON input
    try:
        data = json.loads(input_str)
        question = data["question"]
        student_answer = data["student_answer"]
        ideal_answer = data["ideal_answer"]
        rubric = data["rubric"]
    except Exception as e:
        return f"Error: Could not parse input JSON. {e}"

    # 2) Make an LLM call to evaluate the answer 
    # (In real usage, you'd build a more elaborate prompt.)
    prompt = f"""
    You are a Clarification Evaluator for a student's answer.
    Question: {question}
    Student Answer: {student_answer}
    Ideal Answer: {ideal_answer}
    Rubric: {rubric}

    Please clarify what is missing from the student's answer 
    based on the rubric. Provide a final short JSON snippet, e.g.:

    {{
    "missing_details": "...",
    "score": 3
    }}
    """

    llm = OllamaLLM(model="mistral", temperature=0)
    llm_output = llm.invoke(prompt)
    print("[DEBUG] LLM output from the ClarificationEvaluator prompt:", llm_output)

    # 3) Return LLM's text as the result. 
    # The agent will interpret this as "Final Answer" from the tool.
    return llm_output


def target_guided_eval_tool(input_str: str) -> str:
    """
    Similar approach: parse the JSON input, call an LLM to evaluate transformations,
    and return the LLM's response. 
    """
    print("\n[DEBUG] TargetGuidedEvaluator called with input:", input_str)

    # 1) Parse JSON
    try:
        data = json.loads(input_str)
        question = data["question"]
        student_answer = data["student_answer"]
        ideal_answer = data["ideal_answer"]
        rubric = data["rubric"]
    except Exception as e:
        return f"Error: Could not parse input JSON. {e}"

    # 2) LLM call
    prompt = f"""
    You are a Target-Guided Evaluator.
    Question: {question}
    Student Answer: {student_answer}
    Ideal Answer: {ideal_answer}
    Rubric: {rubric}

    Determine how many transformations or changes are needed 
    to make the student's answer match the ideal. 
    Return a final short JSON snippet, e.g.:

    {{
    "transformations_needed": 1,
    "explanation": "Student needs to mention 'at least once' explicitly"
    }}
    """

    llm = OllamaLLM(model="mistral", temperature=0)
    llm_output = llm.invoke(prompt)
    print("[DEBUG] LLM output from the TargetGuidedEvaluator prompt:", llm_output)

    # 3) Return 
    return llm_output


# -----------------------------------------------------------------
# Wrap these LLM-calling functions as Tools for the ReAct agent
# -----------------------------------------------------------------
clar_tool = Tool(
    name="ClarificationEvaluator",
    func=clarification_eval_tool,
    description=(
        "Use this tool to clarify what's missing. "
        "Input must be a JSON string with question, student_answer, ideal_answer, rubric."
    ),
)

tgt_tool = Tool(
    name="TargetGuidedEvaluator",
    func=target_guided_eval_tool,
    description=(
        "Use this tool to see how many transformations are needed. "
        "Input must be a JSON string with question, student_answer, ideal_answer, rubric."
    ),
)

# -----------------------------------------------------------------
# Build the agent
# -----------------------------------------------------------------
llm = OllamaLLM(model="mistral", temperature=0)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

evaluation_agent = initialize_agent(
    tools=[clar_tool, tgt_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)


def agentic_grading_pipeline(question: str, student_answer: str, ideal_answer: str, rubric: str) -> str:
    """
    1) The agent calls ClarificationEvaluator 
       with a single JSON string input for question, answer, etc.
    2) The agent calls TargetGuidedEvaluator similarly.
    3) The agent does a final step to combine them.
    """
    clar_prompt = f"""
    You are a grading agent. 
    1) Use the 'ClarificationEvaluator' tool by passing a single JSON string
    with "question", "student_answer", "ideal_answer", and "rubric".

    For example:
    Action: ClarificationEvaluator
    Action Input: 
    "{{
    \\"question\\": \\"{question}\\",
    \\"student_answer\\": \\"{student_answer}\\",
    \\"ideal_answer\\": \\"{ideal_answer}\\",
    \\"rubric\\": \\"{rubric}\\"
    }}"

    2) Return the tool's final answer here.
    """
    clar_result = evaluation_agent.run(clar_prompt)
    print("\n--- [DEBUG] Clarification Result ---")
    print(clar_result)

    tgt_prompt = f"""
    Now use the 'TargetGuidedEvaluator' tool by passing a single JSON string
    with "question", "student_answer", "ideal_answer", and "rubric".

    For example:
    Action: TargetGuidedEvaluator
    Action Input: 
    "{{
    \\"question\\": \\"{question}\\",
    \\"student_answer\\": \\"{student_answer}\\",
    \\"ideal_answer\\": \\"{ideal_answer}\\",
    \\"rubric\\": \\"{rubric}\\"
    }}"

    Then return the tool's final answer here.
    """
    tgt_result = evaluation_agent.run(tgt_prompt)
    print("\n--- [DEBUG] Target-Guided Result ---")
    print(tgt_result)

    final_prompt = f"""
    We have two partial outputs:
    Clarification: {clar_result}
    Target-Guided: {tgt_result}

    Please produce a final summary of the student's performance, 
    e.g. a final JSON with "overall_score" and "explanation".
    Return it in standard ReAct final answer format.
    """
    final_result = evaluation_agent.run(final_prompt)
    print("\n--- [DEBUG] Final Decision ---")
    print(final_result)
    return final_result


if __name__ == "__main__":
    question = "What is the main difference between a while and a do...while statement?"
    student_answer = "The do-while runs code first, then checks the condition."
    ideal_answer = "A do...while loop executes its block at least once before checking the condition."
    rubric = "2 points for 'at least once', 3 points for stating difference."

    print("[DEBUG] --- Starting Agentic Grading Pipeline ---")
    final_output = agentic_grading_pipeline(question, student_answer, ideal_answer, rubric)
    print("\n--- [RESULT] Agentic Grading Output ---")
    print(final_output)
