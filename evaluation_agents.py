import os
from typing import Dict, Any

# If you haven't already: pip install langchain_ollama langchain

from langchain.agents import (
    Tool,
    AgentType,
    initialize_agent
)
from langchain_ollama import OllamaLLM  # The Ollama-based LLM
from langchain.prompts import PromptTemplate


###############################################################################
# 1. DEFINE THE "TOOLS" = GRADER, REFLECTOR, REFINER
###############################################################################

def grader_tool_func(inputs: Dict[str, Any]) -> str:
    """
    GRADER TOOL:
      Takes question, ideal_answer, rubrics, student_answer, and total_marks.
      Returns a JSON string with "preliminary_score" and "grader_explanation".
    """
    
    print("GRADER TOOL CALLED")
    
    question = inputs["question"]
    ideal_answer = inputs["ideal_answer"]
    rubrics = inputs["rubrics"]
    student_answer = inputs["student_answer"]
    total_marks = inputs["total_marks"]

    # We'll craft a single prompt for the LLM to act as "Grader."
    # The prompt ends with instructions to produce valid JSON.
    grader_prompt = f"""
You are an expert GRADER. Evaluate the student's answer based on:

- Total marks: {total_marks}
- Question: {question}
- Ideal answer: {ideal_answer}
- Rubrics: {rubrics}

Student's answer: {student_answer}

Step 1: Compare student's answer to the ideal answer & rubrics.
Step 2: Decide a numeric score from 0 up to {total_marks}.
Step 3: Provide a short explanation (why awarding that score).

Return JSON ONLY in this exact format:
{{
  "preliminary_score": <integer_or_float>,
  "grader_explanation": "short chain-of-thought or reason"
}}
---
Now produce the JSON result:
    """

    # Call the local Ollama LLM with "deepseek-r1:7b"
    # temperature=0 for deterministic output
    llm_grader = OllamaLLM(model="deepseek-r1:7b", temperature=0)
    response = llm_grader(grader_prompt)

    return response


grader_tool = Tool(
    name="grader_tool",
    func=grader_tool_func,
    description=(
        "Use this tool to produce a preliminary numeric score (0..total_marks) "
        "and a short explanation referencing how the rubrics and ideal answer compare "
        "to the student's answer."
    )
)


def reflector_tool_func(inputs: Dict[str, Any]) -> str:
    """
    REFLECTOR TOOL:
      Compares the Grader's preliminary output with the rubrics.
      Returns JSON with: 
       - "is_consistent": true/false
       - "corrected_score": <numeric>
       - "reflection_notes": short text about changes or confirmations
    """
    
    print("REFLECTOR TOOL CALLED")
    
    question = inputs["question"]
    ideal_answer = inputs["ideal_answer"]
    rubrics = inputs["rubrics"]
    student_answer = inputs["student_answer"]
    total_marks = inputs["total_marks"]
    grader_output = inputs["grader_output"]

    reflector_prompt = f"""
You are a REFLECTOR. Check if the Grader's output is consistent with the rubrics.

Context:
- Total marks: {total_marks}
- Question: {question}
- Ideal answer: {ideal_answer}
- Rubrics: {rubrics}
- Student's answer: {student_answer}
- Grader's preliminary output: {grader_output}

Instructions:
1. If the Grader's score and explanation align with the rubrics, set "is_consistent": true 
   and "corrected_score" the same as grader's.
2. If the Grader missed something or gave extra points, set "is_consistent": false 
   and "corrected_score": the new correct score.
3. Write a short explanation or note in "reflection_notes".

Return JSON ONLY in this format:
{{
  "is_consistent": true or false,
  "corrected_score": <numeric>,
  "reflection_notes": "string"
}}
---
Now produce the JSON result:
    """

    llm_reflector = OllamaLLM(model="deepseek-r1:7b", temperature=0)
    response = llm_reflector(reflector_prompt)

    return response


reflector_tool = Tool(
    name="reflector_tool",
    func=reflector_tool_func,
    description=(
        "Use this tool to check the Grader's output for consistency with rubrics. "
        "If there's an error, propose a corrected score. Otherwise confirm consistency."
    )
)


def refiner_tool_func(inputs: Dict[str, Any]) -> str:
    """
    REFINER TOOL:
      Merges Grader's and Reflector's results into a final JSON:
        {
          "final_score": <float_or_int>,
          "final_explanation": "short text"
        }
    """
    
    print("REFINER TOOL CALLED")
    
    question = inputs["question"]
    ideal_answer = inputs["ideal_answer"]
    rubrics = inputs["rubrics"]
    student_answer = inputs["student_answer"]
    total_marks = inputs["total_marks"]
    grader_output = inputs["grader_output"]
    reflector_output = inputs["reflector_output"]

    refiner_prompt = f"""
You are the REFINER. Combine the Grader's output and the Reflector's corrections.

Context:
- Total marks: {total_marks}
- Question: {question}
- Ideal answer: {ideal_answer}
- Rubrics: {rubrics}
- Student's answer: {student_answer}
- Grader's preliminary: {grader_output}
- Reflector's feedback: {reflector_output}

Steps:
1. If "is_consistent" is true, use Grader's preliminary_score. 
   Else use the "corrected_score" from the reflector.
2. Summarize final explanation briefly.

Return JSON ONLY in this format:
{{
  "final_score": <numeric>,
  "final_explanation": "short text"
}}
---
Now produce the JSON result:
    """

    llm_refiner = OllamaLLM(model="deepseek-r1:7b", temperature=0)
    response = llm_refiner(refiner_prompt)

    return response


refiner_tool = Tool(
    name="refiner_tool",
    func=refiner_tool_func,
    description=(
        "Use this to finalize the score (final_score) and produce a short final_explanation, "
        "combining the Grader's and Reflector's results."
    )
)

###############################################################################
# 2. SET UP THE META-AGENT USING THE THREE TOOLS (REACT-Like)
###############################################################################

tools = [grader_tool, reflector_tool, refiner_tool]

# The system message instructs the agent to call these tools in order.
SYSTEM_INSTRUCTIONS = """
You are an orchestration agent with three tools:
1) grader_tool
2) reflector_tool
3) refiner_tool

Your task:
- Always call grader_tool first,
- Then call reflector_tool,
- Then call refiner_tool,
- End with the final JSON from refiner_tool.

Do not rewrite rubrics. Do not produce extra commentary. 
Only produce the final JSON from the refiner_tool as your final output.
"""

# We create the OllamaLLM for the main agent "reasoning" calls as well.
# This agent is mostly orchestrating calls to the above tools.
llm_agent = OllamaLLM(model="deepseek-r1:7b", temperature=0)

from langchain.schema import SystemMessage, HumanMessage

def make_meta_agent():
    # We use the initialize_agent function with ReAct. 
    # We'll pass the system message in a stable manner using system_message=...
    meta_agent = initialize_agent(
        tools=tools,
        llm=llm_agent,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        system_message=SystemMessage(content=SYSTEM_INSTRUCTIONS)
    )
    return meta_agent


###############################################################################
# 3. A HELPER FUNCTION TO RUN THE MULTI-AGENT EVALUATION
###############################################################################

def evaluate_student_answer(
    question: str,
    ideal_answer: str,
    rubrics: str,
    student_answer: str,
    total_marks: int
) -> str:
    """
    Orchestrates the multi-agent flow:
      1) grader_tool -> 2) reflector_tool -> 3) refiner_tool
    Returns the final JSON from the refiner_tool.
    """
    meta_agent = make_meta_agent()

    # We'll create a single user message that the agent sees. 
    # The agent will decide how to call the tools in sequence.
    user_input = f"""
QUESTION: {question}
IDEAL_ANSWER: {ideal_answer}
RUBRICS: {rubrics}
STUDENT_ANSWER: {student_answer}
TOTAL_MARKS: {total_marks}

Please follow the steps:
1) Call grader_tool
2) Call reflector_tool
3) Call refiner_tool
Finally, show me the final JSON from refiner_tool.
"""

    # The agent will parse user_input, call each tool, and eventually produce 
    # the final JSON from the refiner_tool's output.
    final_output = meta_agent.run(user_input)

    return final_output


###############################################################################
# 4. USAGE EXAMPLE
###############################################################################
if __name__ == "__main__":
    # Example data:
    question_example = "Q: 1.7 What is the main difference between a while and a do...while statement?"
    
    ideal_answer_example = '''
    1.7 The block inside a do...while statement will execute at least once.
    '''
    
    rubrics_example = '''
    [
    "Accurately identifies that the block inside a do...while statement will execute at least once (2 Marks)",
    "Correctly states the main difference between a while and a do...while statement (3 Marks)"
    ]
    '''
    
    student_answer_example = '''
    1.7 )) What is the main difference between a while and a do...while statement? The do while construct consists of a block of code and a condition. First, the code within the block is executed, and then the condition is evaluated, this is done until it is proven false. The difference between the While loop is it tests the condition before the code within the block is executed.
    '''
    
    total_marks_example = 5

    result = evaluate_student_answer(
        question_example,
        ideal_answer_example,
        rubrics_example,
        student_answer_example,
        total_marks_example
    )
    
    print("\n=== FINAL AGENT OUTPUT ===")
    print(result)
