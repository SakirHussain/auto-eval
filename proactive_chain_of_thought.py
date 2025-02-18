import re
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_md")

def compute_thematic_similarity(student_answer: str, ideal_answer: str):
    """Computes thematic similarity between student answer and ideal answer."""
    student_doc = nlp(student_answer)
    ideal_doc = nlp(ideal_answer)
    return student_doc.similarity(ideal_doc)

def compute_tfidf_similarity(student_answer: str, ideal_answer: str):
    """Computes TF-IDF similarity between student answer and ideal answer."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([student_answer, ideal_answer])
    return cosine_similarity(vectors)[0, 1]

# Initialize DeepSeek R1 model
model = OllamaLLM(model="deepseek-r1:7b", temperature=0.45)

nlp = spacy.load("en_core_web_md")

class ProCoTOutput(BaseModel):
    thought_process: str = Field(..., description="Reasoning before selecting an action.")
    action_taken: str = Field(..., description="Chosen action based on evaluation.")
    response: str = Field(..., description="Generated feedback with deductions or awards.")
    final_adjusted_score: float = Field(..., description="Final adjusted score after refinements.")


# safe parsing
def safe_parse(parser, llm_response):
    """Safely parses the LLM response, ensuring strict JSON format compliance."""
    
    # Remove <think> tags
    llm_response_cleaned = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
    print(f"\n--- Cleaned LLM Response ---\n{llm_response_cleaned}")

    # # Validate response format before parsing
    # if not llm_response_cleaned.startswith("{") or not llm_response_cleaned.endswith("}"):
    #     print("\n--- ERROR: LLM Response is not valid JSON! Retrying... ---")
    #     return None  # Retry instead of crashing

    # Try parsing the cleaned response
    try:
        return parser.parse(llm_response_cleaned)
    except OutputParserException as e:
        print("\n--- Parsing Failed ---")
        print(f"Error: {e}\nRaw Cleaned LLM Response:\n{llm_response_cleaned}")
        print("\nRetrying with stricter format enforcement...")
        return None  # Indicating failure


# Function to generate and parse LLM responses using ProCoT
def evaluate_dialogue(dialogue_type, dialgoue_desc ,question, student_answer, ideal_answer, rubric, conversation_history, available_actions):
    """Evaluates student answers using ProCoT with a structured prompt and output parser."""
    
    thematic_sim = "Not Calculated"
    tfidf_sim = "Not Calculated"
    
    if dialogue_type == "Target-Guided Dialogue":
        thematic_sim = compute_thematic_similarity(student_answer, ideal_answer)
        tfidf_sim = compute_tfidf_similarity(student_answer, ideal_answer)
        
    print("\n--- thematic_sim: ", thematic_sim)
    print("--- tfidf_sim: ", tfidf_sim)
    
    parser = PydanticOutputParser(pydantic_object=ProCoTOutput)
    
    prompt = PromptTemplate(
    template="""
    You are a professor evaluating a student's answer. Your task is to fairly evaluate the student's response based on the provided rubric while ensuring strict adherence to grading criteria.

    Context and Role:
    - You are responsible for grading fairly and consistently based on the rubric provided.  
    - Max Marks Possible: 10  
    - No assumptions should be made**—your evaluation should strictly follow the rubric.  
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
    - D (Task Background): "You are a teacher grading a student's answer based on the rubric."  
    - C (Conversation History): "{conversation_history}"  
    - A (Available Actions): {available_actions}  

    Scoring Guidelines
    - Any addition or deduction of marks** must be explicitly based on whether the rubric is satisfied.  
    - No information must be assumed or added**—only infer from the provided inputs.  

    Response Format (Strict JSON)
    You must return only valid JSON in the exact structure below:  
    ```json
    {{
        "thought_process": "Your reasoning before selecting an action.",
        "action_taken": "Chosen action based on evaluation.",
        "response": "Generated feedback with deductions or awards.",
        "final_adjusted_score": 0.0
    }}
    ```
    - Do NOT include any explanations or formatting outside JSON. 
    - If deductions are made, ensure they align exactly with the rubric weights.  

    {format_instructions}
    """,
    input_variables=["dialogue_type", "dialogue_desc", "question", "student_answer", "ideal_answer", "rubric", "conversation_history", "available_actions", "thematic_sim", "tfidf_sim"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


    print(f"\n--- Running {dialogue_type} Evaluation ---")
    formatted_prompt = prompt.format(
        dialogue_type=dialogue_type,
        dialogue_desc=dialgoue_desc,
        question=question,
        student_answer=student_answer,
        ideal_answer=ideal_answer,
        rubric=rubric,
        conversation_history=conversation_history,
        available_actions=available_actions,
        thematic_sim= thematic_sim,
        tfidf_sim=tfidf_sim
    )

    # print(f"\n--- Prompt ---\n")
    # print(formatted_prompt)
    
    llm_response = model.invoke(formatted_prompt)
    print(f"\n--- Raw LLM Response ---\n{llm_response}")

    result = safe_parse(parser, llm_response)

    if not result:
        print("\n--- ERROR: Invalid response format. Skipping turn. ---")

    print(f"\n--- Evaluation Result ---")
    print(result)

    # Update conversation history for next turn
    # conversation_history.append({
    #     "thought_process": result.thought_process,
    #     "action_taken": result.action_taken,
    #     "response": result.response
    # })

    # Adjust score dynamically
    # score_adjustment = 0
    # if "Award" in result.action_taken:
    #     score_adjustment = float(result.action_taken.split("+")[-1])
    # elif "Deduct" in result.action_taken:
    #     score_adjustment = float(result.action_taken.split("-")[-1]) * -1

    # Apply final adjusted score refinement in the last turn
    return result


# Master function that runs all three evaluation phases sequentially
def evaluate_answer(question, student_answer, ideal_answer, rubric):
    print("\n--- Starting Evaluation ---")
    print(f"Question: {question}")
    print(f"Student Answer: {student_answer}")
    print(f"Ideal Answer: {ideal_answer}")
    print(f"Rubric: {rubric}")
    
    conversation_history = []

    clarification_actions = ["Deduct marks", "Add marks"]
    target_guided_actions = ["Deduct marks", "Add marks"]
    non_collab_actions = ["Deduct marks", "Add marks"]
    
    clarification_desc = '''
    - Identify missing, unclear, or ambiguous details in the student's answer.
    - Deduct marks based on missing information.
    - Explain why marks were deducted.
    '''
    
    target_guided_desc = '''
    - Determine how many transformations (steps or turns) are needed to thematically convert the student's answer into the ideal answer.
    - Deduct marks based on the number of necessary transformations. To further assist you with the evaluation, the thematic similarity score and TF-IDF similarity score are provided.
    '''
    
    non_collab_desc = '''
     - Detect if the student's answer is off-topic, vague, or irrelevant.
    - Deduct marks accordingly and list the detected issues.
    '''

    clarification_result = evaluate_dialogue("Clarification Dialogue", clarification_desc, question, student_answer, ideal_answer, rubric, conversation_history, clarification_actions)
    print("\n--- Clarification Dialogue Completed ---")
    print(clarification_result)
    
    target_guided_result = evaluate_dialogue("Target-Guided Dialogue", target_guided_desc, question, student_answer, ideal_answer, rubric, conversation_history, target_guided_actions)
    print("\n--- Target-Guided Dialogue Completed ---")
    print(target_guided_result)
    
    non_collab_result = evaluate_dialogue("Non-Collaborative Dialogue", non_collab_desc,question, student_answer, ideal_answer, rubric, conversation_history, non_collab_actions)
    print("\n--- Non-Collaborative Dialogue Completed ---")
    print(non_collab_result)

    # Ensure no NoneType errors
    clarification_score = clarification_result.final_adjusted_score if clarification_result else 0
    target_guided_score = target_guided_result.final_adjusted_score if target_guided_result else 0
    non_collab_score = non_collab_result.final_adjusted_score if non_collab_result else 0

    total_adjusted_score = (clarification_score + target_guided_score + non_collab_score)/3.0

    print("\n--- Final Evaluation Summary ---")
    print(f"Total Score: {total_adjusted_score}")

    return {
        "total_score": total_adjusted_score,
        "feedback": [
            clarification_result.response if clarification_result else "Clarification dialogue failed.",
            target_guided_result.response if target_guided_result else "Target-guided dialogue failed.",
            non_collab_result.response if non_collab_result else "Non-collaborative dialogue failed."
        ]
    }


#example
# question = '''
# In the following passage from Cormac McCarthyís novel The Crossing (1994), the narrator describes a dramatic experience. Read the passage carefully. Then, in a well-organized essay, show how McCarthyís techniques convey the impact of the experience on the main character. 
# By the time he reached the first talus[1] slides under the tall escarpments[2] of the Pilares the dawn was not far to come. He reined the horse in a grassy swale and stood down and dropped the reins. His trousers were stiff with blood. He cradled the wolf in his arms and lowered her to the ground and unfolded the sheet. She was stiff and cold and her fur was bristly with the blood dried upon it. He walked the horse back to the creek and left it standing to water and scouted the banks for wood with which to make a fire. Coyotes were yapping along the hills to the south and they were calling from the dark shapes of the rimlands above him where their cries seemed to have no origin other than the night itself. He got the fire going and lifted the wolf from the sheet and took the sheet to the creek and crouched in the dark and washed the blood out of it and brought it back and he cut forked sticks from a mountain hackberry and drove them into the ground with a rock and hung the sheet on a trestlepole where it steamed in the firelight like a burning scrim standing in a wilder-ness where celebrants of some sacred passion had been carried off by rival sects or perhaps had simply fled in the night at the fear of their own doing. He pulled the blanket about his shoulders and sat shiver-ing in the cold and waiting for the dawn that he could find the place where he would bury the wolf. After a while the horse came up from the creek trailing the wet reins through the leaves and stood at the edge of the fire. He fell asleep with his hands palm up before him like some dozing penitent. When he woke it was still dark. The fire had died to a few low flames seething over the coals. He took off his hat and fanned the fire 1 A sloping mass of rock debris at the base of a cliff 2 Steep slopes with it and coaxed it back and fed the wood heíd gathered. He looked for the horse but could not see it. The coyotes were still calling all along the stone ramparts of the Pilares and it was graying faintly in the east. He squatted over the wolf and touched her fur. He touched the cold and perfect teeth. The eye turned to the fire gave back no light and he closed it with his thumb and sat by her and put his hand upon her bloodied forehead and closed his own eyes that he could see her running in the mountains, running in the starlight where the grass was wet and the sunís coming as yet had not undone the rich matrix of creatures passed in the night before her. Deer and hare and dove and groundvole all richly empaneled on the air for her delight, all nations of the possible world ordained by God of which she was one among and not separate from. Where she ran the cries of the coyotes clapped shut as if a door had closed upon them and all was fear and marvel. He took up her stiff head out of the leaves and held it or he reached to hold what cannot be held, what already ran among the mountains at once terrible and of a great beauty, like flowers that feed on flesh. What blood and bone are made of but can themselves not make on any altar nor by any wound of war. What we may well believe has power to cut and shape and hollow out the dark form of the world surely if wind can, if rain can. But which cannot be held never be held and is no flower but is swift and a huntress and the wind itself is in terror of it and the world cannot lose it.
# '''

# student_answer = '''
# The passage from The Crossing conveys a sense of awe and mystery, and in doing so, imparts the depths of the man's emotions towards the wolf. The mourning for the wolf is raised to an elegiac level, as the man reflects upon the wolf, "a tone terrible and of a great beauty." Several devices are employed to effectively enhance the tone of reverence and loss, including figurative language, diction, sentence structure, rhythm, and repetition.

# The pace of the passage fluctuates, alternating from short, detached sentences such as "He squatted over the wolf and touched her fur. He touched the cold and perfect teeth," to unusually long sentences, which are connected by conjunctions (mostly "and") and serve to reflect the outpouring of emotions and the blurred response the man is experiencing, as in lines 41-47 ("The eye... before her"). This dichotomy in sentence structure emphasizes the periods where the man is overcome by remembrances and extrapolations.

# The figurative language interspersed within the passage is also highly effective, causing an air of mystery, wonder, and respect. This mood is set when the cries of the coyotes are described as "seeming to have no origin other than the night itself." The analogy of the sheet steaming (lines 21-24) enhances the aura of power and sacredness by diction such as "celebrants of some sacred passion" and "burning screen. " This sense of religious power is again by his companion to a "dozing penitante." A sense of the awe-inspiring mixture of terror and beauty is evidenced when the narrator compares the wolf’s soul to "flowers that feed on flesh," introducing an element of both horror and reverence. This highlights how "all was fear and marvel" regarding the wolf.

# The repetition of certain phrases and words emphasizes the ideas behind them.
# For example, "What we may well believe has power to cut and shape and hollow out of the dark form of the world, surely if wind can, if rain can..." This repetition contained within this sentence really clarifies the point that our beliefs shape our perception.

# Also, the repetition of "and" throughout the passage, as in lines 15-21, brings a rhythm to the passage while providing a sense of the man not really realizing what he is doing, only going through the motions.

# The unspecific pronoun "he" actually provides a contrast, where the grief of the man becomes more poignant. The passage metamorphosizes from a more detached account about man's treatment of the body to a touching scene where the man reflects upon the wolf and her spirit.

# The final passage, and especially the last line, is made more important by the reflections of the man. The last line is particularly emphasized by the complete lack of punctuation, which Conveys the magnitude of the man's loss. His utter grief over losing the wolf is fully revealed to the reader, especially in the last words:

# "But which cannot be held, never be held, and is no flower but is swift and a huntress, and the wind itself is in terror of it and the world cannot lose it."

# The importance of the wolf's role in "the possible word ordained by God, of which she was one among and not separate from" is made known to the reader through the man’s thoughts and actions.

# In doing so, and in the setting (with the sun beginning to faintly gray the east), a mood of respectful reverence and wonderful power is created. The man is shown to be deeply impacted by his experience.
# '''

# ideal_answer = '''
# In the dark of the night, it runs swiftly along the mountains, up the slopes, past the creek, faster than the winds.

# "What is this 'it' that runs so freely after the body is dead and decaying?" It is purely the soul that escapes after death and returns to its home.

# In the passage from McCarthy’s The Crossing, the soul of the dying wolf leaves the body, and the man carries him to return to his homeland. McCarthy uses imagery and the description of the complete narrative experience to recount the philosophical revelation that the protagonist encounters as he caresses death in the tranquility of nature.

# An outstanding quality about this narrative is the care with which each imagery is told. One repetitive image is that of dark and light. The narrative begins in the dark, though close to dawn.

# The coyotes call from the "dark shapes of the lowlands," giving a clear picture of the grandeur of nature in which the narrator now sits.

# There is also the image of the weak fire out in the cold darkness, a symbol for some hope after death. The fire at first dies; the main character must fan it and relight it, until the dawn sky begins to gray.

# What the main character experiences at dawn can be called mysticism, a philosophical epiphany, and a new window of understanding.

# Such a tone of mystery and enigma is created in the final paragraph (lines 40-65) through the change in the style of writing. The narrative here uses long sentences that run continuously like a stream. The sentences begin to lose the ordinary grammatical form that the narrative followed earlier:

# "What blood and bone are made of but can themselves not make on any altar nor by any wound of war."

# The narrative leaves its traditional form and begins to build on the image of what is passing by the main character's closed eyes, as the limited or omniscient third-person narrator can do.

# The passage has religious allusions: "ordained by God", Personification that breathes life into the mountain: "The flowers feed on flesh.", "The wind and rain cut and shape the land." , "The soul runs wildly through this nation."

# The experience illuminates the power of nature and the strength of the soul to the main character.

# He, in reaching out "to hold what cannot be held," grasped in the moment the mystery of death and eternity—the enigma that is conveyed through the powerful images in this narrative.

# '''

# rubric = '''
# Directions:
# The score you assign should reflect your judgment of the quality of the essay
# as a whole. Reward the writers for what they do well. The score for an
# exceptionally well-written essay may be raised by one point from the score
# otherwise appropriate. In no case may a poorly written essay be scored higher
# than 3.

#  9-8: The writers of these well-constructed essays define the dramatic nature of the experience
# described in Cormac McCarthy's passage and ably demonstrate how the author conveys
# the impact of the experience upon the main character. Having fashioned a convincing
# thesis about the character's reaction to the death of the wolf, these writers support their
# assertions by analyzing the use of specific literary techniques (such as point of view,
# syntax, imagery, or diction) that prove fundamental to their understanding of McCarthy's
# narrative design. They make appropriate references to the text to illustrate their
# argument. Although not without flaws, these essays reflect the writer's ability to control
# a wide range of the elements of effective writing to provide a keen analysis of a literary
# text.

# 7-6: Developing a sound thesis, these writers discuss with clarity and conviction both the
# character's response to the death of the wolf and certain techniques used to convey the
# impact this experience has upon the main character. These essays may not be entirely
# responsive to the rich suggestiveness of the passage or as precise in describing the
# dramatic impact of the event. Although they provide specific references to the text, the
# analysis is less persuasive and perhaps less sophisticated than papers in the 9-8 range:
# they seem less insightful or less controlled, they develop fewer techniques, or their
# discussion of details may be more limited. Nonetheless, they confirm the writer's ability
# to read literary texts with comprehension and to write with organization and control.

# 5: These essays construct a reasonable if reductive thesis; they attempt to link the author's
# literary techniques to the reader's understanding of the impact of the experience on the
# main character. However, the discussion may be superficial, pedestrian, and/or lacking in
# consistent control. The organization may be ineffective or not fully realized. The
# analysis is less developed, less precise, and less convincing than that of upper half
# essays; misinterpretations of particular references or illustrations may detract from the
# overall effect.

# 4-3: These essays attempt to discuss the impact of this dramatic experience upon the main
# character ó and perhaps mention one or more techniques used by McCarthy to effect
# this end. The discussion, however, may be inaccurate or undeveloped. These writers may
# misread the passage in an essential way, rely on paraphrase, or provide only limited
# attention to technique. Illustrations from the text tend to be misconstrued, inexact, or
# omitted altogether. The writing may be sufficient to convey ides, although typically it is
# characterized by weak diction, syntax, grammar, or organization. Essays scored three are
# even less able and may not refer to technique at all.

# 2-1: These essays fail to respond adequately to the question. They may demonstrate confused
# thinking and/or consistent weaknesses in grammar or another basic element of
# composition. They are often unacceptably brief. Although the writer may have made
# some attempt to answer the question, the views presented have little clarity or coherence;
# significant problems with reading comprehension seem evident. Essays that are
# especially inexact, vacuous, and/or mechanically unsound should be scored 1.

# 0: A response with no more than a reference to the task. 
# '''

# evaluation_result = evaluate_answer(question, student_answer, ideal_answer, rubric)