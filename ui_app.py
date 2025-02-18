import streamlit as st
import json
import os
from rag_json_OP import main as generate_answer_main  # Import main function for answer generation
from proactive_chain_of_thought import main as evaluate_answer_main  # Import main function for evaluation

# Define the JSON files for storing data
RUBRIC_FILE = "rubrics.json"
ANSWER_FILE = "generated_answer.json"
EVALUATION_FILE = "evaluation_results.json"

# Ensure required JSON files exist
for file in [RUBRIC_FILE, ANSWER_FILE, EVALUATION_FILE]:
    if not os.path.exists(file):
        with open(file, "w") as f:
            json.dump([], f) if file == RUBRIC_FILE else json.dump({}, f)

# Streamlit UI
st.set_page_config(page_title="AI Answer Generation & Evaluation", layout="wide")
st.title("📘 AI-Powered Answer Generation & Evaluation")

# --- Section 1: Rubric Input ---
st.header("📝 Enter Rubric Details")

question = st.text_area("Enter Question:", placeholder="Example: Compare and contrast perception model and sensor model.")
corpus_path = st.text_input("Enter Corpus Path:", placeholder="C:/Users/KIRTI/Downloads/chap 1 cog.pdf")
total_marks = st.number_input("Total Marks:", min_value=1, max_value=100, value=10)

st.subheader("Define Topics and Mark Allocation")
topics = {}
topic_name = st.text_input("Topic Name:")
topic_marks = st.number_input("Marks for this topic:", min_value=1, max_value=total_marks, value=2)

if st.button("Add Topic"):
    if topic_name and topic_marks:
        topics[topic_name] = topic_marks
        st.success(f"Added topic: {topic_name} ({topic_marks} marks)")
    else:
        st.warning("Please enter both topic name and marks.")

# Button to Save Rubric
if st.button("Save Rubric"):
    if question and topics and corpus_path:
        new_rubric = {
            "question": question,
            "rubric": {"total_marks": total_marks, "topics": topics},
            "corpus_path": corpus_path
        }

        # Load existing rubrics and append new rubric
        with open(RUBRIC_FILE, "r+") as f:
            data = json.load(f)
            data.append(new_rubric)
            f.seek(0)
            json.dump(data, f, indent=4)

        st.success("Rubric saved successfully!")
    else:
        st.warning("Please fill in all fields before saving.")

# --- Section 2: Generate Answer ---
st.header("🤖 Generate AI-Powered Answer")

# if st.button("Generate Answer"):
#     with st.spinner("Generating answer..."):
#         # Run the main function of rag_w_cot_pydantics.py
#         generate_answer_main()

#         # Load generated answer
#         with open(ANSWER_FILE, "r") as f:
#             answer_data = json.load(f)

#         # Display generated answer
#         st.subheader("📜 Generated Answer:")
#         st.write(answer_data["answer"])

if st.button("Generate Answer"):
    with st.spinner("Generating answer..."):
        try:
            print("🔵 Running Answer Generation...")  # Debugging
            generate_answer_main()
            print("✅ Answer Generation Completed!")

            # Verify if JSON file is created
            if not os.path.exists("generated_answer.json"):
                st.error("❌ `generated_answer.json` was not created.")
                print("❌ `generated_answer.json` was not found!")
                exit()

            with open("generated_answer.json", "r") as f:
                answer_data = json.load(f)

            print(f"📜 Generated Answer:\n{answer_data}")  # Debugging

            st.subheader("📜 Generated Answer:")
            st.write(answer_data["answer"])

        except Exception as e:
            st.error(f"❌ Error in answer generation: {e}")
            print(f"❌ Exception: {e}")


# --- Section 3: Evaluate Answer ---
st.header("📊 Evaluate Answer")

if st.button("Evaluate Answer"):
    with st.spinner("Evaluating answer..."):
        # Run the main function of proactive_chain_of_thought.py
        evaluate_answer_main()

        # Load evaluation results
        with open(EVALUATION_FILE, "r") as f:
            evaluation_result = json.load(f)

        # Display Evaluation Results
        st.subheader("✅ Evaluation Results")
        st.write(f"**Total Score:** {evaluation_result['total_score']} / 10")
        st.write(f"**Total Deductions:** {evaluation_result['total_deductions']}")

        st.subheader("💡 Feedback:")
        for feedback in evaluation_result["feedback"]:
            st.write(f"- {feedback}")

