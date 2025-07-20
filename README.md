# Auto-Evaluation System for Student Answers

## Project Overview

This project implements an advanced auto-evaluation system designed to grade student answers against ideal responses and rubrics. It leverages a combination of Graph-based Retrieval-Augmented Generation (RAG), Proactive Chain-of-Thought (ProCoT) reasoning, non-collaborative content filtering, and machine learning-based score softening. The system aims to provide consistent, fair, and explainable evaluations, mimicking a human grading process.

## Core Concepts and Implementation

### 1. Graph-based Retrieval-Augmented Generation (Graph RAG)

**Concept:** Graph RAG enhances traditional RAG by building a knowledge graph from the source material. This allows for more nuanced retrieval and understanding of relationships within the text, leading to more accurate and contextually rich ideal answer generation.

**Implementation (`graphrag.py`):**

- **Document Loading:** Uses `PyPDFLoader` from `langchain-community` to load content from a specified PDF document (`os.pdf`).
- **Text Chunking:** `RecursiveCharacterTextSplitter` is employed to break down the PDF content into smaller, overlapping chunks.
- **Vector Store (FAISS):** `FAISS` (Facebook AI Similarity Search) is used to create a vector index of these text chunks. This allows for efficient semantic search and retrieval of relevant information based on embedding similarity.
- **Knowledge Graph (NetworkX):** A knowledge graph is constructed using the `networkx` library. Each text chunk becomes a node, and edges are formed between semantically similar chunks (based on cosine similarity of their embeddings).
- **Ollama Integration:** `OllamaEmbeddings` and `OllamaLLM` are used for generating embeddings and interacting with local large language models (LLMs) for answer generation.
- **LangChain Expression Language (LCEL):** The entire RAG pipeline is orchestrated using LCEL, creating a flexible and efficient chain that:
  1.  Retrieves relevant context from the FAISS vector store based on the input question.
  2.  Passes the question, rubric items, and retrieved context to the LLM.
  3.  Generates a comprehensive ideal answer that adheres to the rubric and is grounded in the retrieved information.

### 2. Proactive Chain of Thought (ProCoT) Evaluation

**Concept:** ProCoT is a prompting technique that guides the LLM through a structured reasoning process, allowing it to make more deliberate and explainable grading decisions. It involves iterative refinement and different "dialogue types" to simulate human thought processes during evaluation.

**Implementation (`proactive_chain_of_thought.py`):**

- **Iterative Refinement:** The system performs multiple turns of evaluation. In each turn, the LLM re-evaluates its previous score and reasoning, aiming for a more accurate and refined final score. This uses `ConversationChain` and `ConversationBufferMemory` from LangChain to maintain context.
- **Structured Output (Pydantic):** `PydanticOutputParser` is used to enforce a strict JSON output format from the LLM. This ensures that the evaluation results (thought process, action taken, response, final score) are consistently structured and machine-readable.
- **Dialogue Types:** The system can simulate different evaluation approaches (e.g., Clarification Dialogue, Target-Guided Dialogue), each with specific instructions for the LLM to follow during grading.
- **Similarity Metrics:** Thematic (SBERT) and TF-IDF similarities between student and ideal answers are computed and provided to the LLM as additional context for its reasoning.

### 3. Non-Collaborative Content Filtering

**Concept:** This module identifies and filters out irrelevant or off-topic content from student answers before evaluation.

**Implementation (`student_answer_noncollab_filtering_v2.py`):**

- **Sentence Tokenization:** Student answers are broken down into individual sentences using `nltk.sent_tokenize`.
- **Multi-faceted Relevance Scoring:** For each sentence, its relevance to the question is assessed using three methods:
  - **SBERT Similarity:** Cosine similarity of sentence embeddings (from `SentenceTransformer`).
  - **TF-IDF Similarity:** Cosine similarity of TF-IDF vectors.
  - **Zero-Shot Classification:** A `transformers` pipeline classifies the sentence as "relevant" or "irrelevant" to the question.
- **Rolling Context + Dual-Check Filtering:** It processes sentences in a rolling window. It compares the relevance of the current sentence (and its context) against previous context, using a tolerance. A sentence is accepted if at least two out of the three relevance metrics vote for its inclusion, ensuring robust filtering.

### 5. Student Answer Clustering

**Concept:** This module groups similar student answers together based on their semantic content. This allows for faculty to provide more than one ideal/anticipated answer to a question.

**Implementation (`answer_clustering.py`):**

- **Sentence Embeddings:** `SentenceTransformer` is used to generate embeddings for both ideal and student answers.
- **Cosine Similarity:** Student answer embeddings are compared against ideal answer embeddings using cosine similarity to find the closest ideal answer.
- **Clustering:** Student answers are assigned to clusters based on their highest similarity to an ideal answer, provided it exceeds a defined threshold.

## Models Used

This project utilizes a combination of locally run and Hugging Face models:

- **Ollama Models:**

  - `gemma3:latest`: Used as the primary Large Language Model (LLM) for RAG answer generation and ProCoT evaluation.
  - `bge-m3`: Used as the embedding model for generating vector representations of text in Graph RAG.

- **Hugging Face Models:**
  - `sentence-transformers/all-MiniLM-L6-v2`: Used by `sentence-transformers` for generating embeddings in clustering, non-collaborative filtering, and score softening.
  - `facebook/bart-large-mnli`: Used by the `transformers` pipeline for zero-shot classification in the non-collaborative filtering module.

## Project Structure

```
.
├───answer_clustering.py                # Implements student answer clustering and visualization.
├───api.py                              # FastAPI application for exposing evaluation endpoints.
├───app.py                              # Streamlit frontend for interactive demo.
├───auto_eval_research_paper.pdf        # Example PDF document for Graph RAG.
├───config.py                           # Centralized configuration for paths, model names, and parameters.
├───graphrag_sakir.py                   # Implements the Graph RAG pipeline for ideal answer generation.
├───os.pdf                              # The PDF document used as source material for RAG.
├───proactive_chain_of_thought_gaurdrails.py # Core logic for ProCoT evaluation and iterative refinement.
├───prompts.py                          # Stores all LLM prompt templates.
├───requirements.txt                    # Python dependencies.
├───softener_gbr.joblib                 # Persisted machine learning model for score softening.
├───softner.py                          # Implements the score softening model training and inference.
├───student_answer_noncollab_filtering_v2.py # Improved non-collaborative content filtering.
├───student_answer_noncollab_filtering.py    # Original non-collaborative content filtering (deprecated).
├───data/                               # Directory for all JSON data files.
    ├───procot_eval_with_ideal.json     # Dataset for training the score softening model.
    ├───qnia.json                       # Ideal answers data (used by clustering).
    └───student_answers.json            # Student answers data (used by clustering).
├───.git/                               # Git repository files.
└───corpora/                            # Directory for processed data (vectorstores, knowledge graphs).
    └───os/
        ├───os_kg.gpickle               # Knowledge graph for 'os.pdf'.
        └───vectorstore/
            ├───index.faiss             # FAISS index for 'os.pdf'.
            └───index.pkl               # FAISS index metadata.
```

## Setup and Installation

### Prerequisites

- **Python 3.11.9**
- **Ollama:** Ensure Ollama is installed and running on your system. You need to pull the required models:
  ```bash
  ollama pull gemma3:latest
  ollama pull bge-m3
  ```

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a virtual environment:**

    ```bash
    python -m venv venv

    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK data:**
    The `student_answer_noncollab_filtering_v2.py` uses `nltk.sent_tokenize`. You might need to download the `punkt` tokenizer:
    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

### Running the FastAPI Backend

To start the API server:

```bash
uvicorn api:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

**Endpoints:**

- `POST /generate_ideal`: Generates an ideal answer based on a question and rubric.
- `POST /evaluate`: Evaluates a student answer using ProCoT.
- `POST /soften_score`: Softens a ProCoT score using the trained ML model.

### Running the Streamlit Frontend

To launch the interactive Streamlit application:

```bash
streamlit run app.py
```

This will open the application in your web browser, typically at `http://localhost:8501`.
