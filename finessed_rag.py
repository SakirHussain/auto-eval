import json
import re
import nltk
nltk.download('punkt_tab')
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import NLTKTextSplitter
from langchain_core.exceptions import OutputParserException
from langchain_community.document_loaders import PyPDFLoader
from rank_bm25 import BM25Okapi
from transformers import pipeline
from sentence_transformers import CrossEncoder

# Step 1: Initialize Model
model = OllamaLLM(model="deepseek-r1:7b", temperature=0.7)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Step 2: Utility Functions
def clean_text(text):
    """Custom function to clean and normalize text."""
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def remove_think_tags(text):
    """Removes <think> tags and their content."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def expand_query(question):
    """Generates paraphrased versions of the question to improve retrieval."""
    paraphraser = pipeline("text2text-generation", model="t5-small")

    # Ensure sampling is enabled to allow multiple outputs
    paraphrases = paraphraser(
        f"Paraphrase: {question}",
        max_length=64,
        num_return_sequences=3,  # Generate 3 variations
        do_sample=True  # Enable sampling (avoids greedy decoding error)
    )

    return [question] + [p["generated_text"] for p in paraphrases]

def dynamic_text_splitter(text):
    """Dynamically split text into chunks based on sentence structure."""
    splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.create_documents([text])

def load_and_clean_corpus(file_path):
    """Load and clean documents from a PDF file."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return [clean_text(doc.page_content) for doc in documents]

def create_vector_store(corpus_path):
    """Create a FAISS vector store with cleaned documents."""
    documents = load_and_clean_corpus(corpus_path)
    text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def bm25_retrieve(question, corpus):
    """Perform sparse retrieval using BM25."""
    tokenized_corpus = [doc.split() for doc in corpus]  
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(question.split())
    ranked_docs = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked_docs[:5]]

def hybrid_retrieval(question, vector_store, corpus, k=5):
    """Combines FAISS (dense) and BM25 (sparse) retrieval."""
    expanded_queries = expand_query(question)
    faiss_docs = []
    for query in expanded_queries:
        faiss_docs.extend(vector_store.similarity_search(query, k=k))
    bm25_docs = bm25_retrieve(question, corpus)
    all_docs = list({doc: doc for doc in faiss_docs + bm25_docs}.values())  # Deduplicate
    ranked_docs = reranker.predict([(question, doc) for doc in all_docs])  # Rerank
    return [doc for doc, score in sorted(zip(all_docs, ranked_docs), key=lambda x: x[1], reverse=True)][:k]

def rag_generate(question, rubric, corpus_path):
    """Execute RAG with enhanced retrieval optimization."""
    vector_store = create_vector_store(corpus_path)
    corpus = load_and_clean_corpus(corpus_path)
    relevant_docs = hybrid_retrieval(question, vector_store, corpus)
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
    return remove_think_tags(llm_response)

if __name__ == "__main__":
    with open("rubrics.json", "r") as file:
        data = json.load(file)
    
    question = data["question"]
    rubric = data["rubric"]  # Dynamically accepts rubric with topics and weightage
    corpus_path = data["corpus_path"]
    
    generated_answer = rag_generate(question, rubric, corpus_path)
    print(generated_answer)
