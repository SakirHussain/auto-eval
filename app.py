from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaLLM

# Import prompt templates and questions
from prompt_templates import BasicPromptTemplate, DetailedExplanationPromptTemplate
from questions import BasicQuestions, AdvancedQuestions

# Step 1: Initialize the model
model = OllamaLLM(model="deepseek-r1:7b", temperature=0.7)

# Step 2: Load the corpus and create the vector store

def load_corpus(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()

    # Debug: Print documents to verify loading
    print("\n--- Loaded Documents ---")
    for doc in documents:
        print(doc.page_content[:500], "\n--- Document End ---\n")

    return documents


def create_vector_store(corpus_path):
    """Create a FAISS vector store from the corpus."""
    documents = load_corpus(corpus_path)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
      # Debug: Print each chunk to verify splitting
    print("\n--- Document Chunks ---")
    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}: {doc.page_content}\n")
    
    embeddings = OpenAIEmbeddings(api_key="")  # TRY TO USE DIFFERENT MODEL HERE
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def retrieve_relevant_docs(question, vector_store):
    relevant_docs = vector_store.similarity_search(question, k=3)

    # Debug: Print retrieved chunks
    print("\n--- Retrieved Chunks ---")
    for i, doc in enumerate(relevant_docs):
        print(f"Retrieved Chunk {i+1}: {doc.page_content}\n")

    return relevant_docs

# Step 3: Define the RAG chain and process

def rag_chain(prompt_class, questions_class, corpus_path):
    # Load the selected prompt and questions
    prompt = prompt_class.get_template()
    
    print("PROMPT : ", prompt)
    
    questions = questions_class.get_questions()
    vector_store = create_vector_store(corpus_path)

    # Create the LLM chain
    chain = LLMChain(llm=model, prompt=prompt)

    # Process each question
    for question in questions:
        relevant_docs = retrieve_relevant_docs(question, vector_store)        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print(f"Question : {question}")
        print("CONTEXT : ", context)
        response = chain.run({"question": question, "context": context})
        
        print(f"Response: {response}\n")

# Step 4: Run the RAG chain
if __name__ == "__main__":
    # You can choose any prompt template and question set here
    rag_chain(
        prompt_class=BasicPromptTemplate,
        questions_class=BasicQuestions,
        corpus_path="corpus.txt"
    )
