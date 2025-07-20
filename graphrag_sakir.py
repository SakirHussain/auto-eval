import os
import re
import numpy as np
import networkx as nx

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Local imports
import config
from prompts import RAG_PROMPT_TEMPLATE

##############################
#  Debug Utility
##############################
def debug(msg):
    print(f"[DEBUG] {msg}")

##############################
#  Text Cleaning
##############################
def clean_text(text):
    text = re.sub(r"Page\s*\d+\s*of\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

##############################
#  Knowledge Graph (for future use with a custom retriever)
##############################
def build_knowledge_graph(chunks):
    debug(f"Building knowledge graph with {len(chunks)} nodes")
    G = nx.DiGraph()
    for i, chunk in enumerate(chunks):
        node_id = f"chunk_{i}"
        G.add_node(node_id, text=chunk)
    return G

def create_chunk_relationships(G, embeddings_dict, threshold):
    debug(f"Creating edges (threshold={threshold})...")
    node_list = list(G.nodes)
    count = 0
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            n1, n2 = node_list[i], node_list[j]
            e1, e2 = embeddings_dict[n1], embeddings_dict[n2]
            sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
            if sim >= threshold:
                G.add_edge(n1, n2, relation="similar")
                G.add_edge(n2, n1, relation="similar")
                count += 2
    debug(f"Added {count} edges")

def embed_nodes(G, model):
    debug(f"Embedding nodes with model='{model}'")
    embeddings = OllamaEmbeddings(model=model)
    embeddings_dict = {}
    for node in G.nodes():
        embeddings_dict[node] = embeddings.embed_query(G.nodes[node]["text"])
    debug("Embedding complete")
    return embeddings_dict

##############################
#  Main RAG Generation Function
##############################
def rag_generate(query, rubric_items):
    pdf_path = config.PDF_PATH
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(config.CORPORA_DIR, base)
    vectorstore_path = os.path.join(out_dir, "vectorstore")
    kg_path = os.path.join(out_dir, f"{base}_kg.gpickle")
    
    os.makedirs(out_dir, exist_ok=True)

    embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDING_MODEL)

    # === Check if vector store and graph are already processed ===
    if not os.path.exists(vectorstore_path):
        debug("No existing vector store found. Starting fresh...")
        
        # 1. Load and Chunk Documents
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        cleaned_content = [clean_text(doc.page_content) for doc in docs]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " "]
        )
        splits = text_splitter.create_documents(cleaned_content)
        
        # 2. Create and save FAISS vector store
        debug(f"Creating and saving vectorstore to {vectorstore_path}")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(vectorstore_path)
        
        # 3. Build and save Knowledge Graph
        debug(f"Building and saving knowledge graph to {kg_path}")
        text_chunks = [doc.page_content for doc in splits]
        G = build_knowledge_graph(text_chunks)
        embeddings_dict = embed_nodes(G, model=config.OLLAMA_EMBEDDING_MODEL)
        create_chunk_relationships(G, embeddings_dict, threshold=config.SIMILARITY_THRESHOLD)
        nx.write_gpickle(G, kg_path)
        
        debug("Initial processing complete.")
    else:
        debug("Vectorstore & knowledge graph found. Loading...")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        # G = nx.read_gpickle(kg_path) # Graph is loaded but not yet used in the LCEL chain

    # === Create LangChain RAG Chain using LCEL ===
    llm = OllamaLLM(model=config.OLLAMA_LLM_MODEL, temperature=0.7)
    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LangChain Expression Language (LCEL) chain
    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "rubric_items": lambda x: x["rubric_items"]
        }
        | RAG_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    debug(f"Invoking RAG chain for query: '{query}'")
    final_answer = rag_chain.invoke({
        "question": query,
        "rubric_items": "\n".join(rubric_items)
    })
    
    print("\n===== FINAL ANSWER (from LangChain) =====")
    print(final_answer)
    return final_answer
