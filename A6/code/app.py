import streamlit as st
import chromadb
import os
from openai import OpenAI
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# --- 1. ABSOLUTE PATH SETUP ---
# Path to this file (A6/code/app.py)
current_file_path = os.path.abspath(__file__)
# Path to the 'code' folder (A6/code)
code_dir = os.path.dirname(current_file_path)
# Path to the 'A6' folder (one level up)
a6_dir = os.path.dirname(code_dir)

# Load .env from A6/.env
load_dotenv(os.path.join(a6_dir, ".env"))

# --- 2. OPENAI INITIALIZATION ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(f"❌ API Key not found! Checked: {os.path.join(a6_dir, '.env')}")
    st.stop()

client = OpenAI(api_key=api_key)

# --- 3. CHROMADB & EMBEDDING SETUP ---
# Database is in A6/code/chroma_db
db_path = os.path.join(code_dir, "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)

emb_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small"
)

COLLECTION_NAME = "contextual_rag_st125985"

# --- 4. CONNECTION LOGIC & DEBUGGING ---
try:
    # Get list of all collections in the DB folder
    existing_collections = [c.name for c in chroma_client.list_collections()]
    
    if COLLECTION_NAME not in existing_collections:
        st.error(f"⚠️ Collection '{COLLECTION_NAME}' not found.")
        st.warning(f"Looking in folder: `{db_path}`")
        st.info(f"Actually found these collections: {existing_collections}")
        st.markdown("""
        **How to fix:**
        1. Check if you have a `chroma_db` folder in the root `A6` folder.
        2. If yes, move it inside the `code` folder.
        3. Refresh this page.
        """)
        st.stop()
    
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    
except Exception as e:
    st.error(f"Critial Error: {e}")
    st.stop()

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Contextual RAG Chatbot", page_icon="🤖")
st.title("🤖 Chapter 5: Contextual RAG")
st.caption(f"Student ID: st125985 | Connected to: {COLLECTION_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about Word Embeddings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Retrieval Step
        results = collection.query(query_texts=[prompt], n_results=3)
        retrieved_docs = results['documents'][0]
        context = "\n\n".join(retrieved_docs)
        
        # Generation Step
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context from Chapter 5 to answer questions accurately."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ]
        )
        
        full_response = response.choices[0].message.content
        st.markdown(full_response)
        
        # Sources Expander
        with st.expander("🔍 View Retrieved Context"):
            for i, doc in enumerate(retrieved_docs):
                st.info(f"Chunk {i+1}:\n{doc}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})