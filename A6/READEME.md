# 📚 Chapter 5: Contextual RAG Chatbot
**Student ID:** st125985  
**Course:** NLP Assignment A6  

## 🚀 Project Overview
This project implements a **Contextual Retrieval-Augmented Generation (RAG)** system based on Chapter 5 (Word Embeddings) of the Jurafsky & Martin textbook. Unlike standard RAG, this system ensures that retrieved document chunks maintain their context, leading to more accurate and semantically relevant answers regarding word embeddings.

## 🛠️ Features
* **PDF Processing:** Extracts and cleans text from Chapter 5 PDF.
* **Vector Database:** Uses **ChromaDB** for persistent storage and retrieval of document embeddings.
* **Embedding Model:** Utilizes OpenAI's `text-embedding-3-small` for high-quality vector representations.
* **Streamlit UI:** A clean, chat-based interface for interacting with the AI.
* **Source Transparency:** An expander to view the exact text chunks retrieved from the database.

## 📂 Project Structure
```text
A6/
├── code/
│   ├── app.py           # Main Streamlit application
│   ├── code.ipynb       # Jupyter Notebook for data ingestion
│   └── chroma_db/       # Persistent vector database
├── answer/
│   └── response-st125985-chapter-5.json  # Generated evaluation file
├── .env                 # API Keys (OpenAI)
├── requirements.txt     # Python dependencies
└── chapter5.pdf         # Source material
⚙️ Setup & Installation
Clone the repository:

Bash
git clone <your-repo-url>
cd A6
Install Dependencies:

Bash
pip install -r requirements.txt
Configure Environment:
Create a .env file in the root A6 folder and add your OpenAI API Key:

Plaintext
OPENAI_API_KEY=your_key_here
Initialize Database:
Run all cells in code/code.ipynb to download the PDF, chunk the text, and populate the ChromaDB collection contextual_rag_st125985.

Run the App:

Bash
cd code
streamlit run app.py
🤖 Usage
Once the app is running, you can ask questions about:

The difference between sparse and dense vectors.

Word2Vec (Skip-gram and CBOW).

Cosine similarity and vector semantics.

Pointwise Mutual Information (PMI).

📝 Author
Student: st125985

Class: NLP 2026
![Alt text](path/to/Demo.gif)

