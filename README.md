# End to End Document Centric Chatbot

**The Problem Statement:**

Currently, Quality Assurance and Regulatory Affairs teams do not have an easy way to navigate 
to the information quickly if they have any questions on the guidelines. This is an incredibly 
time-consuming & tedious process. It is also critical to understand the regulations very well as 
missing a single point can lead to non-compliance, audit findings, and significant business risk.  

**My Approach**

This chatbot is built using the Retrieval-Augmented Generation (RAG) approach, allowing it to answer queries based on the contents of custom documents. The key components used are:

**=> LangChain –** for managing the retrieval and chaining logic.

**=> Ollama –** a lightweight framework to run local LLMs for generating responses.

**=> RAG (Retrieval-Augmented Generation) –** enables the chatbot to fetch relevant context from documents before answering, making responses more accurate and document-specific.

This combination allows the chatbot to understand, retrieve, and respond to user queries based on custom .txt documents.

**Frontend of The Application**

**1. GCP Regulatory Chatbot**
![GCP 1](https://github.com/user-attachments/assets/f39d66ef-e90e-43d0-a8f3-1a5789c75c08)


**2. Chatbot accurately answering based on context from text_v1.txt and text_v2.txt**


![GCP Regulatory chatbot 2](https://github.com/user-attachments/assets/f9bbb42d-d9e5-4ed0-a6b0-7b8ee8dbff8a)

**3. About this Application**

![Screenshot (1141)](https://github.com/user-attachments/assets/f2d89e0e-b697-4e76-9ae4-76b7f0e656ca)



 # System Architecture Summary

**=> Data Flow :**
**1.** **Document Ingestion:** Raw text files → Processed chunks → Vector embeddings 

**2.** **Query Processing:** User question → Vector search → Context retrieval 

**3.** **Response Generation:** Retrieved context + LLM → Formatted response 

**4.** **Source Attribution:** Chunk metadata → Source citations 



# Technology Stack

**1. Frontend:** Streamlit (Python web framework) 

**2. Backend:** Python with LangChain orchestration 

**3. AI Engine:** Local Ollama LLM (Llama 3.2) 

**4. Vector Store:** ChromaDB with HNSW indexing 

**5. Embeddings:** HuggingFace sentence-transformers


# Key Design Principles

**1. Privacy-First:** Local processing, no external API calls 

**2. Regulatory Compliance:** Audit trails and source attribution 

**3. Scalability:** Modular architecture for easy extension 

**4. User Experience:** Intuitive interface with clear feedback 

**5. Reliability:** Comprehensive error handling and validation




# Steps to Run This Chatbot Locally

**1. Initial Setup**

**=>** Run the following command to perform the initial project setup: **python setup.py**



**1.** **Set up the environment**

**=>** Install the required Python dependencies: **pip install -r requirements.txt**



**2.** **Install Ollama (if not already installed)**

**=>** Download and install Ollama from https://ollama.com based on your OS.



**3.** **Run the chatbot**

**=>** Open your terminal in the project directory and run: **python run_app.py**



   
