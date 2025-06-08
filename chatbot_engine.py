import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import logging

# LangChain imports
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseRetriever, Document

# Local imports
from document_processor import DocumentProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Data class for chat messages"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = None

class GCPChatbot:
    def __init__(self, 
                 model_name: str = "llama3.2",  # or "mistral", "phi3", etc.
                 temperature: float = 0.1,
                 max_tokens: int = 2048,
                 memory_window: int = 10):
        """
        Initialize the GCP Chatbot
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            memory_window: Number of conversation turns to remember
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor()
        
        # Initialize LLM
        self.llm = None
        self.chain = None
        self.memory = None
        self.selected_documents = []
        
        # Chat history
        self.chat_history = []
        
        self._initialize_llm()
        self._setup_prompt_template()
        
    def _initialize_llm(self):
        """Initialize the Ollama LLM"""
        try:
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            self.llm = Ollama(
                model=self.model_name,
                temperature=self.temperature,
                callback_manager=callback_manager,
                # num_predict=self.max_tokens,
                verbose=True
            )
            
            logger.info(f"Initialized LLM: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def _setup_prompt_template(self):
        """Setup the prompt template for RAG"""
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""You are a specialized AI assistant for Good Clinical Practice (GCP) and pharmaceutical regulatory compliance. 
You help Quality Assurance and Regulatory Affairs professionals understand ICH GCP guidelines and regulations.

INSTRUCTIONS:
1. Use ONLY the provided context from the ICH GCP documents to answer questions
2. Be precise, accurate, and cite specific sections when possible
3. If information is not in the provided context, clearly state this
4. Maintain a professional tone appropriate for regulatory professionals
5. Consider the conversation history for context

CONTEXT FROM ICH GCP DOCUMENTS:
{context}

CONVERSATION HISTORY:
{chat_history}

CURRENT QUESTION: {question}

ANSWER:
Please provide a comprehensive answer based on the ICH GCP guidelines. If you reference specific sections, please mention them. If the information is not available in the provided context, please state this clearly.

"""
        )
    
    def load_documents(self, persist_directory: str = "./chroma_db") -> bool:
        """
        Load the processed documents and initialize the retrieval chain
        
        Args:
            persist_directory: Directory containing the vector store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load vector store
            success = self.doc_processor.load_vector_store(persist_directory)
            if not success:
                logger.error("Failed to load vector store")
                return False
            
            # Initialize memory
            self.memory = ConversationBufferWindowMemory(
                k=self.memory_window,
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            )
            
            # Create retrieval chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.doc_processor.vector_store.as_retriever(search_kwargs={"k": 5}),
                memory=self.memory,
                return_source_documents=True,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": self.prompt_template}
            )
            
            logger.info("Chatbot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return False
    
    def set_selected_documents(self, document_names: List[str]):
        """
        Set which documents to search within
        
        Args:
            document_names: List of document names to restrict search to
        """
        available_docs = self.doc_processor.get_available_documents()
        self.selected_documents = [doc for doc in document_names if doc in available_docs]
        
        if self.selected_documents:
            # Update retriever with document filter
            self.chain.retriever = self._create_filtered_retriever(self.selected_documents)
            logger.info(f"Selected documents: {self.selected_documents}")
        else:
            logger.warning("No valid documents selected, using all available documents")
    
    def _create_filtered_retriever(self, document_names: List[str]):
        """
        Create a retriever that filters by selected documents
        
        Args:
            document_names: List of document names to filter by
            
        Returns:
            Filtered retriever
        """
        class FilteredRetriever(BaseRetriever):
            def __init__(self, vector_store, doc_names, k=5):
                self.vector_store = vector_store
                self.doc_names = doc_names
                self.k = k
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self.vector_store.similarity_search(
                    query, 
                    k=self.k,
                    filter={"filename": {"$in": self.doc_names}}
                )
        
        return FilteredRetriever(self.doc_processor.vector_store, document_names)
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and generate response
        
        Args:
            user_input: User's question or message
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            if not self.chain:
                return {
                    "answer": "Chatbot not initialized. Please load documents first.",
                    "sources": [],
                    "error": "Not initialized"
                }
            
            # Process the query
            result = self.chain({
                "question": user_input,
                "chat_history": self._get_chat_history_string()
            })
            
            # Extract sources
            sources = []
            if 'source_documents' in result:
                sources = [
                    {
                        "filename": doc.metadata.get('filename', 'Unknown'),
                        "chunk_id": doc.metadata.get('chunk_id', 'Unknown'),
                        "content_preview": doc.page_content[:200] + "..."
                    }
                    for doc in result['source_documents']
                ]
            
            # Save to chat history
            user_message = ChatMessage(
                role="user",
                content=user_input,
                timestamp=datetime.now()
            )
            
            assistant_message = ChatMessage(
                role="assistant",
                content=result['answer'],
                timestamp=datetime.now(),
                sources=[s['filename'] for s in sources]
            )
            
            self.chat_history.extend([user_message, assistant_message])
            
            return {
                "answer": result['answer'],
                "sources": sources,
                "selected_documents": self.selected_documents,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _get_chat_history_string(self) -> str:
        """
        Get formatted chat history string for context
        
        Returns:
            Formatted chat history
        """
        if not self.chat_history:
            return "No previous conversation."
        
        # Get last few messages
        recent_messages = self.chat_history[-6:]  # Last 3 exchanges
        
        history_parts = []
        for msg in recent_messages:
            role = "Human" if msg.role == "user" else "Assistant"
            history_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(history_parts)
    
    def get_available_documents(self) -> List[str]:
        """
        Get list of available documents
        
        Returns:
            List of document names
        """
        return self.doc_processor.get_available_documents()
    
    def clear_chat_history(self):
        """Clear the chat history and memory"""
        self.chat_history = []
        if self.memory:
            self.memory.clear()
        logger.info("Chat history cleared")
    
    def get_chat_history(self) -> List[Dict]:
        """
        Get chat history in JSON serializable format
        
        Returns:
            List of chat messages
        """
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "sources": msg.sources
            }
            for msg in self.chat_history
        ]
    
    def export_chat_history(self, filename: str):
        """
        Export chat history to JSON file
        
        Args:
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.get_chat_history(), f, indent=2)
            logger.info(f"Chat history exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting chat history: {str(e)}")


def main():
    """
    Example usage of the GCPChatbot
    """
    # Initialize chatbot
    chatbot = GCPChatbot(model_name="llama3.2")
    
    # Load documents
    if not chatbot.load_documents():
        print("Failed to load documents")
        return
    
    print("GCP Chatbot initialized successfully!")
    print(f"Available documents: {chatbot.get_available_documents()}")
    
    # Example interaction
    print("\n" + "="*50)
    print("Example Conversation:")
    print("="*50)
    
    # Set selected documents
    chatbot.set_selected_documents(["Text_v1.txt", "Text_v2.txt"])
    
    # Example questions
    questions = [
        "What are the main responsibilities of a clinical trial investigator?",
        "What is Good Clinical Practice according to ICH guidelines?",
        "What are the requirements for informed consent in clinical trials?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        response = chatbot.chat(question)
        print(f"Assistant: {response['answer'][:300]}...")
        if response['sources']:
            print(f"Sources: {[s['filename'] for s in response['sources']]}")


if __name__ == "__main__":
    main()