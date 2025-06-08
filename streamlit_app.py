import streamlit as st
import os
import json
from datetime import datetime
from typing import List, Dict
import time

# Local imports
from document_processor import DocumentProcessor
from chatbot_engine import GCPChatbot

# Page configuration
st.set_page_config(
    page_title="GCP Regulatory Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff6b6b;
    }
    
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }
    
    .source-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border: 1px solid #dee2e6;
        margin: 0.2rem 0;
    }
    
    .document-tag {
        background-color: #e9ecef;
        color: #495057;
        padding: 0.2rem 0.5rem;
        border-radius: 0.2rem;
        font-size: 0.8rem;
        margin: 0.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    def __init__(self):
        """Initialize the Streamlit application"""
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = None
        
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = False
            
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        if 'selected_documents' not in st.session_state:
            st.session_state.selected_documents = []
            
        if 'available_documents' not in st.session_state:
            st.session_state.available_documents = []
    
    def setup_documents(self):
        """Setup and process documents"""
        st.subheader("📄 Document Setup")
        
        # Check if documents are already processed
        if os.path.exists("./chroma_db"):
            st.success("✅ Documents found and loaded!")
            
            if st.session_state.chatbot is None:
                with st.spinner("Initializing chatbot..."):
                    st.session_state.chatbot = GCPChatbot()
                    success = st.session_state.chatbot.load_documents()
                    
                    if success:
                        st.session_state.documents_loaded = True
                        st.session_state.available_documents = st.session_state.chatbot.get_available_documents()
                        st.success("Chatbot initialized successfully!")
                    else:
                        st.error("Failed to initialize chatbot")
        else:
            st.warning("⚠️ Documents not found. Please process documents first.")
            
            if st.button("🔄 Process Documents"):
                with st.spinner("Processing documents... This may take a few minutes."):
                    try:
                        # Initialize document processor
                        processor = DocumentProcessor()
                        
                        # Define document paths
                        document_paths = ["Text_v1.txt", "Text_v2.txt"]
                        
                        # Check if files exist
                        missing_files = [path for path in document_paths if not os.path.exists(path)]
                        if missing_files:
                            st.error(f"Missing files: {missing_files}")
                            st.info("Please ensure Text_v1.txt and Text_v2.txt are in the same directory as this app.")
                            return
                        
                        # Process documents
                        success = processor.process_documents(document_paths)
                        
                        if success:
                            st.success("✅ Documents processed successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to process documents")
                            
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
    
    def document_selection_sidebar(self):
        """Handle document selection in sidebar"""
        with st.sidebar:
            st.header("📋 Document Selection")
            
            if st.session_state.available_documents:
                st.write("Select documents to search within:")
                
                # Document selection
                selected_docs = []
                for doc in st.session_state.available_documents:
                    if st.checkbox(doc, value=True, key=f"doc_{doc}"):
                        selected_docs.append(doc)
                
                # Update selected documents
                if selected_docs != st.session_state.selected_documents:
                    st.session_state.selected_documents = selected_docs
                    if st.session_state.chatbot:
                        st.session_state.chatbot.set_selected_documents(selected_docs)
                
                st.write(f"**Selected:** {len(selected_docs)} document(s)")
                
                # Document info
                st.subheader("📄 Document Information")
                for doc in st.session_state.available_documents:
                    with st.expander(f"ℹ️ {doc}"):
                        if "v1" in doc:
                            st.write("**ICH E6(R2)** - Good Clinical Practice Guidelines")
                            st.write("Version: E6(R2) (November 2016)")
                        elif "v2" in doc:
                            st.write("**ICH E6(R3)** - Good Clinical Practice Guidelines")
                            st.write("Version: E6(R3) Draft (May 2023)")
                        st.write("Document Type: Regulatory Guidance")
                        st.write("Category: Clinical Trial Conduct")
                
            else:
                st.info("No documents available. Please process documents first.")
            
            # Chat controls
            st.subheader("💬 Chat Controls")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                if st.session_state.chatbot:
                    st.session_state.chatbot.clear_chat_history()
                st.success("Chat history cleared!")
                st.rerun()
            
            # Export chat history
            if st.session_state.chat_history:
                if st.button("📥 Export Chat History"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"chat_history_{timestamp}.json"
                    
                    if st.session_state.chatbot:
                        st.session_state.chatbot.export_chat_history(filename)
                        st.success(f"Chat history exported to {filename}")
    

    def chat_interface(self):
        """Main chat interface"""
        st.subheader("💬 GCP Regulatory Assistant")
        
        if not st.session_state.documents_loaded or not st.session_state.chatbot:
            st.warning("Please set up documents first.")
            return
        
        # Display selected documents
        if st.session_state.selected_documents:
            st.write("**Searching within:**")
            doc_tags = " ".join([f'<span class="document-tag">{doc}</span>' 
                               for doc in st.session_state.selected_documents])
            st.markdown(doc_tags, unsafe_allow_html=True)
        else:
            st.info("No documents selected. Please select documents from the sidebar.")
            return
        
        # Chat history display
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources if available
                    if message.get('sources'):
                        with st.expander("📚 Sources", expanded=False):
                            for source in message['sources']:
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>📄 {source['filename']}</strong><br>
                                    <small>{source['content_preview']}</small>
                                </div>
                                """, unsafe_allow_html=True)
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                user_input = st.text_area(
                    "Ask a question about GCP regulations:",
                    placeholder="e.g., What are the responsibilities of a clinical investigator?",
                    height=100,
                    key="user_input"
                )
            
            with col2:
                st.write("")  # Spacer
                st.write("")  # Spacer
                submit_button = st.form_submit_button("Send 📤", use_container_width=True)
        
        # Process user input
        if submit_button and user_input.strip():
            if not st.session_state.selected_documents:
                st.error("Please select at least one document from the sidebar.")
                return
            
            # Add user message to history
            user_message = {
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.chat_history.append(user_message)
            
            # Generate response
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(user_input)
                
                # Add assistant message to history
                assistant_message = {
                    'role': 'assistant',
                    'content': response['answer'],
                    'timestamp': response.get('timestamp', datetime.now().isoformat()),
                    'sources': response.get('sources', [])
                }
                st.session_state.chat_history.append(assistant_message)
            
            # Rerun to update the display
            st.rerun()
    
    def sample_questions(self):
        """Display sample questions"""
        st.subheader("💡 Sample Questions")
        
        sample_questions = [
            "What are the main responsibilities of a clinical trial investigator according to ICH GCP?",
            "What is required for informed consent in clinical trials?",
            "What are the differences between ICH E6(R2) and E6(R3)?",
            "What is the role of the IRB/IEC in clinical trials?",
            "What are the requirements for investigational product management?",
            "What constitutes a serious adverse event (SAE)?",
            "What are the sponsor's responsibilities in clinical trials?",
            "What is the purpose of monitoring in clinical trials?",
            "What are the essential documents required for clinical trials?",
            "What are the principles of Good Clinical Practice?"
        ]
        
        # Display questions in columns
        col1, col2 = st.columns(2)
        
        for i, question in enumerate(sample_questions):
            column = col1 if i % 2 == 0 else col2
            
            with column:
                if st.button(f"❓ {question[:50]}..." if len(question) > 50 else f"❓ {question}", 
                           key=f"sample_q_{i}"):
                    # Set the question in session state for the chat interface
                    if st.session_state.documents_loaded and st.session_state.selected_documents:
                        # Add to chat history and process
                        user_message = {
                            'role': 'user',
                            'content': question,
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.chat_history.append(user_message)
                        
                        # Generate response
                        with st.spinner("Generating response..."):
                            response = st.session_state.chatbot.chat(question)
                            
                            assistant_message = {
                                'role': 'assistant',
                                'content': response['answer'],
                                'timestamp': response.get('timestamp', datetime.now().isoformat()),
                                'sources': response.get('sources', [])
                            }
                            st.session_state.chat_history.append(assistant_message)
                        
                        st.rerun()
                    else:
                        st.warning("Please set up documents and select them first.")
    
    def statistics_sidebar(self):
        """Display statistics in sidebar"""
        with st.sidebar:
            st.subheader("📊 Statistics")
            
            # Chat statistics
            total_messages = len(st.session_state.chat_history)
            user_messages = len([msg for msg in st.session_state.chat_history if msg['role'] == 'user'])
            
            st.metric("Total Messages", total_messages)
            st.metric("Questions Asked", user_messages)
            
            if st.session_state.chatbot and hasattr(st.session_state.chatbot.doc_processor, 'documents_metadata'):
                metadata = st.session_state.chatbot.doc_processor.documents_metadata
                st.metric("Document Chunks", metadata.get('total_chunks', 0))
                st.metric("Available Documents", len(metadata.get('documents', [])))
    
    def main_header(self):
        """Display main header"""
        st.markdown('<h1 class="main-header">🏥 GCP Regulatory Chatbot</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Your AI assistant for ICH Good Clinical Practice (GCP) guidelines and regulatory compliance
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def about_section(self):
        """Display about section"""
        with st.expander("ℹ️ About this Application"):
            st.markdown("""
            **GCP Regulatory Chatbot** is an AI-powered assistant designed to help Quality Assurance 
            and Regulatory Affairs professionals navigate ICH Good Clinical Practice (GCP) guidelines.
            
            **Features:**
            - 🔍 **Intelligent Search**: Find relevant information across GCP documents quickly
            - 💬 **Conversational Interface**: Ask questions in natural language
            - 📄 **Document Selection**: Choose specific documents to search within
            - 🎯 **Context Awareness**: Maintains conversation context for follow-up questions
            - 📚 **Source Citations**: See exactly where information comes from
            
            **Documents Included:**
            - ICH E6(R2): Good Clinical Practice Guidelines (November 2016)
            - ICH E6(R3): Good Clinical Practice Guidelines Draft (May 2023)
            
            **Technology Stack:**
            - 🤖 Local LLM via Ollama (Llama 3.2)
            - 🔗 LangChain for RAG implementation
            - 🗃️ ChromaDB for vector storage
            - 🎨 Streamlit for the web interface
            """)
    
    def run(self):
        """Run the Streamlit application"""
        # Main header
        self.main_header()
        
        # About section
        self.about_section()
        
        # Document setup
        self.setup_documents()
        
        # Sidebar components
        self.document_selection_sidebar()
        self.statistics_sidebar()
        
        # Main content tabs
        if st.session_state.documents_loaded:
            tab1, tab2 = st.tabs(["💬 Chat", "💡 Sample Questions"])
            
            with tab1:
                self.chat_interface()
            
            with tab2:
                self.sample_questions()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            Built with ❤️ for regulatory professionals | Powered by LangChain & Ollama
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the app"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()