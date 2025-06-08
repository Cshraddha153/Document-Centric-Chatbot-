import os
import json
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.schema import Document
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the document processor with configurable parameters
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            embedding_model: HuggingFace embedding model name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize vector store
        self.vector_store = None
        self.documents_metadata = {}
        
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from file paths
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    continue
                    
                # Load document
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        'source': file_path,
                        'filename': os.path.basename(file_path),
                        'doc_type': 'regulation'
                    })
                
                documents.extend(docs)
                logger.info(f"Loaded document: {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        for doc in documents:
            doc_chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    'chunk_id': f"{chunk.metadata['filename']}_{i}",
                    'chunk_index': i,
                    'total_chunks': len(doc_chunks)
                })
            
            chunks.extend(doc_chunks)
            
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def create_vector_store(self, chunks: List[Document], persist_directory: str = "./chroma_db"):
        """
        Create and persist vector store
        
        Args:
            chunks: Document chunks to embed
            persist_directory: Directory to persist the vector store
        """
        try:
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            
            # Persist the database
            self.vector_store.persist()
            
            # Store metadata
            self.documents_metadata = {
                'total_chunks': len(chunks),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'embedding_model': self.embedding_model,
                'documents': list(set([chunk.metadata['filename'] for chunk in chunks]))
            }
            
            # Save metadata
            with open(os.path.join(persist_directory, 'metadata.json'), 'w') as f:
                json.dump(self.documents_metadata, f, indent=2)
            
            logger.info(f"Vector store created with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vector_store(self, persist_directory: str = "./chroma_db"):
        """
        Load existing vector store
        
        Args:
            persist_directory: Directory containing the persisted vector store
        """
        try:
            if not os.path.exists(persist_directory):
                logger.error(f"Vector store directory not found: {persist_directory}")
                return False
                
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            
            # Load metadata
            metadata_path = os.path.join(persist_directory, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.documents_metadata = json.load(f)
            
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def search_documents(self, query: str, k: int = 5, filter_docs: List[str] = None) -> List[Document]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter_docs: Optional list of document filenames to filter by
            
        Returns:
            List of relevant document chunks
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Prepare filter if specified
            where_filter = None
            if filter_docs:
                where_filter = {"filename": {"$in": filter_docs}}
            
            # Perform similarity search
            if where_filter:
                results = self.vector_store.similarity_search(
                    query, 
                    k=k, 
                    filter=where_filter
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_available_documents(self) -> List[str]:
        """
        Get list of available documents
        
        Returns:
            List of document filenames
        """
        if self.documents_metadata:
            return self.documents_metadata.get('documents', [])
        return []
    
    def process_documents(self, file_paths: List[str], persist_directory: str = "./chroma_db"):
        """
        Complete document processing pipeline
        
        Args:
            file_paths: List of document file paths
            persist_directory: Directory to persist vector store
        """
        logger.info("Starting document processing pipeline...")
        
        # Load documents
        documents = self.load_documents(file_paths)
        if not documents:
            logger.error("No documents loaded")
            return False
        
        # Split documents
        chunks = self.split_documents(documents)
        if not chunks:
            logger.error("No chunks created")
            return False
        
        # Create vector store
        self.create_vector_store(chunks, persist_directory)
        
        logger.info("Document processing completed successfully")
        return True


def main():
    """
    Example usage of the DocumentProcessor
    """
    # Initialize processor
    processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Define document paths
    document_paths = [
        "Text_v1.txt",  # ICH E6(R2)
        "Text_v2.txt"   # ICH E6(R3)
    ]
    
    # Process documents
    success = processor.process_documents(document_paths)
    
    if success:
        print("Documents processed successfully!")
        print(f"Available documents: {processor.get_available_documents()}")
        
        # Test search
        test_query = "What are the responsibilities of the investigator?"
        results = processor.search_documents(test_query, k=3)
        
        print(f"\nTest search for: '{test_query}'")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Source: {result.metadata.get('filename', 'Unknown')}")
            print(f"Content: {result.page_content[:200]}...")
    else:
        print("Failed to process documents")


if __name__ == "__main__":
    main()