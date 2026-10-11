import os
import sys
from typing import List

# Ensure Python can find our 'src' module when running this script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from langchain_chroma import Chroma
from langchain_core.documents import Document

# Import our modular layers
from src.embeddings.provider import EmbeddingProvider
from src.ingestion.chunking import DocumentChunker

class ChromaDBManager:
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = None):
        """
        Initializes the ChromaDB Vector Store.
        """
        # Set the default save location to a 'chroma_db' folder in the project root
        if persist_directory is None:
            self.persist_directory = os.path.join(project_root, "chroma_db")
        else:
            self.persist_directory = persist_directory

        self.collection_name = collection_name
        
        # Instantiate our modular Embedding Layer (The "Librarian")
        provider = EmbeddingProvider(model_name="all-MiniLM-L6-v2")
        self.embedding_function = provider.get_embedder()
        
        print(f"Connecting to ChromaDB at: {self.persist_directory}")
        print(f"Using Collection: {self.collection_name}")
        
        # Initialize the LangChain Chroma wrapper
        # This will create the SQLite and Parquet files locally.
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents: List[Document]):
        """Passes chunks through the embedder and saves them to the Vector DB."""
        if not documents:
            print("No documents to add.")
            return
            
        print(f"Adding {len(documents)} chunks to Vector DB...")
        # Under the hood, Chroma will take our chunks, hand them to the MiniLM 
        # embedding function to get the 384d vectors, and save both to the hard drive!
        self.vector_store.add_documents(documents=documents)
        print("Successfully embedded and saved to database.")

    def get_retriever(self, k: int = 3):
        """
        Returns a retriever object for querying.
        k = the number of closest chunks to return (efSearch equivalent).
        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})

# Testing execution block
if __name__ == "__main__":
    print("--- Starting End-to-End Ingestion Test ---")
    
    # 1. Start the Chunking Layer
    data_dir = os.path.join(project_root, "data")
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.process_directory(data_dir)
    
    if chunks:
        # 2. Start the Vector DB Layer (which automatically triggers the Embedding Layer)
        # Notice we name the collection specifically so we know it expects 384 dimensions!
        db_manager = ChromaDBManager(collection_name="pdfs_minilm_384")
        
        # This one line takes the English chunks, turns them into math, and saves them to disk.
        db_manager.add_documents(chunks)
        
        # 3. Quick Retrieval Test (The Librarian in action!)
        print("\n--- Testing Retrieval Search ---")
        retriever = db_manager.get_retriever(k=1)
        test_query = "Summarize the key points of the document."
        
        print(f"Searching for: '{test_query}'")
        results = retriever.invoke(test_query)
        
        if results:
            print(f"\nTop Match found! Handing this context to the Professor (LLM):")
            print("-" * 50)
            print(results[0].page_content)
            print("-" * 50)
            print(f"Metadata (from SQLite): {results[0].metadata}")
    else:
        print(f"No PDFs found in {data_dir}. Cannot test Vector DB insertion.")
