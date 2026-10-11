import os
import sys

# Ensure Python can find our 'src' module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.ingestion.chunking import DocumentChunker
from src.db.vector_store import ChromaDBManager

def main():
    print("--- Starting RAG Ingestion Pipeline ---")
    
    # Target the 'pdf' folder in the root directory (as requested)
    pdf_dir = os.path.join(current_dir, "PDFs")
    
    if not os.path.exists(pdf_dir):
        print(f"Error: The directory '{pdf_dir}' does not exist.")
        print("Please create a 'pdf' folder in the root and add your PDFs.")
        return

    # 1. Chunking Layer
    print("\n[Step 1/3] Chunking PDFs...")
    # Using 300 tokens chunk size and 50 tokens overlap
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.process_directory(pdf_dir)
    
    if not chunks:
        print(f"No PDFs found in '{pdf_dir}'. Please add some PDFs and try again.")
        return

    # 2. Vector DB & Embedding Layer
    print("\n[Step 2/3] Generating Embeddings and Saving to ChromaDB...")
    # This automatically calls our HuggingFace MiniLM model and saves to the local DB
    db_manager = ChromaDBManager(collection_name="pdfs_minilm_384")
    db_manager.add_documents(chunks)
    
    # 3. Done
    print("\n[Step 3/3] Success! All PDFs have been successfully embedded and stored.")
    print("Your Vector Database is fully populated and ready!")
    print("You can now start your FastAPI backend using: uvicorn src.api.main:app --reload")

if __name__ == "__main__":
    main()
