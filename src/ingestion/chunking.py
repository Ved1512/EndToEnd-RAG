import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """ 
        Initializes the chunker.
        We use from_tiktoken_encoder for Token-based chunking instead of Character-based.
        This ensures our chunks respect the token limits of our embedding models.
        """
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Recursive strategy: prioritize splitting on paragraphs, then lines, then spaces
            separators=["\n\n", "\n", " ", ""]
        )

    def load_and_chunk_pdf(self, file_path: str) -> List[Document]:
        """Loads a single PDF and splits it into semantic token chunks."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found at {file_path}")
            
        print(f"Loading PDF: {file_path}")
        # PyPDFLoader extracts the text while maintaining page number metadata
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        print(f"Loaded {len(pages)} pages. Starting chunking process...")
        chunks = self.text_splitter.split_documents(pages)
        
        print(f"Successfully split into {len(chunks)} chunks.")
        return chunks

    def process_directory(self, directory_path: str) -> List[Document]:
        """Processes all PDFs in a given directory."""
        all_chunks = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                chunks = self.load_and_chunk_pdf(file_path)
                all_chunks.extend(chunks)
        return all_chunks

# Testing execution block
if __name__ == "__main__":
    # Resolve the data directory (assuming e:/Projects/EndToEndRAG/data)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    data_dir = os.path.join(project_root, "data")
    
    if os.path.exists(data_dir):
        # 300 tokens is a very safe size for local open-source embedding models
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        documents = chunker.process_directory(data_dir)
        
        if documents:
            print(f"\nTotal chunks generated across all PDFs: {len(documents)}")
            print("Preview of the first chunk metadata:", documents[0].metadata)
            print("Preview of the first chunk content:")
            print("-" * 50)
            print(documents[0].page_content)
            print("-" * 50)
        else:
            print(f"No PDFs found in {data_dir}. Put your PDFs here!")
    else:
        print(f"Data directory not found at {data_dir}. Creating one now...")
        os.makedirs(data_dir, exist_ok=True)
        print("Please place your PDFs inside the newly created 'data' directory and run again.")
