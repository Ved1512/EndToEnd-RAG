import os
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingProvider:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the local embedding model.
        By default, uses 'all-MiniLM-L6-v2' which is highly optimized, 
        small, fast, and outputs 384-dimensional vectors.
        """
        self.model_name = model_name
        
        print(f"Initializing Embedding Model: {self.model_name}...")
        
        # The HuggingFaceEmbeddings class automatically downloads the model 
        # from HuggingFace and runs it locally on your CPU/GPU.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            # Use 'cpu' for standard laptops. If you have an Nvidia GPU, you can change this to 'cuda'
            model_kwargs={'device': 'cpu'}, 
            
            # Normalizing embeddings ensures that all vectors have a length of 1. 
            # This is mathematically required to make Cosine Similarity search work perfectly.
            encode_kwargs={'normalize_embeddings': True} 
        )
        print("Embedding model loaded successfully.")

    def get_embedder(self) -> HuggingFaceEmbeddings:
        """Returns the configured embedding object to be consumed by the Vector DB."""
        return self.embeddings

# Testing execution block
if __name__ == "__main__":
    # 1. Instantiate the layer
    provider = EmbeddingProvider()
    embedder = provider.get_embedder()
    
    # 2. Let's do a quick dry run to see the math in action
    test_text = "The company's Q3 revenue was $2M."
    print(f"\nPassing text through the model: '{test_text}'")
    
    vector = embedder.embed_query(test_text)
    
    # 3. Print the results to verify the dimensionality (columns)
    print(f"Success! The model converted the text into a vector.")
    print(f"Vector Dimension (Number of Columns): {len(vector)}")
    print(f"First 5 values of the vector array: {vector}")
