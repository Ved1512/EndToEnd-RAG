import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Ensure Python can find our 'src' module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our Vector DB Orchestrator
from src.db.vector_store import ChromaDBManager

# Import LangChain modules for the LLM
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- 1. FastAPI Setup ---
app = FastAPI(title="RAG Backend API", description="API for React Frontend to interact with the RAG pipeline.")

# Add CORS Middleware so your React app (e.g., localhost:3000) can securely talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins during local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Initialize Models & DB ---
# Initialize the Vector DB (The Librarian)
db_manager = ChromaDBManager(collection_name="pdfs_minilm_384")
retriever = db_manager.get_retriever(k=3)

# Initialize the Generative LLM (The Professor)
# We are using Groq (Option B) for lightning-fast, free Llama-3 generation.
# Make sure to set the GROQ_API_KEY environment variable.
llm = ChatGroq(
    model="openai/gpt-oss-20b   ", # 128k context window, fast & free generative LLM
    temperature=0.2 # Low temperature for factual, non-creative answers
)

# --- 3. Prompt Template ---
system_prompt = """You are an intelligent assistant. Use the following pieces of retrieved context to answer the question. 
If the answer is not in the context, just say "I cannot find the answer in the provided documents." Do not make up information.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(system_prompt)

# --- 4. API Data Models (For React JS) ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

# --- 5. The Core Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives a query from the React frontend, retrieves context from ChromaDB,
    generates an answer using Groq (Llama-3), and returns the JSON payload.
    """
    try:
        # Step A: Retrieve the closest chunks from the Vector Database
        docs = retriever.invoke(request.query)
        
        # Step B: Extract the raw text and metadata for the prompt
        context_text = "\n\n".join([f"--- Chunk ---\n{doc.page_content}" for doc in docs])
        
        # Extract source filenames so the frontend can display "Sources used: report.pdf"
        sources = [doc.metadata.get('source', 'Unknown Document') for doc in docs]
        unique_sources = list(set(sources)) # Remove duplicates
        
        # Step C: Construct the final prompt with the Context + User Query
        messages = prompt.format_messages(context=context_text, question=request.query)
        
        # Step D: Call the Generative LLM (Groq)
        response = llm.invoke(messages)
        
        # Step E: Return the JSON payload back to the React frontend
        return ChatResponse(
            answer=response.content,
            sources=unique_sources
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc() # This will print the exact error to your Python terminal!
        raise HTTPException(status_code=500, detail=str(e))

# Note: To run this server, you would execute:
# uvicorn src.api.main:app --reload
