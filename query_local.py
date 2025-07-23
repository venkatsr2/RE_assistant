import os
from dotenv import load_dotenv
import logging
import argparse
import pickle
import numpy as np
import faiss
import re

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata_store.pkl")

# --- INITIALIZATION ---
def initialize_services():
    """Load all necessary clients, models, and local data files."""
    print("[*] Loading services and data...")
    load_dotenv()
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key: raise ValueError("FATAL: GOOGLE_API_KEY not found in .env file.")
    genai.configure(api_key=google_api_key)
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
    except FileNotFoundError:
        print(f"\n[!] FATAL: Index files not found. Please run 'process_data.py' first.")
        return None, None, None, None

    print("[+] Services and data loaded.")
    return genai, embedding_model, faiss_index, metadata_store

# --- ANSWER GENERATION (THE FIX) ---
def get_answer_from_gemini(genai_client, query, context):
    """Uses the Gemini 1.5 Pro model to analyze a broad context and provide a specific answer."""
    # The prompt structure for Gemini is a simple list of strings.
    prompt_parts = [
        "You are a highly intelligent AI assistant for a real estate business. Your task is to answer the user's question based *only* on the provided context from internal documents.",
        "RULES:",
        "1. Carefully read the user's entire question to understand their specific intent.",
        "2. Thoroughly analyze all the provided context chunks.",
        "3. Synthesize a single, concise, and accurate answer using ONLY the information found in the context.",
        "4. Cite the source file for every piece of information using the format [source: file_name].",
        "5. If you cannot find the answer within the context, you MUST state: 'Based on the retrieved documents, I could not find a specific answer.'",
        "\n--- CONTEXT START ---\n",
        context,
        "\n--- CONTEXT END ---\n",
        "User's Question: " + query,
        "Final Answer: "
    ]
    
    try:
        model = genai_client.GenerativeModel(model_name="gemini-1.5-pro-latest")
        # The FIX is to use the correct variable name here
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"(Could not generate answer from Gemini. Error: {e})"

# --- MAIN EXECUTION ---
def main():
    parser = argparse.ArgumentParser(description="Ask a natural language question to your local real estate data.")
    parser.add_argument("query", type=str, help="Your full question, enclosed in quotes.")
    parser.add_argument("--top_k", type=int, default=100, help="Number of initial candidate documents to retrieve for the LLM to analyze.")
    args = parser.parse_args()
    
    genai_client, embedding_model, faiss_index, metadata_store = initialize_services()
    if not genai_client: return

    try:
        print(f"\n[1] Performing a broad semantic search for: \"{args.query}\"")
        query_embedding = embedding_model.encode([args.query])
        
        distances, indices = faiss_index.search(query_embedding, args.top_k)
        retrieved_indices = indices[0]
        
        if len(retrieved_indices) == 0 or retrieved_indices[0] == -1:
            print("\n[!] No relevant documents found in the initial search.")
            return

        initial_results = [metadata_store[i] for i in retrieved_indices]
        
        print("\n" + "="*80)
        print("                  INITIAL RETRIEVED CONTEXT (Top 5 for Verification)")
        print("="*80)
        for i, doc in enumerate(initial_results[:5]):
            print(f"\n--- Result {i+1} ---")
            source = doc.get('source', 'N/A')
            text_snippet = doc.get('original_text', '').strip().replace('\n', ' ')[:350]
            print(f"  - Source:  {source}")
            print(f"  - Text:    \"{text_snippet}...\"")
        print("="*80)

        context_for_llm = "\n---\n".join([f"Source: {doc.get('source', 'N/A')}\nContent: {doc.get('original_text', '')}" for doc in initial_results])
        
        print(f"\n[2] Passing {len(initial_results)} candidate chunks to Gemini 1.5 Pro for final filtering and analysis...")
        final_answer = get_answer_from_gemini(genai_client, args.query, context_for_llm)
        
        print("\n" + "="*80 + "\n                        FINAL ANSWER\n" + "="*80 + "\n")
        print(f"Answer:\n{final_answer}\n")

    except Exception as e:
        print(f"\n[!] An unexpected critical error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()