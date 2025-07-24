import os
from dotenv import load_dotenv
import logging
import argparse
import pickle
import numpy as np
import faiss

from openai import OpenAI
import tiktoken

# --- CONFIGURATION ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata_store.pkl")
CONTEXT_TOKEN_LIMIT = 100000
EMBEDDING_MODEL = "text-embedding-3-small"

# --- INITIALIZATION ---
def initialize_services():
    """Load OpenAI client, FAISS index, and metadata."""
    print("[*] Loading services and data...")
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key: raise ValueError("FATAL: OPENAI_API_KEY not found.")
    
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
    except FileNotFoundError:
        print(f"\n[!] FATAL: Index files not found. Please run 'process_data.py' first.")
        return None, None, None

    print("[+] Services and data loaded.")
    return client, faiss_index, metadata_store

# --- ANSWER GENERATION ---
def get_answer_from_gpt(client, query, context):
    """Uses gpt-4o to analyze context and provide an answer."""
    system_prompt = """
    You are a highly intelligent AI assistant for a real estate business. You will be given a user's question and a broad set of retrieved text chunks that may or may not contain the answer.

    Your task is to act as an expert analyst:
    1.  Carefully read the user's entire question to understand their specific intent, including any names, dates, conditions, or document types (like "email" or "whatsapp chat").
    2.  Thoroughly analyze all the provided context chunks. Internally filter this context to find only the chunks that match the user's specific conditions.
    3.  Synthesize a single, concise, and accurate answer using ONLY the information found in the filtered context.
    4.  Cite the source file for every piece of information using the format [source: file_name].
    5.  If, after careful analysis, you cannot find the answer, you MUST state: "Based on the retrieved documents, I could not find a specific answer to your question."
    """
    user_prompt = f"Retrieved Context:\n---\n{context}\n---\n\nUser's Original Question:\n{query}\n\nFinal Answer:"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0)
        return response.choices[0].message.content
    except Exception as e:
        return f"(Could not generate answer from gpt-4o. Error: {e})"

# --- MAIN EXECUTION ---
def main():
    parser = argparse.ArgumentParser(description="Ask a natural language question to your local real estate data.")
    parser.add_argument("query", type=str, help="Your full question, enclosed in quotes.")
    parser.add_argument("--top_k", type=int, default=100, help="Number of initial candidate documents to retrieve for the LLM to analyze.")
    args = parser.parse_args()
    
    client, faiss_index, metadata_store = initialize_services()
    if not client: return

    try:
        print(f"\n[1] Performing a broad semantic search for: \"{args.query}\"")
        
        response = client.embeddings.create(input=[args.query], model=EMBEDDING_MODEL)
        query_embedding = np.array([response.data[0].embedding]).astype('float32')
        
        distances, indices = faiss_index.search(query_embedding, args.top_k)
        retrieved_indices = indices[0]
        
        if len(retrieved_indices) == 0 or retrieved_indices[0] == -1:
            print("\n[!] No relevant documents found in the initial search.")
            return

        retrieved_metadatas = [metadata_store[i] for i in retrieved_indices]
        
        print("\n" + "="*80)
        print("                  INITIAL RETRIEVED CONTEXT (Top 5 for Verification)")
        print("="*80)
        for i, doc in enumerate(retrieved_metadatas[:5]):
            print(f"\n--- Result {i+1} ---")
            print(f"  - Source:  {doc.get('source', 'N/A')}")
            if 'sender' in doc: print(f"  - Sender:  {doc.get('sender')}")
            if 'from' in doc: print(f"  - From:    {doc.get('from')}")
            if 'subject' in doc: print(f"  - Subject: {doc.get('subject')}")
            text_snippet = doc.get('original_text', '').strip().replace('\n', ' ')[:250]
            print(f"  - Text:    \"{text_snippet}...\"")
        print("="*80)

        # Build and truncate context
        tokenizer = tiktoken.get_encoding("cl100k_base")
        context_for_llm, total_tokens, included_chunks = "", 0, 0
        
        base_prompt_tokens = len(tokenizer.encode(f"User's Question: {args.query}")) + 300
        token_budget = CONTEXT_TOKEN_LIMIT - base_prompt_tokens

        for doc in retrieved_metadatas:
            chunk_text = f"Source: {doc.get('source', 'N/A')}\nContent: {doc.get('original_text', '')}\n---\n"
            chunk_tokens = len(tokenizer.encode(chunk_text))
            if total_tokens + chunk_tokens > token_budget: break
            context_for_llm += chunk_text
            total_tokens += chunk_tokens
            included_chunks += 1
        
        print(f"\n[2] Passing {included_chunks} of {len(retrieved_metadatas)} chunks ({total_tokens} tokens) to gpt-4o for final analysis...")
        final_answer = get_answer_from_gpt(client, args.query, context_for_llm)
        
        print("\n" + "="*80 + "\n                        FINAL ANSWER\n" + "="*80 + "\n")
        print(f"Answer:\n{final_answer}\n")

    except Exception as e:
        print(f"\n[!] An unexpected critical error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()