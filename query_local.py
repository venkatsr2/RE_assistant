import os
from dotenv import load_dotenv
import logging
import argparse
import pickle
import numpy as np
import faiss
import re
import json
from datetime import datetime, timedelta
from openai import OpenAI
import tiktoken

# --- CONFIGURATION ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata_store.pkl")
CONTEXT_TOKEN_LIMIT = 100000

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

# --- QUERY DECOMPOSER (Unchanged, it is correct) ---
def decompose_query_with_llm(client, user_query, metadata_keys):
    """Uses an LLM to decompose a query into a structured search plan."""
    today_str = datetime.now().strftime('%Y-%m-%d')
    system_prompt = f"""
    You are an expert query analyzer. Decompose a user's query into a structured JSON object.
    Today's date is {today_str}.

    ## AVAILABLE METADATA FIELDS FOR FILTERING:
    These are the only fields you can use: {', '.join(f"'{key}'" for key in metadata_keys)}.
    The `timestamp` and `date` fields are strings; you must create filters for them.
    
    Decompose into:
    1.  `semantic_query`: The core topic to search for.
    2.  `metadata_filter`: A dictionary of key-value pairs to filter on.

    RULES:
    - Your response MUST be ONLY the single JSON object.
    - If no filters are mentioned, `metadata_filter` must be an empty dictionary {{}}.
    - For emails, the sender is `from`. For WhatsApp, the sender is `sender`.
    - For document names, use the `source` field.
    - If the query contains a date or a relative time period (e.g., "last month", "this week", "in July 2024"), create a date filter using the `timestamp` or `date` field.
    - A date filter should be a dictionary with `"$gte"` (start date) and/or `"$lte"` (end date) in "YYYY-MM-DD" format.

    EXAMPLE:
    User Query: "What is the sentiment of venkata satya in the Houston carpool whatsapp chat?"
    Your JSON:
    {{
      "semantic_query": "sentiment analysis",
      "metadata_filter": {{
        "sender": "venkat",
        "source": "Houston carpool whatsapp"
      }}
    }}

    User Query: "What is the sentiment of venkata satya in the Houston carpool whatsapp chat last month?"
    Your JSON:
    {{
      "semantic_query": "sentiment analysis of venkata satya's messages",
      "metadata_filter": {{
        "sender": "venkata satya",
        "source": "Houston carpool whatsapp",
        "timestamp": {{
          "$gte": "{ (datetime.now().replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d') }",
          "$lte": "{ (datetime.now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d') }"
        }}
      }}
    }}

    User Query: "Find emails from customer communications this week"
    Your JSON:
    {{
      "semantic_query": "emails from customer communications",
      "metadata_filter": {{
        "from": "customer.communications",
        "date": {{ "$gte": "{ (datetime.now() - timedelta(days=datetime.now().weekday())).strftime('%Y-%m-%d') }" }}
      }}
    }}
    """
    try:
        # Using a powerful model for better decomposition
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"\n[!] Could not decompose query. Falling back to simple search. Error: {e}")
        return {"semantic_query": user_query, "metadata_filter": {}}
    
# --- NEW: ROBUST DATE PARSING FUNCTION (THE FIX) ---
def parse_flexible_date(date_string):
    """
    Tries multiple common date formats to parse a date string.
    Returns a datetime object or None.
    """
    if not isinstance(date_string, str):
        return None
        
    # List of formats to try, from most to least specific
    formats_to_try = [
        '%m/%d/%y, %H:%M',           # WhatsApp format: 6/18/24, 16:08
        '%A, %d %B, %Y %I.%M %p',    # Email format: Monday, 10 March, 2025 11.27 AM
        '%m/%d/%Y, %H:%M',          # WhatsApp format with 4-digit year
        '%Y-%m-%dT%H:%M:%S',         # ISO format without timezone
        '%Y-%m-%d',                 # Date only
    ]
    
    for fmt in formats_to_try:
        try:
            # Strip any potential whitespace
            return datetime.strptime(date_string.strip(), fmt)
        except (ValueError, TypeError):
            continue # Try the next format
            
    # If all formats fail, return None
    return None

# --- FUZZY METADATA FILTERING (THE DEFINITIVE FIX) ---
def apply_fuzzy_filter_and_score(metadata_store, filter_dict):
    """
    Scores every document based on how well its metadata matches the filter plan.
    This handles partial matches for names and sources.
    """
    if not filter_dict:
        # If no filter, all documents are candidates with a neutral score
        return list(range(len(metadata_store)))
    
    candidate_scores = []
    # Normalize filter values once
    filter_values = {key: (str(value).lower().split() if key not in ['timestamp', 'date'] else value) for key, value in filter_dict.items()}
    for i, metadata in enumerate(metadata_store):
        score = 0
        match_count = 0
        for key, query_parts in filter_values.items():
            metadata_value = metadata.get(key)
            if metadata_value is not None:
                # --- DATE FILTERING LOGIC (Using the robust parser) ---
                if key in ['timestamp', 'date'] and isinstance(query_parts, dict):
                    doc_date = parse_flexible_date(metadata_value)
                    if doc_date:    
                        try:
                            if "$gte" in query_parts:
                                start_date = datetime.strptime(query_parts["$gte"], '%Y-%m-%d')
                                if doc_date < start_date:
                                    match_count = 0
                                    continue
                            if "$lte" in query_parts:
                                end_date = datetime.strptime(query_parts["$lte"], '%Y-%m-%d')
                                # We only check the date part, ignore time for lte
                                if doc_date.date() > end_date.date(): 
                                    match_count = 0
                                    continue
                        except (ValueError, TypeError):
                            continue # Skip if filter date is malformed
                else:    
                    metadata_value_lower = str(metadata_value).lower()
                    # Check if all parts of the query value are in the metadata value
                    if all(part in metadata_value_lower for part in query_parts):
                        score += 1 # Add 1 point for each matching key
                        match_count += 1

        # Only consider documents that matched at least one filter condition
        if match_count > 0:
            # We can add more sophisticated scoring here later if needed
            candidate_scores.append({'index': i, 'score': score})

    # Sort candidates by their score (higher is better)
    candidate_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Return the indices of the sorted candidates
    # print(f"inside fuzzy filter and score, Found {len(candidate_scores)} candidates after fuzzy filtering.")
    return [item['index'] for item in candidate_scores]

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
    5.  For sentiment analysis, focus on the overall tone and key phrases. Try to determine if the sentiment is positive, negative, or neutral based on the context.
    6.  Only in cases where the context is completely irrelevant, answer with - "Based on the retrieved documents, I could not find a specific answer to your question.". Try to avoid this response unless absolutely necessary.
    """
    user_prompt = f"Retrieved Context:\n---\n{context}\n---\n\nUser's Original Question:\n{query}\n\nFinal Answer:"
    # print(f"\n[3] Asking gpt-4o for final answer to: \"{context}\"")
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
    parser.add_argument("--top_k", type=int, default=100, help="Number of documents to retrieve/re-rank.")
    parser.add_argument("--candidates", type=int, default=500, help="Number of initial candidates to consider from metadata filter.")
    args = parser.parse_args()
    
    client, faiss_index, metadata_store = initialize_services()
    if not client: return

    try:
        available_keys = list(metadata_store[0].keys()) if metadata_store else []
        
        print(f"\n[1] Decomposing your query with LLM: \"{args.query}\"")
        search_plan = decompose_query_with_llm(client, args.query, available_keys)
        
        semantic_query = search_plan.get("semantic_query", args.query)
        metadata_filter = search_plan.get("metadata_filter", {})

        print(f"    - Semantic Search For: \"{semantic_query}\"")
        if metadata_filter: print(f"    - Applying Fuzzy Filter: {json.dumps(metadata_filter)}")
        else: print("    - No specific filters identified.")
        
        # STAGE 1: FUZZY METADATA-FIRST FILTERING
        candidate_indices = apply_fuzzy_filter_and_score(metadata_store, metadata_filter)
        # Limit the number of candidates to avoid huge in-memory indexes
        candidate_indices = candidate_indices[:args.candidates]
        print(f"    - Found {len(candidate_indices)} candidate chunks after fuzzy filtering.")

        if not candidate_indices:
            print("\n[!] No documents found matching your filter criteria.")
            return

        # STAGE 2: SEMANTIC RE-RANKING
        print(f"[2] Performing semantic re-ranking on candidates...")
        candidate_vectors = np.array([faiss_index.reconstruct(i) for i in candidate_indices]).astype('float32')
        if candidate_vectors.size == 0:
            print("\n[!] No vectors found for candidate chunks.")
            return
            
        temp_index = faiss.IndexFlatL2(candidate_vectors.shape[1])
        temp_index.add(candidate_vectors)
        
        response = client.embeddings.create(input=[semantic_query], model="text-embedding-3-small")
        query_embedding = np.array([response.data[0].embedding]).astype('float32')
        
        distances, temp_indices = temp_index.search(query_embedding, k=min(args.top_k, len(candidate_indices)))
        
        original_indices = [candidate_indices[i] for i in temp_indices[0] if i != -1]
        final_results = [metadata_store[i] for i in original_indices]
        
        print("\n" + "="*80)
        print("                  INITIAL RETRIEVED CONTEXT (Top 5 for Verification)")
        print("="*80)
        for i, doc in enumerate(final_results[:5]):
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

        for doc in final_results:
            chunk_text = f"Source: {doc.get('source', 'N/A')}\nContent: {doc.get('original_text', '')}\n---\n"
            chunk_tokens = len(tokenizer.encode(chunk_text))
            if total_tokens + chunk_tokens > token_budget: break
            context_for_llm += chunk_text
            total_tokens += chunk_tokens
            included_chunks += 1
        # print(f"\n[3] Context for llm: {context_for_llm}") #for debugging
        print(f"\n[2] Passing {included_chunks} of {len(final_results)} chunks ({total_tokens} tokens) to gpt-4o for final analysis...")
        final_answer = get_answer_from_gpt(client, args.query, context_for_llm)
        
        print("\n" + "="*80 + "\n                        FINAL ANSWER\n" + "="*80 + "\n")
        print(f"Answer:\n{final_answer}\n")

    except Exception as e:
        print(f"\n[!] An unexpected critical error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()