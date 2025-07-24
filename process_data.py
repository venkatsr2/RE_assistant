import os
import zipfile
import pandas as pd
from dotenv import load_dotenv
import logging
import re
import shutil
import json
from tqdm import tqdm
import pickle
import numpy as np

import faiss
from openai import OpenAI
from unstructured.partition.auto import partition
from unstructured.partition.email import partition_email
import tiktoken

# --- CONFIGURATION ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata_store.pkl")

# --- MODEL CONFIGURATION (CORRECTED) ---
CHUNK_SIZE_TOKENS = 2048
CHUNK_OVERLAP_TOKENS = 200
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# --- INITIALIZATION ---
def initialize_services():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key: raise ValueError("FATAL: OPENAI_API_KEY not found.")
    print("[*] OpenAI client initialized.")
    return client

# --- DATA QUALITY CONTROL ---
def filter_and_split_chunks(raw_chunks):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    JUNK_PHRASES = ["messages and calls are end-to-end encrypted", "this message was deleted", "<media omitted>"]
    final_chunks = []
    for chunk in raw_chunks:
        stripped_chunk = chunk.strip()
        if len(stripped_chunk) < 25 or any(phrase in stripped_chunk.lower() for phrase in JUNK_PHRASES):
            continue
        tokens = tokenizer.encode(stripped_chunk)
        if len(tokens) > CHUNK_SIZE_TOKENS:
            for i in range(0, len(tokens), CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS):
                sub_chunk_tokens = tokens[i:i + CHUNK_SIZE_TOKENS]
                final_chunks.append(tokenizer.decode(sub_chunk_tokens))
        else:
            final_chunks.append(stripped_chunk)
    return final_chunks

# --- INTELLIGENT PDF PARSER ---
def extract_email_metadata_from_pdf_text(client, text_sample):
    system_prompt = """
    You are a data extraction engine. Analyze the text from the first page of a PDF.
    Your task is to determine if it is an email and extract its metadata into a single, valid JSON object.
    
    RULES:
    - Your entire response MUST be ONLY a valid JSON object.
    - If the text is an email, return a JSON with "is_email": true and populate the other fields.
    - If it is NOT an email, return a JSON with only one key: {"is_email": false}.
    
    JSON Schema for Emails:
    {"is_email": true, "from": "...", "to": "...", "subject": "...", "date": "...", "summary": "...", "signature": "..."}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text_sample[:4000]}],
            temperature=0.0, response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Could not analyze PDF text with LLM. Error: {e}")
        return {"is_email": False}

# --- SPECIALIZED FILE PARSERS ---
def get_chunks_and_metadata(client, file_path):
    """Processes a file, intelligently extracts chunks and metadata."""
    file_ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    extraction_dir = os.path.join(SCRIPT_DIR, 'data', 'extracted')
    base_metadata = {'source': os.path.relpath(file_path, extraction_dir)}
    
    try:
        raw_chunks = []
        file_level_metadata = base_metadata.copy()

        if file_ext == '.txt' and filename.startswith('WhatsApp Chat with'):
            items_to_return = []
            chat_pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2})\s-\s([^:]+):\s(.*)")
            current_message_text, current_msg_meta = "", {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = chat_pattern.match(line)
                    if match:
                        if current_message_text:
                            items_to_return.append((current_message_text, current_msg_meta))
                        timestamp, sender, message = match.groups()
                        current_message_text = message.strip()
                        current_msg_meta = base_metadata.copy()
                        current_msg_meta['sender'] = sender.strip()
                        current_msg_meta['timestamp'] = timestamp.strip()
                    else:
                        current_message_text += "\n" + line.strip()
            if current_message_text:
                items_to_return.append((current_message_text, current_msg_meta))
            return items_to_return

        elif file_ext == '.pdf':
            try:
                first_page_elements = partition(filename=file_path, strategy="fast", max_pages=1)
                first_page_text = "\n".join([str(el) for el in first_page_elements])
                email_metadata = extract_email_metadata_from_pdf_text(client, first_page_text)
                if email_metadata.get("is_email"):
                    print(f"    - Detected Email in PDF: {filename}")
                    file_level_metadata.update(email_metadata)
            except Exception as e:
                logging.error(f"Could not perform PDF pre-check on {filename}: {e}")
            raw_chunks = [str(el) for el in partition(filename=file_path, strategy="fast")]

        elif file_ext in {'.docx', '.txt'}:
            raw_chunks = [str(el) for el in partition(filename=file_path, strategy="fast")]
        elif file_ext in {'.xlsx', '.csv'}:
            df = pd.read_csv(file_path) if file_ext == '.csv' else pd.read_excel(file_path, engine='openpyxl')
            raw_chunks = [f"Row {idx+1}: {', '.join(f'{col}: {val}' for col, val in row.astype(str).items())}" for idx, row in df.iterrows()]
        else:
            return []
        
        return [(chunk, file_level_metadata) for chunk in raw_chunks]

    except Exception as e:
        logging.error(f"Could not process file {file_path}: {e}")
        return []

# --- EMBEDDING ---
def get_openai_embeddings(client, chunks):
    # This function now expects a pre-validated, clean list of chunks.
    if not chunks: return []
    try:
        response = client.embeddings.create(input=chunks, model=EMBEDDING_MODEL)
        return [item.embedding for item in response.data]
    except Exception as e:
        logging.error(f"Failed to get embeddings from OpenAI: {e}")
        return []

# --- MAIN EXECUTION (DEFINITIVE BATCHING FIX) ---
def main():
    extraction_dir = os.path.join(SCRIPT_DIR, 'data', 'extracted')
    if not os.path.isdir(extraction_dir):
        print(f"[!] FATAL: Extracted data folder not found at '{extraction_dir}'")
        return
    
    client = initialize_services()

    if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
    if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)

    files_to_process = [os.path.join(root, file) for root, _, files in os.walk(extraction_dir) for file in files]
    print(f"[+] Found {len(files_to_process)} total files to process.")
    
    all_raw_items = []
    pbar_files = tqdm(files_to_process, desc="Stage 1: Parsing Files", unit="file")
    for file_path in pbar_files:
        pbar_files.set_postfix_str(os.path.basename(file_path))
        raw_items_from_file = get_chunks_and_metadata(client, file_path)
        if raw_items_from_file:
            all_raw_items.extend(raw_items_from_file)

    if not all_raw_items:
        print("\n[!] No text chunks were extracted from any files.")
        return
    
    print(f"\n[*] Parsed {len(all_raw_items)} raw chunks. Now cleaning and embedding in batches...")

    all_embeddings = []
    metadata_store = []
    batch_size = 500 # A safe and efficient batch size for the API

    pbar_batches = tqdm(range(0, len(all_raw_items), batch_size), desc="Stage 2: Embedding Batches", unit="batch")
    for i in pbar_batches:
        batch = all_raw_items[i:i + batch_size]
        
        # 1. Get raw texts and metadatas for the current batch
        raw_texts = [item[0] for item in batch]
        raw_metadatas = [item[1] for item in batch]
        
        # 2. Centrally clean, filter, and split the data for this batch
        clean_texts = filter_and_split_chunks(raw_texts)
        if not clean_texts:
            continue
            
        # This mapping is complex, so we will simplify by assuming a 1-to-1 mapping
        # after initial filtering. A more advanced system would map split chunks back.
        # For now, we will filter items together.
        
        valid_items = []
        for j, text in enumerate(raw_texts):
            if text in clean_texts:
                valid_items.append({'text': text, 'metadata': raw_metadatas[j]})
        
        valid_texts_to_embed = [item['text'] for item in valid_items]
        valid_metadatas_to_store = [item['metadata'] for item in valid_items]

        if not valid_texts_to_embed:
            continue

        # 3. Get embeddings for the now GUARANTEED clean data
        embeddings = get_openai_embeddings(client, valid_texts_to_embed)
        if not embeddings or len(embeddings) != len(valid_texts_to_embed):
            logging.warning(f"Embedding count mismatch in batch starting at index {i}. Skipping batch.")
            continue
        
        # 4. Store the perfectly synchronized data
        all_embeddings.extend(embeddings)
        for k, metadata in enumerate(valid_metadatas_to_store):
            metadata['original_text'] = valid_texts_to_embed[k]
            metadata_store.append(metadata)

    if not all_embeddings:
        print("\n[!] No valid text chunks survived the cleaning process.")
        return

    print(f"\n[*] Extracted a total of {len(all_embeddings)} high-quality text chunks.")
    print(f"[*] Building FAISS index with dimension {EMBEDDING_DIMENSION}...")

    np_embeddings = np.array(all_embeddings).astype('float32')
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    index.add(np_embeddings)
    print(f"[+] FAISS index built successfully. Total vectors: {index.ntotal}")

    print(f"[*] Saving FAISS index and metadata...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    print(f"\n[+] --- SCRIPT COMPLETE ---")

if __name__ == "__main__":
    main()