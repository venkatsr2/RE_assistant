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
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition
from unstructured.partition.email import partition_email
import tiktoken

# --- CONFIGURATION ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- !! CRITICAL !! ---
# Path to your custom-trained, domain-expert model
FINETUNED_MODEL_PATH = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")
# The dimension of the bge-base-en-v1.5 model we fine-tuned
EMBEDDING_DIMENSION = 768
COLLECTION_NAME = "real_estate_finetuned_local" # A new name for our expert database

FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_faiss.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_metadata.pkl")
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

# --- INITIALIZATION ---
def initialize_services():
    """Load the fine-tuned local embedding model."""
    if not os.path.isdir(FINETUNED_MODEL_PATH):
        raise FileNotFoundError(
            f"\n[!] FATAL: Fine-tuned model not found at '{FINETUNED_MODEL_PATH}'."
            f"\n[!] Please run 'finetune_retriever.py' successfully before running this script."
        )
    
    print(f"[*] Loading your fine-tuned embedding model from '{FINETUNED_MODEL_PATH}'...")
    embedding_model = SentenceTransformer(FINETUNED_MODEL_PATH, device='cpu')
    print("[+] Fine-tuned model loaded successfully.")
    return embedding_model

# --- DATA QUALITY CONTROL ---
def filter_and_split_chunks(raw_chunks):
    """Filters junk and splits oversized chunks."""
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

# --- SPECIALIZED FILE PARSERS ---
def get_chunks_and_metadata(file_path):
    """Processes a file, extracts text chunks and relevant metadata."""
    file_ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    extraction_dir = os.path.join(SCRIPT_DIR, 'data', 'extracted')
    base_metadata = {'source': os.path.relpath(file_path, extraction_dir)}
    
    try:
        items_to_return = []
        file_level_metadata = base_metadata.copy()

        if file_ext == '.txt' and filename.lower().startswith('whatsapp chat with'):
            chat_pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s[ap]m)?)\s-\s([^:]+):\s(.*)", re.IGNORECASE)
            current_message_text = ""
            current_msg_meta = {}
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
            
            final_items = []
            for text, metadata in items_to_return:
                clean_chunks = filter_and_split_chunks([text])
                for clean_chunk in clean_chunks:
                    final_items.append((clean_chunk, metadata))
            return final_items

        elif file_ext == '.eml':
            elements = partition_email(filename=file_path)
            if elements: file_level_metadata.update(elements[0].metadata.to_dict())
            raw_chunks = [str(el) for el in elements]
        elif file_ext in {'.pdf', '.docx', '.txt'}:
            raw_chunks = [str(el) for el in partition(filename=file_path, strategy="fast")]
        elif file_ext in {'.xlsx', '.csv'}:
            df = pd.read_csv(file_path) if file_ext == '.csv' else pd.read_excel(file_path, engine='openpyxl')
            raw_chunks = [f"Row {idx+1}: {', '.join(f'{col}: {val}' for col, val in row.astype(str).items())}" for idx, row in df.iterrows()]
        else:
            return []
        
        final_chunks = filter_and_split_chunks(raw_chunks)
        return [(chunk, file_level_metadata) for chunk in final_chunks]

    except Exception as e:
        logging.error(f"Could not process file {file_path}: {e}")
        return []
    
# --- UTILITIES ---
def sanitize_filename(filename):
    filename = filename.strip()
    return re.sub(r'[<>:"/\\|?*]', '_', filename).rstrip('. ')

def get_local_embeddings(model, chunks):
    try:
        return model.encode(chunks, show_progress_bar=False).tolist()
    except Exception as e:
        logging.error(f"Failed to get local embeddings: {e}")
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
    
    all_embeddings = []
    metadata_store = []

    pbar = tqdm(files_to_process, desc="Processing Files", unit="file")
    for file_path in pbar:
        pbar.set_postfix_str(os.path.basename(file_path))
        
        items_from_file = get_chunks_and_metadata(file_path)
        if not items_from_file: continue
        
        texts_to_embed = [item[0] for item in items_from_file]
        metadatas_to_store = [item[1] for item in items_from_file]
        
        embeddings = get_local_embeddings(client, texts_to_embed)
        if not embeddings: continue

        for i, embedding in enumerate(embeddings):
            all_embeddings.append(embedding)
            metadata = metadatas_to_store[i]
            metadata['original_text'] = texts_to_embed[i]
            metadata_store.append(metadata)

    if not all_embeddings:
        print("\n[!] No text chunks were extracted from any files.")
        return

    print(f"\n[*] Extracted a total of {len(all_embeddings)} high-quality text chunks.")
    print(f"[*] Building FAISS index with dimension {EMBEDDING_DIMENSION}...")

    np_embeddings = np.array(all_embeddings).astype('float32')
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    index.add(np_embeddings)
    print(f"[+] FAISS index built successfully. Total vectors: {index.ntotal}")

    print(f"[*] Saving FAISS index to '{FAISS_INDEX_PATH}'...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    print(f"[*] Saving metadata to '{METADATA_PATH}'...")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    print(f"\n[+] --- SCRIPT COMPLETE ---")
    print(f"[*] Your new, expert database files ('{os.path.basename(FAISS_INDEX_PATH)}' and '{os.path.basename(METADATA_PATH)}') are ready.")

if __name__ == "__main__":
    main()