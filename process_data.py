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
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata_store.pkl")
CHUNK_SIZE_TOKENS = 512      # The target size for each text chunk in tokens
CHUNK_OVERLAP_TOKENS = 50    # The number of tokens to overlap between chunks

# --- INITIALIZATION ---
def initialize_services():
    """Load the local embedding model."""
    print("[*] Loading local embedding model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    print("[+] Embedding model loaded.")
    return embedding_model

# --- DATA QUALITY CONTROL ---
def filter_and_split_chunks(raw_chunks):
    """Filters out junk and splits oversized chunks to ensure data quality."""
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    JUNK_PHRASES = [
        "messages and calls are end-to-end encrypted",
        "this message was deleted",
        "<media omitted>",
        "created group",
        "added you",
        "you're now an admin"
    ]
    
    final_chunks = []
    
    for chunk in raw_chunks:
        if len(chunk) < 25 or any(phrase in chunk.lower() for phrase in JUNK_PHRASES):
            continue
            
        tokens = tokenizer.encode(chunk)
        if len(tokens) > CHUNK_SIZE_TOKENS:
            for i in range(0, len(tokens), CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS):
                sub_chunk_tokens = tokens[i:i + CHUNK_SIZE_TOKENS]
                final_chunks.append(tokenizer.decode(sub_chunk_tokens))
        else:
            final_chunks.append(chunk)
            
    return final_chunks

# --- SPECIALIZED FILE PARSERS ---
def get_chunks_and_metadata(file_path):
    """Processes a single file, extracts text chunks and relevant metadata."""
    file_ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    extraction_dir = os.path.join(SCRIPT_DIR, 'data', 'extracted')
    base_metadata = {'source': os.path.relpath(file_path, extraction_dir)}
    
    try:
        items_to_return = []
        file_level_metadata = base_metadata.copy()

        if file_ext == '.txt' and filename.startswith('WhatsApp Chat with'):
            chat_pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[ap]m)\s-\s([^:]+):\s(.*)", re.IGNORECASE)
            current_message_text = ""
            current_message_metadata = base_metadata.copy()
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = chat_pattern.match(line)
                    if match:
                        if current_message_text:
                            filtered = filter_and_split_chunks([current_message_text])
                            for f_chunk in filtered: items_to_return.append((f_chunk, current_message_metadata))
                        timestamp, sender, message = match.groups()
                        current_message_text = message.strip()
                        current_message_metadata = base_metadata.copy()
                        current_message_metadata['sender'] = sender.strip()
                        current_message_metadata['timestamp'] = timestamp.strip()
                    else:
                        current_message_text += "\n" + line.strip()
            if current_message_text:
                filtered = filter_and_split_chunks([current_message_text])
                for f_chunk in filtered: items_to_return.append((f_chunk, current_message_metadata))
            return items_to_return

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

# --- UTILITY FUNCTIONS ---
def sanitize_filename(filename):
    filename = filename.strip()
    return re.sub(r'[<>:"/\\|?*]', '_', filename).rstrip('. ')

def robust_recursive_unzip(zip_path, extract_to):
    print(f"[*] Extracting '{os.path.basename(zip_path)}'...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                path_parts = member.filename.replace('\\', '/').split('/')
                sanitized_parts = [sanitize_filename(part) for part in path_parts]
                safe_path = os.path.join(extract_to, *sanitized_parts)
                if member.is_dir(): os.makedirs(safe_path, exist_ok=True)
                else:
                    parent_dir = os.path.dirname(safe_path)
                    os.makedirs(parent_dir, exist_ok=True)
                    with open(safe_path, 'wb') as f: f.write(zip_ref.read(member.filename))
    except Exception as e: logging.error(f"Could not process zip file {zip_path}. Reason: {e}. Skipping.")
    print(f"[*] Scanning for and extracting nested zip files...")
    for root, _, files in os.walk(extract_to):
        for filename in files:
            if filename.lower().endswith('.zip'):
                nested_zip_path = os.path.join(root, filename)
                nested_extract_folder = os.path.join(root, sanitize_filename(os.path.splitext(filename)[0]))
                os.makedirs(nested_extract_folder, exist_ok=True)
                robust_recursive_unzip(nested_zip_path, nested_extract_folder)
                try: os.remove(nested_zip_path)
                except OSError as e: logging.error(f"Could not remove nested archive {nested_zip_path}: {e}")

def get_local_embeddings(model, chunks):
    try:
        return model.encode(chunks, show_progress_bar=False).tolist()
    except Exception as e:
        logging.error(f"Failed to get local embeddings: {e}")
        return []

# --- MAIN EXECUTION ---
def main():
    source_zip_path = os.path.join(SCRIPT_DIR, 'data', 'data.zip')
    extraction_dir = os.path.join(SCRIPT_DIR, 'data', 'extracted')
    if not os.path.exists(source_zip_path):
        print(f"[!] FATAL: Source file not found at '{source_zip_path}'")
        return
    
    embedding_model = initialize_services()

    if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
    if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)

    if not os.path.exists(extraction_dir):
        robust_recursive_unzip(source_zip_path, extraction_dir)
    else:
        print(f"[*] Found existing extraction directory. Skipping extraction.")
    
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
        
        embeddings = get_local_embeddings(embedding_model, texts_to_embed)
        if not embeddings: continue

        for i, embedding in enumerate(embeddings):
            all_embeddings.append(embedding)
            metadata = metadatas_to_store[i]
            metadata['original_text'] = texts_to_embed[i]
            metadata_store.append(metadata)

    if not all_embeddings:
        print("\n[!] No text chunks were extracted from any files. The process has finished with no data.")
        return

    print(f"\n[*] Extracted a total of {len(all_embeddings)} high-quality text chunks.")
    print("[*] Building FAISS index...")

    np_embeddings = np.array(all_embeddings).astype('float32')
    d = embedding_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(d)
    index.add(np_embeddings)
    print(f"[+] FAISS index built successfully. Total vectors: {index.ntotal}")

    print(f"[*] Saving FAISS index to '{FAISS_INDEX_PATH}'...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    print(f"[*] Saving metadata to '{METADATA_PATH}'...")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    print(f"\n[+] --- SCRIPT COMPLETE ---")
    print(f"[*] Your local database files ('{os.path.basename(FAISS_INDEX_PATH)}' and '{os.path.basename(METADATA_PATH)}') are ready.")

if __name__ == "__main__":
    main()