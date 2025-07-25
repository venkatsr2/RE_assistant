import os
import pickle
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import json
import random
from datasets import Dataset

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata_store.pkl")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "real_estate_training_dataset.jsonl")
# We will only generate questions for a subset of chunks to manage cost and time
NUM_SAMPLES_TO_GENERATE = 2000

# --- INITIALIZATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- MAIN SCRIPT ---
def generate_synthetic_dataset():
    """
    Loads clean text chunks and uses an LLM to generate a question for each,
    creating a (question, passage) dataset for fine-tuning.
    """
    print(f"[*] Loading metadata from '{METADATA_PATH}'...")
    try:
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
    except FileNotFoundError:
        print(f"[!] FATAL: Metadata file not found. Please run 'process_data.py' first.")
        return

    if not metadata_store:
        print("[!] Metadata store is empty. No data to process.")
        return

    print(f"[+] Found {len(metadata_store)} total chunks.")

    # Randomly sample a subset of the data to process
    if len(metadata_store) > NUM_SAMPLES_TO_GENERATE:
        print(f"[*] Sampling {NUM_SAMPLES_TO_GENERATE} chunks to generate questions for...")
        sampled_metadata = random.sample(metadata_store, NUM_SAMPLES_TO_GENERATE)
    else:
        sampled_metadata = metadata_store

    system_prompt = """
    You are an expert data analyst. Your task is to generate a single, relevant question that could be answered by the provided text passage.
    The question should be something a user in a real estate company would realistically ask.
    Focus on the specific details in the text. Your entire response must be ONLY the question text.
    """
    
    training_data = []
    pbar = tqdm(sampled_metadata, desc="Generating Questions", unit="chunk")
    for metadata in pbar:
        passage = metadata.get('original_text')
        if not passage:
            continue

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"PASSAGE:\n---\n{passage}\n---\nQUESTION:"}
                ],
                temperature=0.7
            )
            question = response.choices[0].message.content.strip()
            
            if question:
                training_data.append({"question": question, "passage": passage})
        except Exception as e:
            print(f"\n[!] Error generating question for a chunk: {e}")
            continue
            
    if not training_data:
        print("\n[!] No training data was generated.")
        return

    # Save the dataset to a JSONL file
    print(f"\n[*] Saving {len(training_data)} generated pairs to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"[+] Dataset generation complete!")

if __name__ == "__main__":
    generate_synthetic_dataset()