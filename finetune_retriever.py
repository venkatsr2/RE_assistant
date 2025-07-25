import os
from dotenv import load_dotenv
from tqdm import tqdm
import json
import random
import logging
from torch.utils.data import DataLoader

# --- Import directly from sentence-transformers ---
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(SCRIPT_DIR, "real_estate_training_dataset.jsonl")
BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
OUTPUT_MODEL_PATH = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")

# --- DATASET PREPARATION ---
def create_dataset_from_jsonl(file_path):
    """Loads a JSONL file and creates training and evaluation datasets."""
    print(f"[*] Loading and preparing dataset from '{file_path}'...")
    queries = {}
    corpus = {}
    dev_samples = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            query_id = f"q_{i}"
            passage_id = f"p_{i}"
            
            queries[query_id] = data["question"]
            corpus[passage_id] = data["passage"]
            
            # For evaluation, we need to know which passage matches which query
            if len(dev_samples) < 200: # Create a small dev set
                dev_samples[query_id] = {passage_id}

    print(f"[+] Dataset prepared. Corpus size: {len(corpus)}, Query size: {len(queries)}")
    return queries, corpus, dev_samples

def create_training_examples(queries, corpus):
    """Creates InputExample objects for training."""
    train_examples = []
    for qid, query in queries.items():
        # The corresponding passage has the same numeric id
        pid = qid.replace('q_', 'p_')
        if pid in corpus:
            passage = corpus[pid]
            # For MultipleNegativesRankingLoss, we just need the positive pair
            train_examples.append(InputExample(texts=[query, passage]))
    return train_examples

# --- MAIN SCRIPT ---
def main():
    """
    Fine-tunes a SentenceTransformer model using the library's native training loop.
    """
    if not os.path.exists(DATASET_FILE):
        print(f"[!] FATAL: Training dataset '{DATASET_FILE}' not found. Please run 'generate_training_data.py' first.")
        return

    # 1. Load and prepare the data
    queries, corpus, dev_samples = create_dataset_from_jsonl(DATASET_FILE)
    train_examples = create_training_examples(queries, corpus)

    # 2. Load the base model
    print(f"[*] Loading base model: {BASE_MODEL_NAME}")
    model = SentenceTransformer(BASE_MODEL_NAME)

    # 3. Create a DataLoader and a Loss function
    # The DataLoader will handle batching the training examples
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # MultipleNegativesRankingLoss is a powerful and standard loss function for this task
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # (Optional but recommended) Create an evaluator to see how well the model is learning
    evaluator = InformationRetrievalEvaluator(queries, corpus, dev_samples, name="eval")

    # 4. Fine-tune the model
    print("[*] Starting model fine-tuning... This will take some time and use your CPU.")
    
    # The `fit` method is the core training loop
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=1,
        warmup_steps=100,
        output_path=OUTPUT_MODEL_PATH,
        show_progress_bar=True
    )
    
    print(f"\n[+] Fine-tuning complete!")
    print(f"[*] The fine-tuned model has been saved to '{OUTPUT_MODEL_PATH}'.")

if __name__ == "__main__":
    # The generate_training_data.py script does not need to change.
    # You can run it first, then run this script.
    main()