import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import logging
import re

# --- CONFIGURATION ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- !! CRITICAL !! ---
# Use the new database files created by the fine-tuned model
COLLECTION_NAME = "real_estate_finetuned_local"
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_faiss.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_metadata.pkl")
# Path to your custom-trained, domain-expert model
FINETUNED_MODEL_PATH = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")

# --- KNOWLEDGE BASE TOOL ---
class KnowledgeBaseTool:
    """A tool that encapsulates our RAG pipeline using the fine-tuned local model."""
    def __init__(self):
        print("[*] Initializing Knowledge Base Tool...")
        if not os.path.isdir(FINETUNED_MODEL_PATH):
            raise FileNotFoundError(f"FATAL: Fine-tuned model not found at '{FINETUNED_MODEL_PATH}'.")
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"FATAL: Database files not found. Please run 'process_data.py' with the fine-tuned model first.")
            
        print(f"[*] Loading fine-tuned model for queries...")
        # --- Load the same fine-tuned model used for processing ---
        self.embedding_model = SentenceTransformer(FINETUNED_MODEL_PATH, device='cpu')
        
        print(f"[*] Loading database from disk...")
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self.metadata_store = pickle.load(f)
        print("[+] Knowledge Base Tool is ready.")

    def search(self, query: str, top_k: int = 15) -> str:
        """Performs a semantic search using the fine-tuned local model."""
        print(f"    - TOOL: Searching knowledge base for: '{query}'")
        
        # --- Use the loaded local model to create a matching query vector ---
        query_embedding = self.embedding_model.encode([query])
        # FAISS expects a numpy array of float32
        query_embedding_np = np.array(query_embedding).astype('float32')

        distances, indices = self.faiss_index.search(query_embedding_np, top_k)
        
        retrieved_metadatas = [self.metadata_store[i] for i in indices[0] if i != -1]
        if not retrieved_metadatas:
            return "No relevant information was found in the knowledge base for that query."
            
        context = "\n---\n".join([f"Source: {doc.get('source', 'N/A')}\nContent: {doc.get('original_text', '')}" for doc in retrieved_metadatas])
        return context

# --- AGENT IMPLEMENTATION ---
class ReActAgent:
    """A simple ReAct Agent that can reason and use tools."""
    def __init__(self, client: OpenAI, tool: KnowledgeBaseTool):
        self.client = client
        self.tool = tool
        self.system_prompt = """
        You are an autonomous AI assistant for a real estate business. Your goal is to answer the user's question by breaking it down into a series of steps.
        You have access to ONE tool: `knowledge_base_search(query: str)`. Use this tool to find information within the company's documents.

        For each step, you must first think about your plan and then decide on an action. Follow this format exactly:

        Thought: [Your reasoning and plan for the next step.]
        Action: [The tool call you will make, e.g., `knowledge_base_search(query='customer complaints about water leakage')`. If you have enough information, this should be `Final Answer(answer='Your final, synthesized answer.')`.]
        """
        self.history = [("system", self.system_prompt)]

    def run(self, user_query: str):
        self.history = [("system", self.system_prompt)] # Reset history for each new query
        self.history.append(("user", user_query))
        
        for i in range(5): # Limit to 5 steps
            print("\n" + "="*50 + f" STEP {i+1} " + "="*50)
            
            # Create the prompt from the history
            prompt_messages = [{"role": role, "content": content} for role, content in self.history]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=prompt_messages,
                temperature=0.0
            )
            action_text = response.choices[0].message.content
            
            print(action_text) # Show the agent's "inner monologue"
            self.history.append(("assistant", action_text))

            if "Final Answer(" in action_text:
                try:
                    # --- A flexible regex that accepts single or double quotes ---
                    match = re.search(r"Final Answer\(answer=(['\"])(.*)\1\)", action_text, re.DOTALL)
                    if match:
                        final_answer = match.group(2) # Group 2 now captures the content inside the quotes
                        print("\n" + "="*50 + " FINAL ANSWER " + "="*50)
                        print(final_answer)
                    else:
                        print("\n[!] Could not parse the final answer from the agent's response.")
                except Exception as e:
                    print(f"\n[!] Error parsing final answer: {e}")
                return
            
            elif "knowledge_base_search(" in action_text:
                try:
                    # Make this regex flexible too
                    match = re.search(r"knowledge_base_search\(query=(['\"])(.*)\1\)", action_text, re.DOTALL)
                    if match:
                        query = match.group(2)
                        observation = self.tool.search(query=query)
                        self.history.append(("user", f"Observation: {observation}"))
                    else:
                        self.history.append(("user", "Observation: Could not parse the tool query."))
                except Exception as e:
                    print(f"\n[!] Could not parse or execute tool action: {e}")
                    self.history.append(("user", "Observation: Error executing the tool."))
        
        print("\n[!] Agent stopped after reaching the maximum number of steps.")

# --- MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    load_dotenv()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not openai_client.api_key:
        print("[!] FATAL: OPENAI_API_KEY not found in .env file.")
    else:
        try:
            kb_tool = KnowledgeBaseTool()
            agent = ReActAgent(client=openai_client, tool=kb_tool)
            
            print("\n--- AI Business Analyst is Ready ---")
            print("Ask a complex question (or type 'exit' to quit).")
            
            while True:
                user_input = input("> ")
                if user_input.lower() == 'exit':
                    break
                agent.run(user_query=user_input)
        except Exception as e:
            print(f"\n[!] A critical error occurred during initialization: {e}")