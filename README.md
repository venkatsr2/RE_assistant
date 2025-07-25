# Local AI Business Analyst Agent

This project implements a sophisticated, locally-run AI assistant designed to understand and answer natural language questions about a large corpus of private business documents. It leverages an advanced **Agentic RAG (Retrieval-Augmented Generation)** architecture, capable of understanding complex, multi-step natural language queries and providing nuanced, data-driven answers.

The system is built to handle a messy, unorganized collection of real-world business data, including PDFs, emails, WhatsApp chats, spreadsheets, and Word documents.

---

## ✨ Core Features & Architecture

This application is built on a state-of-the-art pipeline that incorporates a **Model Context Protocol (MCP)** by fine-tuning its own domain-expert retrieval model.

### 1. The Data Processing & Fine-Tuning Pipeline

This is a multi-stage, one-time process to create an expert knowledge base.

`data.zip` -> **Hybrid Parsing** -> `Clean Text Chunks` -> **Synthetic Data Generation (GPT-4o-mini)** -> `(Question, Passage) Pairs` -> **Fine-Tune Embedding Model (SetFit)** -> **Expert Local Model (BGE)** -> **FAISS Index + Metadata Store**

*   **Intelligent Hybrid Parsing (`process_data.py`):**
    *   Uses specialized parsers for **Emails** (`.eml`) and **WhatsApp chats** (`.txt`) to extract rich, structured metadata (e.g., `sender`, `subject`, `timestamp`).
    *   Employs a fast, non-OCR parser for general documents (`.pdf`, `.docx`).
*   **Data Quality Control:**
    *   A rigorous cleaning pipeline filters out "junk" text (like system messages), removes uselessly short snippets, and splits oversized documents into perfectly sized chunks for optimal embedding.
*   **Model Context Protocol (MCP):**
    *   **Synthetic Data Generation (`generate_training_data.py`):** Uses `gpt-4o-mini` to automatically create a high-quality `(question, passage)` training dataset from your own documents.
    *   **Retriever Fine-Tuning (`finetune_retriever.py`):** Uses the `setfit` library to fine-tune an open-source `BAAI/bge-base-en-v1.5` model on your synthetic dataset, transforming it into a domain-expert.
*   **Local Vector Database:**
    *   The final, expert model is used to create embeddings, which are stored in a lightning-fast **FAISS** index (`faiss_index.bin`) and a synchronized **Pickle file** for metadata (`metadata_store.pkl`).

### 2. The Agentic Query Pipeline (`agent.py`)

This is the interactive engine that uses the expert knowledge base.

`User Query` -> **ReAct Agent (GPT-4o)** -> `Thought: Create Plan` -> `Action: Use Tool` -> **Knowledge Base Search** -> `Observation: Retrieved Context` -> `Thought: Synthesize` -> **Final Answer**

*   **AI Agent (`agent.py`):**
    *   Employs a **ReAct (Reason + Act)** agent powered by a top-tier LLM like **GPT-4o**.
    *   The agent maintains an "inner monologue," breaking down complex user goals into a series of logical steps.
*   **Knowledge Base Tool:**
    *   The agent's primary tool is a `knowledge_base_search` function, which queries the custom-trained FAISS index. Because the index was built with a domain-expert model, this retrieval step is highly accurate and relevant.
*   **Autonomous Reasoning:**
    *   The agent can perform multiple searches to gather information from different sources within your data and synthesize the findings to answer complex, multi-part analytical questions.

---

## 🚀 Getting Started

Follow these steps to set up and run the application.

### 1. Prerequisites
*   Python 3.8+
*   Git

### 2. Setup and Installation

1.  **Clone the repository (or navigate to your existing project folder):**
    ```bash
    git clone [your-github-repo-url]
    cd [your-project-folder]
    ```

2.  **Install all dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create and Configure the Environment File:**
    *   Create a file named `.env` in the root of the project.
    *   Get an API key from [OpenAI](https://platform.openai.com/).
    *   Add your key to the `.env` file:
        ```env
        OPENAI_API_KEY="sk-..."
        ```

4.  **Add Your Data:**
    *   Place your compressed data file in the `data/` folder, named exactly `data.zip`.

---

## 🛠️ How to Use: The Full Workflow

This is a multi-step process. You must run these scripts in order.

### Step 1: Process Initial Data (Run Once)
First, we need to create a baseline `metadata_store.pkl` file from which to generate our training data. This uses the generic, non-fine-tuned model.

This uses the generic model to create the initial metadata store
python process_data.py 

### Step 2: Generate Fine-Tuning Data (Run Once)
This script reads the metadata store and uses gpt-4o-mini to create the training file. This step has an API cost and will take time.
python generate_training_data.py


### Step 3: Fine-Tune Your Expert Model (Run Once)
This script takes the training data and fine-tunes the local embedding model. This will be CPU-intensive and will take a significant amount of time.
python finetune_retriever.py

Upon completion, this will create a finetuned_bge_real_estate_model folder in your project directory.

### Step 4: Re-Process Data with Your Expert Model (Run Once)
Now that you have an expert model, you must re-build your database with it.
IMPORTANT: Open process_data.py and ensure the FINETUNED_MODEL_PATH variable is correctly pointing to the folder created in the last step.
Run the script again. It will delete the old database files and create new, superior ones.
python process_data.py

### Step 5: Run the Agent and Ask Questions
Your system is now ready. Run the agent script to start the interactive session.
python agent.py

You can now ask complex, multi-step questions.
## Example Interaction:

--- AI Business Analyst is Ready ---
Ask a complex question (or type 'exit' to quit).
> Compare the customer complaints about delivery delays in the WhatsApp chats to the complaints in the main customer emails from last quarter.

================================================== STEP 1 ==================================================
Thought: The user wants to compare complaints from two different sources (WhatsApp and emails) in a specific time frame. I need to search for each source separately and then combine the results. First, I'll search for the WhatsApp complaints.
Action: knowledge_base_search(query='customer complaints about delivery delays in WhatsApp chats from last quarter')
    - TOOL: Searching knowledge base for: 'customer complaints about delivery delays in WhatsApp chats from last quarter'
================================================== STEP 2 ==================================================
Thought: I have retrieved the complaints from the WhatsApp chats. Now I need to find the complaints from the customer communication emails for the same period.
Action: knowledge_base_search(query='customer complaints about delivery delays in customer communication emails from last quarter')
    - TOOL: Searching knowledge base for: 'customer complaints about delivery delays in customer communication emails from last quarter'
================================================== STEP 3 ==================================================
Thought: I now have the information from both sources. The WhatsApp chats mention issues with material delivery and on-site delays. The emails also mention delivery delays but focus more on billing and refund disputes related to those delays. I have enough information to synthesize a final answer.
Action: Final Answer(answer="During the last quarter, complaints in the WhatsApp chats focused on operational issues like material delivery and on-site work delays [source: ...]. In contrast, the customer emails from the same period also mentioned these delays but had a higher number of complaints related to the financial aspects, such as refund statements and billing discrepancies that arose from the delays [source: ...].")