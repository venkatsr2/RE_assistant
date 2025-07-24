# Local AI Business Data Assistant

This project is a powerful, locally-run AI assistant designed to understand and answer natural language questions about a large corpus of private business documents. It leverages a sophisticated Retrieval-Augmented Generation (RAG) architecture, combining fast local search with the advanced reasoning of large language models.

The system is built to handle a messy, unorganized collection of real-world business data, including PDFs, emails saved as PDFs, WhatsApp chats, spreadsheets, and Word documents.

---

## ✨ Core Features & Architecture

This application is built on a modern, robust RAG pipeline:

*   **Intelligent Data Processing (`process_data.py`):**
    *   **Smart Parsing:** Intelligently identifies different file types and uses specialized parsers for each.
    *   **Email-in-PDF Detection:** Uses a fast LLM (`gpt-4o-mini`) to analyze PDFs and automatically extract email metadata (`from`, `to`, `subject`, `summary`) if a document is identified as an email.
    *   **WhatsApp Parsing:** A custom parser extracts message-level metadata (`sender`, `timestamp`) from `.txt` chat logs.
    *   **Data Quality Control:** A rigorous cleaning and chunking pipeline filters out junk text (e.g., system messages), removes uselessly short snippets, and splits oversized documents into perfectly sized chunks for optimal embedding.

*   **High-Performance Local Database:**
    *   **Embedding Model:** Uses OpenAI's high-performance `text-embedding-3-small` model to generate vector embeddings.
    *   **Vector Search:** Employs **FAISS** (from Meta AI) for lightning-fast, in-memory similarity search. The index is stored locally in `faiss_index.bin`.
    *   **Metadata Storage:** All extracted metadata is stored locally in a synchronized `metadata_store.pkl` file.

*   **Intelligent Querying (`query_local.py`):**
    *   **Broad Retrieval, Smart Synthesis:** Takes a user's full, complex natural language question and performs a broad semantic search to retrieve a rich, diverse context of candidate documents.
    *   **Advanced Reasoning:** Passes this large context to a state-of-the-art reasoning engine (**GPT-4o**) which is empowered to perform the final, nuanced filtering, analysis, and synthesis to generate a precise, cited answer.

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
    cd local-real-estate-assistant
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
    *   This script assumes your data has **already been extracted**.
    *   Place all your unzipped files and folders into the `data/extracted/` directory.

### 3. How to Use

The application works in two stages: a one-time data processing step, followed by the interactive query step.

#### Stage 1: Process Your Data (Run Once)

This script reads all files from `data/extracted/`, intelligently parses them, creates high-quality embeddings, and builds your local search database (`faiss_index.bin` and `metadata_store.pkl`).

Run this command from your terminal. It will take time, especially for large datasets.
```bash
python process_data.py