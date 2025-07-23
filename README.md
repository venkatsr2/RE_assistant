# Local AI Business Data Assistant
A powerful, fully local AI assistant that answers natural language questions about your private business documents.
✨ Key Features
- Intelligent Q&A: Ask complex questions in natural language, not just keywords.
- All Data Types: Processes PDFs, DOCX, emails, WhatsApp chats, spreadsheets, and more from a single .zip file.
- Fast & Free: Uses a local FAISS index and SentenceTransformer embeddings for high-speed, zero-cost searching.
- Advanced Reasoning: Leverages Google's Gemini 1.5 Pro to analyze retrieved information and provide accurate, synthesized answers.
- Fully Private: Your documents and search index never leave your local machine. Only the final, small context is sent to the AI for reasoning.
🚀 Quick Start
1. Setup
Install Libraries:
pip install -r requirements.txt
Create a .env file.
Get a key from Google AI Studio.
Add it to the file: GOOGLE_API_KEY="AIza..."
Add Data:
Place your data file in the data/ folder.
Name it data.zip.

3. Process Data (Run Once)
This creates your local search database. Run this once, and again only if your data.zip changes.
python process_data.py

4. Ask Questions
Use this script to ask anything about your data.
python query_local.py "Your question in quotes"

💡 Example Queries
Generated bash
### General question
python query_local.py "Summarize the main points of the lease agreement"

### Analytical question
python query_local.py "What is the sentiment of the <sender_name> whatsapp chat?"

### Complex, specific question
python query_local.py "Find the latest email from <sender_name> and summarize it"
