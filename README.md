# AI_Knowledge_graph_builder_for_enterprise
Enterprise Semantic RAG Chatbot is an intelligent question-answering system designed to retrieve and generate accurate answers from enterprise data sources. The chatbot supports structured (CSV), semi-structured (JSON), and unstructured (PDF) data formats.
The system uses Sentence Transformers to generate semantic embeddings and FAISS for efficient vector-based search. For structured queries, it directly processes data using logical filters, while for unstructured queries it follows a Retrieval-Augmented Generation (RAG) approach by combining relevant context with a Groq LLaMA-3.1 Large Language Model to generate meaningful responses.

🔹 Programming & Frameworks Python – Core backend development Streamlit – Interactive web UI for the chatbot

🔹 Data Sources CSV files – Assets, employees, attendance, vendors, policies JSON files – Project data, emails, tickets, performance records PDF documents – Annual reports, company policies, engagement reports

🔹 Natural Language Processing (NLP) Sentence Transformers (all-MiniLM-L6-v2) Converts text into semantic embeddings

🔹 Vector Database & Search FAISS (Facebook AI Similarity Search) Stores embeddings Enables fast semantic similarity search

🔹 Retrieval-Augmented Generation (RAG) Hybrid RAG Architecture Structured query handling (Pandas logic) Unstructured query handling (FAISS + LLM)

🔹 Large Language Model (LLM) Groq LLM Model: LLaMA-3.1-8B-Instant Used for: Answer generation Summarization: Context-aware responses from PDFs & text

🔹 Libraries & Tools Pandas – Structured data querying NumPy – Numerical operations PyPDF – PDF text extraction LangChain Text Splitter – Chunking large documents Dotenv – Secure API key management FAISS-CPU – Vector indexing Groq SDK – LLM integration

🔹 UI & Styling Custom CSS Glassmorphism design Dark gradient theme Responsive layout

Questions may ask Which assets are under maintenance? Which laptops are available? Explain leave policy of TCS? What is Infosys company overview?

To run: streamlit run streamlit_app.py

The chatbot is deployed using Streamlit with a modern interactive interface and intelligently routes user queries to the appropriate data source, making it suitable for enterprise use cases such as asset tracking, policy lookup, project management, and report summarization.

Technology Stack
Backend
Python
Pandas
NumPy
LangChain Text Splitter
PyPDF
FAISS-CPU

Frontend
Streamlit Web Interface
Custom CSS (Glassmorphism UI)
Dark Gradient Theme
Responsive Chat Layout
