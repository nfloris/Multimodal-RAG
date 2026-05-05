# Multimodal-RAG# Multimodal RAG with Unstructured and AstraDB

This repository contains code for building a Multimodal Retrieval-Augmented Generation (RAG) system. It leverages unstructured for document parsing, and AstraDB for vector storage to process and retrieve information from text and images within PDFs.
## Features
- **PDF Partitioning**: Uses unstructured to extract and structure text, tables, and images from PDF documents.
- **Multimodal Data Handling**: Designed to extract both textual content and image payloads (base64 encoded) from PDFs.
- **Intelligent Chunking**: Employs unstructured's advanced chunking strategies (by_title, max_characters, etc.) to create semantically coherent document chunks.
- **Vector Database Integration**: Utilizes AstraDB to store and retrieve vector embeddings of text chunks and image summaries, enabling efficient similarity search.

## Strategy

The RAG strategy implemented follows these steps:
1. **Image Summarization**: A multimodal LLM is used to generate text summaries from extracted images.
2. **Multimodal Embedding & Retrieval**: Image summaries are embedded along with text chunks (potentially in separate collections) and retrieved using similarity search.
3. **Answer Generation**: The retrieved raw images and relevant text chunks are passed to a multimodal LLM for comprehensive answer generation.


## Setup

1.  **Clone the Repository:**
    
    git clone [<https://github.com/theserenecoder/MultiModel_RAG>](https://github.com/theserenecoder/MultiModel_RAG)
    

2.  **Create a Virtual Environment:**
    ```bash
    python -3.11.15 -m venv venv
    source venv/Scripts/activate  # On Windows
    # source venv/bin/activate    # On macOS/Linux
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    # (Ensure requirements.txt contains: unstructured, langchain-unstructured[local], langchain-astradb, pypdf, python-dotenv, etc.)
    ```
    If you don't have a `requirements.txt`, you can install manually:
    ```bash
    pip install unstructured langchain-unstructured[local] langchain-astradb pypdf python-dotenv
    ```

4.  **Install External Dependencies:**
    * **Poppler:** Required by `unstructured` for PDF processing.
        * Download from [https://github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases) (for Windows).
        * Extract and add the `bin` directory to your system's `PATH`.
    * **Tesseract OCR:** Required by `unstructured` for image OCR.
        * Download from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki) (for Windows).
        * Install and add the installation directory (e.g., `C:\Program Files\Tesseract-OCR`) to your system's `PATH`.

5.  **Set up AstraDB:**
    * Create an AstraDB instance and a database.
    * Obtain your AstraDB API Endpoint and Application Token.
    * Set these as environment variables (e.g., in a `.env` file):
        ```
        ASTRA_DB_API_ENDPOINT="your_api_endpoint"
        ASTRA_DB_APPLICATION_TOKEN="your_application_token"
        ASTRA_DB_KEYSPACE="your_keyspace" # Optional
        OPENAI_API_KEY="your_openai_api_key" # If using OpenAI embeddings/LLMs
        ```

## Usage

1.  Place your PDF documents (e.g., `llava.pdf`, `attention.pdf`) in the `./content/` directory.
2.  Run the Jupyter notebook `rag_test.ipynb` to see the RAG pipeline in action.
