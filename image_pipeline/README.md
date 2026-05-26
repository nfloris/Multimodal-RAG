# Image Pipeline

This folder contains the notebook:

- `rag_pipeline_imgs.ipynb`

It builds and evaluates a multimodal RAG pipeline for PDF documents that contain text, tables, charts, diagrams, and other visual elements.

## What It Does

The image pipeline compares multiple RAG strategies for document understanding:

- Extracts text chunks and visual elements from PDFs with `unstructured`.
- Summarizes images and table images with Qwen2.5-VL.
- Embeds text and image summaries with a local text embedding model such as BGE-M3.
- Builds a Qdrant vector index for summary-based retrieval.
- Builds a CLIP shared-space index where text and images are embedded directly into the same vector space.
- Supports BM25, dense retrieval, hybrid retrieval, and cross-encoder reranking where appropriate.
- Generates final answers with either a local Hugging Face model or Ollama.
- Evaluates RAG approaches with BERTScore, precision, recall, context recall, visual claim recall, and must/should claim recall.
- Provides an interactive evaluation table that filters cached evaluation results without regenerating answers.

## Inputs and Outputs

- Put input PDFs in `image_pipeline/content/`.
- Cached summaries and intermediate artifacts are written under `image_pipeline/cache/`.
- Qdrant collections are stored in the running Qdrant service, not inside this folder.

The `cache/` directory is intentionally ignored by git.

## Requirements

Install the shared Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Main Python packages used by this pipeline include:

- `unstructured`, `unstructured-inference`, `pdfminer.six`, `pdf2image`, `pypdf`
- `transformers`, `qwen-vl-utils`, `torch`
- `sentence-transformers`, `FlagEmbedding`, `open-clip-torch`
- `langchain`, `langchain-qdrant`, `langchain-ollama`
- `qdrant-client`, `rank-bm25`
- `bert-score`, `ipywidgets`

External services and system tools:

- Qdrant running locally, usually at `http://localhost:6333`
- Poppler, required for PDF rendering and extraction
- Tesseract OCR, required by `unstructured` for OCR-heavy PDFs
- Ollama, optional, if `GENERATION_MODEL` is an Ollama model such as `mistral-nemo:latest`

Start Qdrant from the repository root with:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 ^
  -v %cd%/qdrant_storage:/qdrant/storage qdrant/qdrant
```

On Linux or WSL, use:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

## Notes

- A CUDA GPU is strongly recommended for Qwen2.5-VL, CLIP, BGE-M3, and reranking.
- If Ollama is used on Windows, set the client URL in `setup.env` with `OLLAMA_BASE_URL=http://127.0.0.1:11434`.
- Keep `OLLAMA_HOST` and `OLLAMA_BASE_URL` conceptually separate: `OLLAMA_HOST` is the server bind address, while `OLLAMA_BASE_URL` is the URL the notebook connects to.
