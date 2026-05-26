# Multimodal RAG Pipelines

This repository contains three local multimodal Retrieval-Augmented Generation
(RAG) pipelines for images/PDFs, audio, and video. Each pipeline is implemented
as a Jupyter notebook and has its own README with pipeline-specific details.

## Repository Structure

```text
Multimodal-RAG/
+-- image_pipeline/
|   +-- rag_pipeline_imgs.ipynb
|   +-- content/
|   +-- cache/
|   +-- README.md
+-- audio_pipeline/
|   +-- rag_pipeline_audios.ipynb
|   +-- content/
|   +-- cache/
|   +-- README.md
+-- video_pipeline/
|   +-- rag_pipeline_videos.ipynb
|   +-- content/
|   +-- cache/
|   +-- README.md
+-- requirements.txt
+-- setup.env
+-- qdrant_storage/
```

The `cache/` directories and local virtual environments are ignored by git.

## Pipelines

### Image Pipeline

Folder: `image_pipeline/`

Notebook: `rag_pipeline_imgs.ipynb`

This pipeline processes PDFs containing text, tables, charts, diagrams, and
images. It compares:

- VLM image/table summarization plus text embedding retrieval.
- CLIP shared text-image embedding retrieval.
- Dense retrieval, BM25, hybrid retrieval, and cross-encoder reranking.
- Answer generation with local Hugging Face models or Ollama.
- Evaluation with BERTScore, precision, recall, context recall, visual claim
  recall, and must/should claim recall.

See `image_pipeline/README.md` for details.

### Audio Pipeline

Folder: `audio_pipeline/`

Notebook: `rag_pipeline_audios.ipynb`

This pipeline compares two audio RAG strategies:

- Whisper transcript RAG: audio is transcribed, chunked, embedded, and retrieved
  as timestamped text.
- CLAP audio-text RAG: raw audio segments and text queries are embedded in a
  shared CLAP space, with Whisper transcripts used as LLM context.

The evaluation includes BERTScore, precision, recall, context recall, timestamp
coverage, and must/should claim recall.

See `audio_pipeline/README.md` for details.

### Video Pipeline

Folder: `video_pipeline/`

Notebook: `rag_pipeline_videos.ipynb`

This pipeline compares:

- InternVideo2 shared video-text retrieval over video clips.
- A unified VLM + Whisper + BGE pipeline that turns sampled frames and speech
  into aligned text evidence.

The evaluation includes BERTScore, precision, recall, context recall, evidence
coverage, and must/should claim recall.

Important: run the full video pipeline on Linux or WSL. Native Windows is not
supported for the complete InternVideo2 workflow because InternVideo2 and
related video dependencies are unreliable or broken in this setup.

See `video_pipeline/README.md` for details.

## Requirements

Use Python 3.10 or 3.11. A CUDA-capable GPU is strongly recommended for the
full notebooks, especially Qwen2.5-VL, CLIP, CLAP, Whisper, InternVideo2,
BGE-M3, and cross-encoder reranking.

Install Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Main dependency groups include:

- Parsing and PDF extraction: `unstructured`, `pdfminer.six`, `pdf2image`,
  `pypdf`, `pytesseract`
- Embeddings and reranking: `sentence-transformers`, `FlagEmbedding`,
  `open-clip-torch`, `rank-bm25`
- Multimodal and language models: `transformers`, `qwen-vl-utils`, `timm`,
  `accelerate`, `bitsandbytes`
- Audio/video: `librosa`, `faster-whisper`, `ctranslate2`, `decord`
- Vector storage and orchestration: `qdrant-client`, `langchain`,
  `langchain-qdrant`, `langchain-ollama`
- Evaluation and notebooks: `bert-score`, `ipywidgets`, `jupyter`

External tools:

- Docker, for local Qdrant
- Qdrant, usually at `http://localhost:6333`
- Poppler and Tesseract, for the image/PDF pipeline
- ffmpeg, for audio/video decoding and video audio extraction
- Ollama, optional, for local text generation

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On Linux or WSL:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional configuration can be placed in `setup.env`. The notebooks load this
file with `python-dotenv`.

Common variables:

```env
QDRANT_URL=http://localhost:6333
GENERATION_BACKEND=ollama
GENERATION_MODEL=mistral-nemo:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

If Ollama runs on Windows and a notebook runs inside WSL, expose Ollama on
Windows with `OLLAMA_HOST=0.0.0.0:11434`, then set `OLLAMA_BASE_URL` in the
notebook or `setup.env` to the Windows host IP, for example:

```env
OLLAMA_BASE_URL=http://172.28.16.1:11434
```

Keep `OLLAMA_HOST` and `OLLAMA_BASE_URL` separate: `OLLAMA_HOST` controls where
the Ollama server listens, while `OLLAMA_BASE_URL` is the URL the Python client
connects to.

## Start Qdrant

From the repository root on Windows PowerShell:

```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 `
  -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
```

From Linux or WSL:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

If a container named `qdrant` already exists, start it with:

```bash
docker start qdrant
```

## Running the Notebooks

1. Activate the virtual environment.
2. Start Qdrant.
3. Put input files in the relevant `content/` folder.
4. Open the target notebook with Jupyter or VS Code.
5. Run the notebook cells in order.
6. Run the evaluation generation cell once.
7. Use the interactive evaluation table to filter cached evaluation results.

Input locations:

- PDFs: `image_pipeline/content/`
- Audio files: `audio_pipeline/content/`
- Video files: `video_pipeline/content/`

## Evaluation Design

Each notebook has an evaluation section near the end. The evaluation is split
into two parts:

- Answer and metric generation: performs retrieval, answer generation, and
  metric computation once.
- Interactive table: filters cached rows without rerunning generation or
  BERTScore.

This keeps the expensive operations separate from lightweight visualization.

## Notes

- Model downloads can be large. Hugging Face may download `model.safetensors`
  files the first time a model is used.
- The video pipeline should be run in Linux or WSL only.
- The `cache/` folders can be deleted if you want to regenerate summaries,
  transcripts, or evaluation artifacts.
- Qdrant collections can be reset with the reset variables defined inside each
  notebook.
