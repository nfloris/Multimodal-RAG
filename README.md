> This project explores multimodal Retrieval-Augmented Generation locally,
> without relying on external APIs, comparing retrieval strategies across
> modalities and measuring their impact on answer quality through a structured
> evaluation framework.

# Multimodal RAG Pipelines

This repository contains three local multimodal RAG pipelines for images/PDFs,
audio, and video, each implemented as a Jupyter notebook. Two retrieval
strategies are compared across all pipelines:

- **Shared semantic space** — query and source modality are projected into the
  same vector space.

- **Unified translation** — each modality is first converted to text, then
  embedded with a text model. Retrieval operates entirely in text space.

Comparing these two strategies across modalities is the core experimental
question of this repository.

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

Git ignores the `cache/` directories and local virtual environments.

## Pipelines

### Image Pipeline
Folder: `image_pipeline/`
Notebook: `rag_pipeline_imgs.ipynb`

![Pipeline](images/rag_image)

This pipeline processes PDFs containing text, tables, charts, diagrams, and
images. It compares two core retrieval strategies:

- **VLM summarization + text embedding**: images and
  tables are summarized by a VLM, and the resulting text is embedded. Two sub-modes are evaluated:
  - *With source linking*: the original image is also passed to the LLM at
    generation time alongside its summary.
  - *Without source linking*: only the VLM summary is used, both for
    retrieval and generation.

  ![Comparison](images/rag_image_modalities)

- **CLIP shared text-image embedding** (shared semantic space): text chunks
  and images are embedded directly into a joint vector space, with no intermediate summarization step.

Both strategies are combined with dense retrieval, BM25, hybrid retrieval,
and cross-encoder reranking. Answer generation uses local Hugging Face models
or Ollama. Evaluation covers BERTScore, must/should claim recall, visual claim
recall, and context recall.

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

## General Setup Flow

### 1. Clone the Repository

```bash
git clone https://github.com/theserenecoder/MultiModel_RAG
cd MultiModel_RAG
```

If your local folder is named `Multimodal-RAG`, use that folder name instead in
the `cd` command.

### 2. Create a Virtual Environment

Windows:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
py -3.11 -m venv venv
venv\Scripts\activate.bat
```

Linux, macOS, or WSL:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you are only testing a small subset and do not want to use the full
`requirements.txt`, install the relevant packages manually. For example:

```bash
pip install python-dotenv unstructured pypdf pdf2image pytesseract \
  transformers sentence-transformers qdrant-client langchain \
  langchain-qdrant langchain-ollama bert-score ipywidgets
```

The full pipeline notebooks need the complete `requirements.txt`.

### 4. Pre-Download Hugging Face Models

The repository includes `download_hf_models.py`, which pre-caches common
Hugging Face models used by the notebooks. This is optional, but useful before
long notebook runs because it separates model downloads from execution.

Download the default open models:

```bash
python download_hf_models.py --profile defaults
```

Download a broader set of open model options:

```bash
python download_hf_models.py --profile all-open
```

For gated models, log in first and pass the gated option:

```bash
huggingface-cli login
python download_hf_models.py --profile all --include-gated
```

### 5. Install External Dependencies

Poppler is required by `unstructured` and `pdf2image` for PDF processing.

On Windows:

- Download Poppler from `https://github.com/oschwartz10612/poppler-windows/releases`.
- Extract it.
- Add the extracted `bin` directory to your system `PATH`.

Tesseract OCR is required by `unstructured` for OCR-heavy PDFs.

On Windows:

- Download Tesseract from `https://github.com/UB-Mannheim/tesseract/wiki`.
- Install it.
- Add the install directory, usually `C:\Program Files\Tesseract-OCR`, to your
  system `PATH`.

On Ubuntu or WSL:

```bash
sudo apt update
sudo apt install -y poppler-utils tesseract-ocr ffmpeg
```

`ffmpeg` is required for video audio extraction and is recommended for robust
audio decoding.

### 6. Set Up Qdrant

Qdrant is used as the local vector database for embeddings.

From the repository root on Linux or WSL:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

From Windows PowerShell:

```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 `
  -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
```

REST dashboard:

```text
http://localhost:6333/dashboard
```

If the container already exists, start it with:

```bash
docker start qdrant
```

### 7. Configure Optional Environment Variables

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
