# Audio Pipeline

This folder contains the notebook:

- `rag_pipeline_audios.ipynb`

It builds and evaluates audio RAG systems over spoken content, comparing transcript-first retrieval with audio-text shared-space retrieval.

## What It Does

The audio pipeline implements two main approaches:

- Whisper transcript RAG:
  - Transcribes audio with Whisper or faster-whisper.
  - Splits the transcript into timestamped chunks.
  - Embeds transcript chunks with a text embedding model such as BGE-M3.
  - Retrieves with dense search, BM25, hybrid search, and optional cross-encoder reranking.

- CLAP audio-text RAG:
  - Splits raw audio into overlapping fixed-length segments.
  - Embeds audio segments with CLAP.
  - Encodes text queries with CLAP's text encoder in the same shared space.
  - Retrieves audio segments directly, then uses the attached Whisper transcript as LLM context.
  - Supports dense CLAP retrieval and optional transcript reranking. BM25 is intentionally not treated as a CLAP retrieval mode.

<center><img src="../images/mmr3.jpg" alt="Text + audio pipeline" width="75%"></center>

The final evaluation section computes:

- BERTScore
- answer precision and recall
- context recall
- timestamp coverage
- must/should claim recall

The notebook separates answer generation from the interactive evaluation table, so changing filters does not rerun retrieval, generation, or BERTScore.

## Inputs and Outputs

- Put input audio files in `audio_pipeline/content/`.
- Cached transcripts and intermediate artifacts are written under `audio_pipeline/cache/`.
- Qdrant collections are stored in the running Qdrant service.

The `cache/` directory is intentionally ignored by git.

## Requirements

Install the shared Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Main Python packages used by this pipeline include:

- `faster-whisper`, `ctranslate2`
- `transformers`
- `librosa`, `numpy`
- `sentence-transformers`, `FlagEmbedding`
- `qdrant-client`, `langchain-qdrant`
- `rank-bm25`
- `bert-score`
- `ipywidgets`
- `langchain-ollama`, optional for Ollama generation

External services and system tools:

- Qdrant running locally, usually at `http://localhost:6333`
- Ollama, optional, if using an Ollama generation model
- ffmpeg, recommended for robust MP3 and audio decoding support

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

- A CUDA GPU is recommended for faster Whisper transcription, CLAP embeddings, BGE-M3, and reranking.
- CPU execution is possible for smaller tests, but full notebook runs can be slow.
- If Ollama runs on Windows and the notebook runs in WSL, expose Ollama on Windows with `OLLAMA_HOST=0.0.0.0:11434` and set the notebook client URL to the Windows host IP, for example `OLLAMA_BASE_URL=http://<windows-host-ip>:11434`.
