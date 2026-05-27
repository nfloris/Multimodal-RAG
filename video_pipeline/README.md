# Video Pipeline

This folder contains the notebook:

- `rag_pipeline_videos.ipynb`

It builds and evaluates video RAG systems that combine visual frames, speech transcripts, timestamped video segments, and multimodal retrieval.

## What It Does

The video pipeline contains two complementary approaches:

- InternVideo2 shared video-text space:
  - Splits the video into fixed-length clips.
  - Encodes video clips with InternVideo2.
  - Encodes text queries into the same video-text space.
  - Stores clip embeddings in Qdrant.
  - Uses Whisper transcripts as context for answer generation.
  - Supports dense retrieval and optional cross-encoder reranking.

- Unified VLM + Whisper + BGE pipeline:
  - Samples representative video frames.
  - Summarizes frames with Qwen2.5-VL.
  - Extracts and transcribes the audio track with Whisper Turbo.
  - Aligns frame summaries with transcript windows.
  - Embeds the combined `VISION + TRANSCRIPT` text with BGE-M3.
  - Retrieves with dense search, BM25, hybrid search, and optional cross-encoder reranking.
  
<center><img src="../images/mmr4.jpg" alt="Text + video pipeline" width="75%"></center>
The final evaluation section computes:

- BERTScore
- answer precision and recall
- context recall
- evidence coverage
- must/should claim recall

Evidence coverage replaces plain timestamp coverage because a video segment can overlap the right time window while still missing the relevant visual or spoken evidence.

The notebook separates answer generation from the interactive evaluation table, so changing filters does not rerun retrieval, generation, or BERTScore.

## Platform Requirement

Run the full video notebook on Linux or WSL.

Native Windows is not supported for the complete InternVideo2 pipeline. InternVideo2 and related video dependencies are currently unreliable or broken on native Windows in this setup. If you are on Windows, use WSL with a Linux Python environment for this notebook.

The other pipelines can run on native Windows, but the video pipeline should be treated as Linux/WSL-only.

## Inputs and Outputs

- Put input video files in `video_pipeline/content/`.
- Cached transcripts, VLM summaries, and intermediate artifacts are written under `video_pipeline/cache/`.
- Qdrant collections are stored in the running Qdrant service.

The `cache/` directory is intentionally ignored by git.

## Requirements

Install the shared Python dependencies from the repository root inside the Linux or WSL environment:

```bash
pip install -r requirements.txt
```

Main Python packages used by this pipeline include:

- `transformers`, `torch`, `timm`, `einops`, `safetensors`
- `decord`, `qwen-vl-utils`
- `faster-whisper`, `ctranslate2`
- `sentence-transformers`, `FlagEmbedding`
- `qdrant-client`, `langchain-qdrant`
- `rank-bm25`
- `bert-score`
- `ipywidgets`
- `langchain-ollama`, optional for Ollama generation

Additional model/runtime requirements:

- InternVideo2-compatible Hugging Face checkpoints, such as `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` or the CLIP-style checkpoint used in the notebook.
- Qwen2.5-VL for frame summaries in the unified pipeline.
- Whisper or faster-whisper for speech transcription.
- BGE-M3 for the unified text embedding pipeline.

External services and system tools:

- Linux or WSL
- ffmpeg, required to extract the audio track from video
- Qdrant running locally, usually at `http://localhost:6333`
- Ollama, optional, if using an Ollama generation model

Install ffmpeg on Ubuntu/WSL with:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Start Qdrant from the repository root with:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

## Notes

- A CUDA GPU is strongly recommended. InternVideo2, Qwen2.5-VL, Whisper, BGE-M3, and reranking are expensive on CPU.
- If Qdrant runs on Windows and the notebook runs in WSL, use the Windows host IP in `QDRANT_URL` instead of `localhost`.
- If Ollama runs on Windows and the notebook runs in WSL, expose Ollama on Windows with `OLLAMA_HOST=0.0.0.0:11434` and set the notebook client URL to the Windows host IP, for example `OLLAMA_BASE_URL=http://<windows-host-ip>:11434`.
- Do not run the InternVideo2 cells in a native Windows Python environment.
