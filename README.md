# CortexRAG

> A developer-first, fully local conversational AI pipeline combining **Retrieval-Augmented Generation** (RAG), **Speech-to-Text** (STT), and **Text-to-Speech** (TTS) powered by a QWEN model running on your own hardware. Built for builders who want full control — no cloud, no data leakage.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
   - [Ingest Documents](#1-ingest-documents)
   - [Text Chat](#2-text-chat)
   - [Voice Chat](#3-voice-chat)
   - [Single Query](#4-single-query)
   - [Status Check](#5-status-check)
8. [Project Structure](#project-structure)
9. [How It Works](#how-it-works)
10. [Development](#development)
11. [NVIDIA NIM (Optional Cloud LLM)](#nvidia-nim-optional-cloud-llm)
12. [Security](#security)
13. [Troubleshooting](#troubleshooting)
14. [License](#license)

---

## Overview

This project implements an **agentic** RAG pipeline where the LLM actively decides whether to consult a knowledge base before answering. You speak (or type) a question; the system:

1. **Transcribes** your speech with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (local Whisper model)
2. **Classifies** your query — does it need retrieval?
3. **Retrieves** relevant document chunks from a ChromaDB vector store (if needed)
4. **Generates** an answer with the locally running QWEN 3.5 4B model
5. **Speaks** the response aloud via [Microsoft Edge TTS](https://github.com/rany2/edge-tts) (free, no API key)

Everything runs **100% locally** on your LAN — no data leaves your network.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       User Interface (CLI)                       │
│              text chat │ voice chat │ single ask                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   RAGAgent           │  ← orchestrates all steps
                    └──────────┬──────────┘
          ┌───────────────────┼───────────────────┐
          │                   │                   │
 ┌────────▼───────┐  ┌────────▼───────┐  ┌───────▼────────┐
 │  STT Module    │  │  LLM Client    │  │  TTS Module    │
 │ faster-whisper │  │  QWEN 3.5 4B   │  │  Edge TTS      │
 │  (local WAV)   │  │  (LAN server)  │  │  (MP3 + play)  │
 └────────────────┘  └────────┬───────┘  └────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  RAG Sub-system      │
                   │  ┌────────────────┐ │
                   │  │ DocumentLoader │ │  PDF, TXT, MD
                   │  └───────┬────────┘ │
                   │  ┌───────▼────────┐ │
                   │  │   Embedder     │ │  sentence-transformers
                   │  └───────┬────────┘ │
                   │  ┌───────▼────────┐ │
                   │  │  VectorStore   │ │  ChromaDB (local)
                   │  └───────┬────────┘ │
                   │  ┌───────▼────────┐ │
                   │  │   Retriever    │ │  cosine similarity
                   │  └────────────────┘ │
                   └─────────────────────┘
```

### Agentic Decision Loop (ReAct-style)

```
User Input
    │
    ▼
[OBSERVE]  STT → text
    │
    ▼
[THINK]    LLM classifies: RETRIEVE or DIRECT?
    │
    ├── RETRIEVE → [ACT] vector search → top-k chunks
    │                                        │
    │                                        ▼
    └── DIRECT ──────────────────► [RESPOND] LLM generates answer
                                             │
                                             ▼
                                  [SPEAK] TTS → audio playback
```

---

## Features

| Feature | Detail |
|---|---|
| **100% local** | LLM, STT, and vector store all run on your hardware |
| **Agentic retrieval** | LLM decides whether retrieval is needed per query |
| **Multi-format ingestion** | PDF, Markdown, plain text |
| **Voice I/O** | Microphone → Whisper STT → Edge TTS playback |
| **Persistent KB** | ChromaDB persists between restarts |
| **Deduplication** | Re-ingesting the same file is safe (no duplicates) |
| **CLI** | Rich terminal interface with `typer` |
| **Configurable** | All settings in `.env` with sensible defaults |

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| LM Studio (or compatible server) | latest |
| QWEN 3.5 4B model loaded | — |
| Microphone | for voice mode |
| Internet access | for Edge TTS synthesis only |

> **Local LLM Server**  
> The project is configured for a server at `http://192.168.68.113:1234` using the API contract below.  
> Adjust `CORTEXRAG_LLM_BASE_URL` in your `.env` if your server address differs.
>
> ```bash
> curl http://192.168.68.113:1234/api/v1/chat \
>   -H "Content-Type: application/json" \
>   -d '{
>     "model": "qwen3.5-4b",
>     "system_prompt": "You answer only in rhymes.",
>     "input": "What is your favorite color?"
>   }'
> ```

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd cortexrag

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install the package in editable mode (includes all dependencies)
pip install -e .

# 4. (Development) Install dev extras
pip install -e ".[dev]"

# 5. Copy the example environment file and edit as needed
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

> **Note on Windows audio packages**  
> `sounddevice` requires the PortAudio library. Install it via:  
> ```bash
> pip install sounddevice
> ```  
> If you encounter errors, download a pre-built wheel from  
> https://www.lfd.uci.edu/~gohlke/pythonlibs/.

---

## Configuration

All settings are controlled via environment variables (with the `CORTEXRAG_` prefix) or a `.env` file in the project root. Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `CORTEXRAG_LLM_BASE_URL` | `http://192.168.68.113:1234` | LLM server base URL |
| `CORTEXRAG_LLM_MODEL` | `qwen3.5-4b` | Model identifier |
| `CORTEXRAG_LLM_SYSTEM_PROMPT` | *(helpful assistant)* | Default system prompt |
| `CORTEXRAG_LLM_TIMEOUT` | `60` | HTTP timeout (seconds) |
| `CORTEXRAG_VECTOR_STORE_PATH` | `./data/chroma_db` | ChromaDB persist directory |
| `CORTEXRAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CORTEXRAG_CHUNK_SIZE` | `512` | Characters per document chunk |
| `CORTEXRAG_CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `CORTEXRAG_RETRIEVAL_TOP_K` | `3` | Documents returned per query |
| `CORTEXRAG_SIMILARITY_THRESHOLD` | `0.3` | Minimum relevance score |
| `CORTEXRAG_WHISPER_MODEL` | `base` | Whisper size (tiny/base/small/medium/large-v3) |
| `CORTEXRAG_WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `CORTEXRAG_TTS_VOICE` | `en-US-AriaNeural` | Edge TTS voice name |
| `CORTEXRAG_TTS_RATE` | `+0%` | Speech rate offset |
| `CORTEXRAG_AUDIO_SAMPLE_RATE` | `16000` | Recording sample rate (Hz) |
| `CORTEXRAG_AUDIO_SILENCE_DURATION` | `2.0` | Seconds of silence to end recording |
| `CORTEXRAG_AUDIO_MAX_DURATION` | `30` | Maximum recording length (seconds) |

---

## Usage

### 1. Ingest Documents

```bash
# Index a single file
cortexrag ingest data/documents/manual.pdf

# Index an entire folder (recursively)
cortexrag ingest data/documents/

# Non-recursive directory scan
cortexrag ingest data/documents/ --no-recursive
```

Supported formats: `.pdf`, `.txt`, `.md`, `.markdown`

---

### 2. Text Chat

```bash
# Interactive text chat (no audio)
cortexrag chat

# Text chat with spoken responses
cortexrag chat --tts
```

```
╭─── CortexRAG — Text Chat ────────────────────────────────────────────╮
│ Model: qwen3.5-4b @ http://192.168.68.113:1234             │
│ Knowledge base: 142 chunk(s) indexed                        │
│ TTS: off                                                    │
╰─────────────────────────────────────────────────────────────╯

You: What does the warranty policy say?

Assistant [RAG]:
Based on the retrieved documents, the warranty policy states...

Sources: /data/documents/warranty.pdf
```

---

### 3. Voice Chat

```bash
# Full voice pipeline (microphone → STT → RAG → LLM → TTS speaker)
cortexrag voice

# Voice input only (no TTS output)
cortexrag voice --no-tts
```

The system prints:
- The transcription of your speech
- Whether retrieval was used
- The full response text (even while speaking)

---

### 4. Single Query

```bash
# Ask a single question and exit
cortexrag ask "What is the refund process?"

# With TTS
cortexrag ask "Summarise the product roadmap." --tts
```

---

### 5. Status Check

```bash
cortexrag status
```

```
┌─────────────────────────────────────────────────────────┐
│                    CortexRAG Status                         │
├─────────────────┬───────────────────────────────────────┤
│ LLM Server      │ http://192.168.68.113:1234             │
│ LLM Model       │ qwen3.5-4b                             │
│ Server Status   │ ✓ reachable                            │
│ Vector Store    │ ./data/chroma_db                       │
│ Indexed Chunks  │ 142                                    │
│ Whisper Model   │ base                                   │
│ TTS Voice       │ en-US-AriaNeural                       │
└─────────────────┴───────────────────────────────────────┘
```

---

## Project Structure

```
cortexrag/
├── README.md                  # This file
├── pyproject.toml             # Project metadata & dependencies (PEP 517/518)
├── requirements.txt           # Flat dependency list
├── .env.example               # Template for environment variables
├── .gitignore
├── main.py                    # Top-level convenience entry point
│
├── src/
│   └── cortexrag/           # Installable Python package
│       ├── __init__.py
│       ├── __main__.py        # python -m cortexrag
│       ├── main.py            # Typer CLI definitions
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py    # Pydantic-settings configuration
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   └── client.py      # HTTP client for the local LLM API
│       │
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── document_loader.py  # File → DocumentChunk
│       │   ├── embedder.py         # sentence-transformers wrapper
│       │   ├── vector_store.py     # ChromaDB persistence
│       │   └── retriever.py        # High-level search + ingestion façade
│       │
│       ├── stt/
│       │   ├── __init__.py
│       │   └── speech_to_text.py   # Microphone recording + Whisper
│       │
│       ├── tts/
│       │   ├── __init__.py
│       │   └── text_to_speech.py   # Edge TTS synthesis + playback
│       │
│       └── agent/
│           ├── __init__.py
│           └── rag_agent.py        # ReAct-style agentic orchestrator
│
├── tests/
│   ├── __init__.py
│   ├── test_llm_client.py
│   ├── test_rag.py
│   ├── test_stt.py
│   ├── test_tts.py
│   └── test_agent.py
│
└── data/
    └── documents/             # Drop your source documents here
```

---

## How It Works

### Retrieval Decision (Agentic Behaviour)

Before performing any vector search, the agent asks the LLM:

> *"Does this query require searching a private knowledge base (RETRIEVE) or can it be answered from general knowledge (DIRECT)?"*

This prevents unnecessary latency on conversational messages ("hello", "thanks") and ensures focused retrieval where it matters.

### Document Ingestion Pipeline

```
File on disk
    │
    ▼ DocumentLoader
Recursive text splitting (RecursiveCharacterTextSplitter)
    │  chunk_size=512, chunk_overlap=64
    ▼ Embedder
Dense vectors (all-MiniLM-L6-v2, 384 dims, L2-normalised)
    │
    ▼ VectorStore (ChromaDB)
Persistent HNSW index (cosine space)
```

### RAG Answer Prompt

When retrieval is used, the following prompt structure is sent to the LLM:

```
[System]
You are a helpful AI assistant...

[User]
--- Retrieved Context ---
[1] Source: /data/documents/manual.pdf (chunk 3, score: 0.87)
<chunk text>

[2] Source: /data/documents/faq.txt (chunk 1, score: 0.74)
<chunk text>
--- End Context ---

Question: <user query>

Answer:
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run the full test suite
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type check
mypy src/

# List available Edge TTS voices
edge-tts --list-voices | findstr "en-US"   # Windows
edge-tts --list-voices | grep en-US        # macOS / Linux
```

---

## NVIDIA NIM (Optional Cloud LLM)

The file `_test_nvidia.py` provides an optional integration with **NVIDIA NIM** (hosted cloud models such as `deepseek-ai/deepseek-v3.1`). This is **not** required for the core local pipeline but useful for experimentation.

### Setup

1. Create an account at [build.nvidia.com](https://build.nvidia.com) and generate an API key.
2. Add it to your **local** `.env` file (never commit this file):
   ```dotenv
   NVIDIA_API_KEY=nvapi-<your-key-here>
   ```
3. Run the test script:
   ```bash
   python _test_nvidia.py
   ```

> **Security note:** `_test_nvidia.py` and `.env` are both listed in `.gitignore` and will **never** be committed to version control. Only `.env.example` (with a placeholder value) is committed.

---

## Security

| What | How it is protected |
|---|---|
| `.env` (real secrets) | Listed in `.gitignore` — never committed |
| `_test_nvidia.py` | Listed in `.gitignore` — never committed |
| NVIDIA API key | Read from `NVIDIA_API_KEY` env var via `python-dotenv` |
| LLM server URL | Configurable via `CORTEXRAG_LLM_BASE_URL` — defaults to LAN address |
| ChromaDB data | In `data/chroma_db/` — listed in `.gitignore` |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot connect to LLM server` | Ensure LM Studio is running on `192.168.68.113:1234` |
| `No module named 'sounddevice'` | `pip install sounddevice` (needs PortAudio) |
| `No module named 'faster_whisper'` | `pip install faster-whisper` |
| Slow first query | Whisper and embedding models are downloaded on first use |
| TTS has no audio output | Check default audio output device in system settings |
| `chromadb` import error | `pip install chromadb` (needs SQLite ≥ 3.35) |
| Empty transcription | Speak louder or reduce `CORTEXRAG_AUDIO_SILENCE_DURATION` |
| `NVIDIA_API_KEY is not set` | Add `NVIDIA_API_KEY=nvapi-...` to your `.env` file |

---

## License

MIT — see [LICENSE](LICENSE) for details.
