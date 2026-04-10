"""
Quick smoke-test script — runs without any heavy ML models or audio hardware.
Shows configuration, verifies all modules import cleanly, and tests the LLM
server connection.
"""
import sys

# ── 1. Imports ────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Agentic RAG STT/TTS — Smoke Test")
print("=" * 60)

modules = [
    ("Config / Settings",   "cortexrag.config.settings",    "Settings"),
    ("LLM Client",          "cortexrag.llm.client",          "LLMClient"),
    ("Document Loader",     "cortexrag.rag.document_loader", "DocumentLoader"),
    ("Embedder",            "cortexrag.rag.embedder",        "Embedder"),
    ("VectorStore",         "cortexrag.rag.vector_store",    "VectorStore"),
    ("Retriever",           "cortexrag.rag.retriever",       "Retriever"),
    ("STT",                 "cortexrag.stt.speech_to_text",  "SpeechToText"),
    ("TTS",                 "cortexrag.tts.text_to_speech",  "TextToSpeech"),
    ("RAGAgent",            "cortexrag.agent.rag_agent",     "RAGAgent"),
]

print("\n[1] Module imports")
all_ok = True
for label, module_path, cls_name in modules:
    try:
        mod = __import__(module_path, fromlist=[cls_name])
        getattr(mod, cls_name)
        print(f"   OK  {label}")
    except Exception as e:
        print(f"  ERR  {label}: {e}")
        all_ok = False

# ── 2. Settings ───────────────────────────────────────────────────────────────
print("\n[2] Configuration (from defaults / .env)")
from cortexrag.config.settings import get_settings
s = get_settings()
rows = [
    ("LLM Base URL",        s.llm_base_url),
    ("LLM Model",           s.llm_model),
    ("Embedding Model",     s.embedding_model),
    ("Chunk size",          str(s.chunk_size)),
    ("Retrieval Top-K",     str(s.retrieval_top_k)),
    ("Whisper Model",       s.whisper_model),
    ("Whisper Device",      s.whisper_device),
    ("TTS Voice",           s.tts_voice),
    ("Vector Store Path",   s.vector_store_path),
]
for k, v in rows:
    print(f"   {k:<24} {v}")

# ── 3. LLM server connectivity ────────────────────────────────────────────────
print("\n[3] LLM Server connectivity")
from cortexrag.llm.client import LLMClient, LLMClientError
llm = LLMClient(s)
reachable = llm.health_check()
if reachable:
    print(f"   OK  Server at {s.llm_base_url} is REACHABLE")
    # Try a real chat call
    print(f"   >>  Sending test query to {s.llm_model} ...")
    try:
        response = llm.chat(
            "Say hello in exactly 5 words.",
            system_prompt="You are a concise assistant.",
        )
        print(f"   LLM reply: {response.content.strip()}")
    except LLMClientError as e:
        print(f"   WARN chat call failed: {e}")
else:
    print(f"   WARN  Server at {s.llm_base_url} is NOT reachable")
    print("   Make sure LM Studio is running and the model is loaded.")

# ── 4. Document Loader ────────────────────────────────────────────────────────
print("\n[4] Document Loader (in-memory)")
from cortexrag.rag.document_loader import DocumentLoader
loader = DocumentLoader(s)
chunks = loader.load_text(
    "The Agentic RAG system combines retrieval-augmented generation with "
    "speech recognition and synthesis. " * 10,
    source="smoke-test",
)
print(f"   OK  Produced {len(chunks)} chunk(s) from inline text")
if chunks:
    preview = chunks[0].text[:80].replace("\n", " ")
    print(f"   Chunk #0 preview: {preview!r}...")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_ok:
    print("  All modules imported successfully.")
else:
    print("  Some modules had import errors (see above).")
print(f"  LLM server: {'REACHABLE' if reachable else 'NOT REACHABLE'}")
print("  To ingest docs:  python main.py ingest data/documents/")
print("  To start chat:   python main.py chat")
print("  To use voice:    python main.py voice")
print("=" * 60)
