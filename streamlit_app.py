"""
Streamlit UI for the Agentic RAG + STT + TTS system.

Launch:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG · STT & TTS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure src/ is importable when run from project root ──────────────────────
_src = Path(__file__).parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# ── Lazy imports (heavy ML libs) cached as singletons ─────────────────────────

@st.cache_resource(show_spinner="Loading settings…")
def _get_settings():
    from agentic_rag.config.settings import get_settings
    return get_settings()


@st.cache_resource(show_spinner="Connecting to LLM…")
def _get_llm():
    from agentic_rag.llm.client import LLMClient
    return LLMClient(_get_settings())


@st.cache_resource(show_spinner="Initialising knowledge base…")
def _get_retriever():
    from agentic_rag.rag.retriever import Retriever
    return Retriever(_get_settings())


@st.cache_resource(show_spinner="Loading Whisper STT model…")
def _get_stt():
    from agentic_rag.stt.speech_to_text import SpeechToText
    return SpeechToText(_get_settings())


@st.cache_resource(show_spinner="Loading TTS…")
def _get_tts():
    from agentic_rag.tts.text_to_speech import TextToSpeech
    return TextToSpeech(_get_settings())


# ── Session state helpers ──────────────────────────────────────────────────────

def _init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {"role": "user"|"assistant", "content": str, "badge": str}
    if "ingest_log" not in st.session_state:
        st.session_state.ingest_log = []


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — System Status
# ═════════════════════════════════════════════════════════════════════════════

def _sidebar():
    settings = _get_settings()

    st.sidebar.title("🤖 Agentic RAG")
    st.sidebar.caption("STT · RAG · LLM · TTS — all local")
    st.sidebar.divider()

    # LLM health
    with st.sidebar.expander("⚙️ LLM Server", expanded=True):
        llm = _get_llm()
        reachable = llm.health_check()
        dot = "🟢" if reachable else "🔴"
        st.write(f"{dot} **{settings.llm_base_url}**")
        st.write(f"Model: `{settings.llm_model}`")
        if not reachable:
            st.warning("Server unreachable — start LM Studio.")

    # Knowledge base
    with st.sidebar.expander("📚 Knowledge Base", expanded=True):
        retriever = _get_retriever()
        count = retriever.document_count()
        st.metric("Indexed chunks", count)
        st.write(f"Embed model: `{settings.embedding_model}`")
        st.write(f"Top-K: `{settings.retrieval_top_k}`")
        if st.button("🗑️ Reset knowledge base", type="secondary", use_container_width=True):
            retriever.reset()
            st.session_state.ingest_log = []
            st.cache_resource.clear()
            st.success("Knowledge base cleared.")
            st.rerun()

    # STT / TTS config
    with st.sidebar.expander("🎙️ STT / TTS Config"):
        st.write(f"Whisper: `{settings.whisper_model}` on `{settings.whisper_device}`")
        st.write(f"TTS voice: `{settings.tts_voice}`")
        st.caption("Edit `.env` to change these settings.")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Chat
# ═════════════════════════════════════════════════════════════════════════════

def _tab_chat():
    st.subheader("💬 Chat with your knowledge base")

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                badge_color = "green" if msg.get("badge") == "RAG" else "blue"
                st.markdown(
                    f":{badge_color}[{'📄 RAG' if msg.get('badge') == 'RAG' else '💡 Direct'}]  "
                    f"{msg['content']}"
                )
                if msg.get("sources"):
                    st.caption("📎 Sources: " + ", ".join(msg["sources"]))
            else:
                st.markdown(msg["content"])

    # Input box
    user_input = st.chat_input("Ask anything…")
    if not user_input:
        return

    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            from agentic_rag.agent.rag_agent import RAGAgent
            agent = RAGAgent(_get_settings())
            # Inject cached retriever & LLM to skip re-init
            agent._retriever = _get_retriever()
            agent._llm = _get_llm()
            response = agent.query(user_input)

        badge = "RAG" if response.used_retrieval else "Direct"
        badge_color = "green" if badge == "RAG" else "blue"
        st.markdown(
            f":{badge_color}[{'📄 RAG' if badge == 'RAG' else '💡 Direct'}]  "
            f"{response.answer}"
        )

        sources = sorted({doc.source for doc in response.retrieved_docs}) if response.retrieved_docs else []
        if sources:
            st.caption("📎 Sources: " + ", ".join(Path(s).name for s in sources))

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response.answer,
        "badge": badge,
        "sources": [Path(s).name for s in sources],
    })

    # Clear history button
    if st.session_state.chat_history:
        if st.button("🧹 Clear chat history", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Ingest Documents
# ═════════════════════════════════════════════════════════════════════════════

def _tab_ingest():
    st.subheader("📥 Ingest Documents into the Knowledge Base")
    st.caption("Upload PDF, TXT, or Markdown files to index them for retrieval.")

    retriever = _get_retriever()

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Drop files here",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            help="Supported: PDF, TXT, Markdown",
        )

        if uploaded and st.button("⚡ Index selected files", type="primary", use_container_width=True):
            progress = st.progress(0, text="Starting…")
            total_added = 0
            for i, file in enumerate(uploaded):
                progress.progress((i + 1) / len(uploaded), text=f"Indexing {file.name}…")
                suffix = Path(file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                try:
                    added = retriever.ingest_file(tmp_path)
                    total_added += added
                    log_entry = f"✅ **{file.name}** — {added} new chunk(s)"
                except Exception as e:
                    log_entry = f"❌ **{file.name}** — {e}"
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

                st.session_state.ingest_log.append(log_entry)

            progress.empty()
            st.success(f"Done! {total_added} new chunk(s) added. Total: {retriever.document_count()}")
            st.cache_resource.clear()
            st.rerun()

    with col2:
        st.markdown("**Ingest log**")
        if st.session_state.ingest_log:
            for entry in reversed(st.session_state.ingest_log[-15:]):
                st.markdown(entry)
        else:
            st.caption("No files ingested yet.")

    st.divider()

    # Inline text ingestion
    st.markdown("**Or paste text directly**")
    source_name = st.text_input("Source label", value="inline-paste", key="ingest_source")
    raw_text = st.text_area("Text to index", height=160, key="ingest_text",
                            placeholder="Paste any text here and click Index…")
    if st.button("⚡ Index pasted text", use_container_width=True) and raw_text.strip():
        with st.spinner("Indexing…"):
            added = retriever.ingest_text(raw_text, source=source_name or "inline")
        st.success(f"Added {added} chunk(s). Total: {retriever.document_count()}")
        st.session_state.ingest_log.append(f"✅ **{source_name}** (inline) — {added} chunk(s)")
        st.cache_resource.clear()


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Speech to Text
# ═════════════════════════════════════════════════════════════════════════════

def _tab_stt():
    st.subheader("🎙️ Speech-to-Text (Whisper)")
    st.caption("Upload a WAV or MP3 audio file to transcribe it using the local Whisper model.")

    settings = _get_settings()
    st.info(
        f"Whisper model: **{settings.whisper_model}** · device: **{settings.whisper_device}** · "
        f"compute: **{settings.whisper_compute_type}**"
    )

    audio_file = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="stt_upload",
        help="Any format Whisper supports",
    )

    if audio_file:
        st.audio(audio_file, format=f"audio/{Path(audio_file.name).suffix.lstrip('.')}")

        if st.button("📝 Transcribe", type="primary", use_container_width=True):
            suffix = Path(audio_file.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            try:
                with st.spinner("Transcribing… (first run downloads Whisper weights)"):
                    stt = _get_stt()
                    result = stt.transcribe_file(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            st.success("Transcription complete!")
            st.markdown("**Transcript:**")
            st.text_area("", value=result.text, height=140, key="stt_result")

            col1, col2, col3 = st.columns(3)
            col1.metric("Language detected", result.language)
            col2.metric("Audio duration", f"{result.duration_seconds:.1f}s")
            col3.metric("Segments", len(result.segments))

            if result.text.strip():
                st.markdown("---")
                st.markdown("**Send transcription to Chat?**")
                if st.button("💬 Ask this question in Chat tab"):
                    st.session_state.chat_history.append(
                        {"role": "user", "content": result.text}
                    )
                    st.info("Added to chat history — switch to the Chat tab to see the response.")

    st.divider()
    # Try live recorder widget
    st.markdown("**Or record from microphone (browser)**")
    try:
        from audiorecorder import audiorecorder
        audio = audiorecorder("🔴 Start recording", "⏹ Stop recording", key="mic_record")
        if len(audio) > 0:
            st.audio(audio.export().read(), format="audio/wav")
            if st.button("📝 Transcribe recording", type="primary"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    audio.export(tmp.name, format="wav")
                    tmp_path = tmp.name
                try:
                    with st.spinner("Transcribing…"):
                        result = _get_stt().transcribe_file(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                st.success("Done!")
                st.text_area("Transcript", value=result.text, height=120, key="mic_result")
    except ImportError:
        st.caption("Install `streamlit-audiorecorder` for live microphone recording.")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Text to Speech
# ═════════════════════════════════════════════════════════════════════════════

def _tab_tts():
    st.subheader("🔊 Text-to-Speech (Edge TTS)")
    st.caption("Type any text and listen to the synthesised speech via Microsoft Edge TTS.")

    settings = _get_settings()

    col1, col2 = st.columns([3, 1])
    with col1:
        tts_text = st.text_area(
            "Text to synthesise",
            height=160,
            key="tts_input",
            placeholder="Type or paste the text you want to hear…",
            value="Hello! I am the Agentic RAG assistant. I can search your documents and answer your questions.",
        )
    with col2:
        st.markdown("**Voice settings**")
        voice = st.text_input("Voice", value=settings.tts_voice, key="tts_voice")
        rate = st.select_slider("Rate", options=["-20%", "-10%", "+0%", "+10%", "+20%"], value="+0%")
        volume = st.select_slider("Volume", options=["-20%", "-10%", "+0%", "+10%", "+20%"], value="+0%")

    if st.button("🔊 Synthesise", type="primary", use_container_width=True) and tts_text.strip():
        with st.spinner("Synthesising audio…"):
            # Override settings inline for this call
            from agentic_rag.config.settings import Settings
            temp_settings = Settings(
                tts_voice=voice,
                tts_rate=rate,
                tts_volume=volume,
            )
            from agentic_rag.tts.text_to_speech import TextToSpeech
            tts = TextToSpeech(temp_settings)
            out_path = tempfile.mktemp(suffix=".mp3")
            try:
                tts.synthesize(tts_text, output_path=out_path)
                audio_bytes = Path(out_path).read_bytes()
            finally:
                Path(out_path).unlink(missing_ok=True)

        st.success("Synthesis complete — play below:")
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            "⬇️ Download MP3",
            data=audio_bytes,
            file_name="tts_output.mp3",
            mime="audio/mp3",
        )


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 5 — Status / Config
# ═════════════════════════════════════════════════════════════════════════════

def _tab_status():
    st.subheader("⚙️ System Status & Configuration")

    settings = _get_settings()
    llm = _get_llm()
    retriever = _get_retriever()

    reachable = llm.health_check()
    status_color = "🟢" if reachable else "🔴"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM Server", f"{status_color} {'OK' if reachable else 'DOWN'}")
    col2.metric("Indexed Chunks", retriever.document_count())
    col3.metric("Whisper Model", settings.whisper_model)
    col4.metric("TTS Voice", settings.tts_voice.replace("Neural", ""))

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🧠 LLM")
        st.json({
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "timeout_s": settings.llm_timeout,
            "reachable": reachable,
        })

        st.markdown("### 📚 RAG")
        st.json({
            "vector_store": settings.vector_store_path,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.retrieval_top_k,
            "similarity_threshold": settings.similarity_threshold,
        })

    with col_b:
        st.markdown("### 🎙️ STT")
        st.json({
            "whisper_model": settings.whisper_model,
            "device": settings.whisper_device,
            "compute_type": settings.whisper_compute_type,
            "sample_rate_hz": settings.audio_sample_rate,
            "silence_duration_s": settings.audio_silence_duration,
            "max_duration_s": settings.audio_max_duration,
        })

        st.markdown("### 🔊 TTS")
        st.json({
            "voice": settings.tts_voice,
            "rate": settings.tts_rate,
            "volume": settings.tts_volume,
        })

    st.divider()

    # Live LLM test
    st.markdown("### 🔬 Live LLM Test")
    test_prompt = st.text_input(
        "Test prompt",
        value="Say hello in exactly five words.",
        key="live_test_prompt",
    )
    if st.button("▶️ Send test query", type="primary"):
        if not reachable:
            st.error("LLM server is not reachable.")
        else:
            with st.spinner("Waiting for LLM response…"):
                from agentic_rag.llm.client import LLMClientError
                try:
                    t0 = time.perf_counter()
                    resp = llm.chat(test_prompt, system_prompt="You are a concise assistant.")
                    elapsed = time.perf_counter() - t0
                    st.success(f"Response ({elapsed:.2f}s):")
                    st.info(resp.content)
                    if resp.usage:
                        st.caption(f"Tokens: {resp.usage}")
                except LLMClientError as e:
                    st.error(str(e))


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    _init_state()
    _sidebar()

    st.title("🤖 Agentic RAG · STT & TTS")
    st.caption(
        "Retrieval-Augmented Generation with Speech-to-Text and Text-to-Speech "
        "powered by QWEN 3.5 4B running locally on your LAN."
    )

    tab_chat, tab_ingest, tab_stt, tab_tts, tab_status = st.tabs([
        "💬 Chat",
        "📥 Ingest Docs",
        "🎙️ Speech → Text",
        "🔊 Text → Speech",
        "⚙️ Status",
    ])

    with tab_chat:
        _tab_chat()

    with tab_ingest:
        _tab_ingest()

    with tab_stt:
        _tab_stt()

    with tab_tts:
        _tab_tts()

    with tab_status:
        _tab_status()


if __name__ == "__main__":
    main()
