"""
VidMind — Personalized, code-aware video notes.

Run: streamlit run app.py
"""

import core.bootstrap  # noqa: F401 — env vars before ML imports

import json
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.feedback import submit_feedback
from core.pipeline import run_pipeline
from core.primary_reference import primary_reference_status
from core.rag import ensure_indexed, index_knowledge_collection, index_style_collection

st.set_page_config(page_title="VidMind", layout="wide")

st.title("VidMind")
st.caption("YouTube / video → transcript → personalized software engineering study notes")

with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Transcription language", ["english", "hinglish"])
    capture_code = st.checkbox("Capture on-screen code", value=True)
    use_course_planner = st.checkbox("Course planner (long videos)", value=True)
    enforce_domain = st.checkbox("Software engineering only", value=True)
    frame_interval = st.slider(
        "Frame interval (seconds)", 15, 120,
        int(os.getenv("FRAME_INTERVAL_SECONDS", "45")), 15,
    )
    os.environ["FRAME_INTERVAL_SECONDS"] = str(frame_interval)

    st.divider()
    ref = primary_reference_status()
    st.subheader("Primary reference")
    if ref["ready"]:
        st.success(f"`{ref['name']}` — {ref['chunk_count']} chunks loaded")
    else:
        st.warning(f"`{ref['name']}` not extracted yet")
        st.code(f"python scripts/digitize_pdf.py {ref['pdf']}", language="bash")

    st.divider()
    if os.getenv("MISTRAL_API_KEY"):
        st.success("LLM: Mistral API")
    else:
        st.info(f"LLM: Ollama (`{os.getenv('OLLAMA_MODEL', 'mistral')}`)")

    if st.button("Re-index style DB"):
        with st.spinner("Indexing style collection..."):
            st.success(f"Added {index_style_collection()} doc(s)")

    if st.button("Re-index knowledge DB"):
        with st.spinner("Indexing knowledge collection..."):
            st.success(f"Added {index_knowledge_collection()} doc(s)")

source = st.text_input(
    "YouTube URL or local file path",
    placeholder="https://www.youtube.com/watch?v=...",
)

col1, col2 = st.columns(2)
run_btn = col1.button("Generate notes", type="primary", use_container_width=True)
index_btn = col2.button("Index all samples", use_container_width=True)

if index_btn:
    with st.spinner("Indexing style + knowledge collections..."):
        ensure_indexed()
        s = index_style_collection()
        k = index_knowledge_collection()
        st.success(f"Indexed style={s}, knowledge={k} new doc(s)")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if run_btn:
    if not source.strip():
        st.error("Enter a YouTube URL or local file path.")
    else:
        progress = st.progress(0, text="Starting...")
        try:
            ensure_indexed()
            progress.progress(20, text="Transcribing...")
            result = run_pipeline(
                source.strip(),
                language=language,
                capture_code=capture_code,
                use_course_planner=use_course_planner,
                enforce_domain=enforce_domain,
            )
            st.session_state.last_result = result
            progress.progress(100, text="Done!")
            st.success(f"**{result.title}**")
            if result.modules:
                st.info("Modules: " + " | ".join(result.modules))
        except Exception as exc:
            progress.empty()
            st.error(str(exc))
            st.exception(exc)

result = st.session_state.last_result
if result:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Notes", "Summary", "Transcript", "Downloads", "Feedback"]
    )

    with tab1:
        st.markdown(result.notes_md)

    with tab2:
        st.markdown(result.summary)

    with tab3:
        st.text_area("Transcript", result.transcript, height=400)

    with tab4:
        for label, key, mime in [
            ("Markdown", "md", "text/markdown"),
            ("PDF", "pdf", "application/pdf"),
            ("JSON", "json", "application/json"),
        ]:
            path = result.note_paths.get(key)
            if path and os.path.isfile(path):
                with open(path, "rb") as f:
                    st.download_button(f"Download {label}", f, os.path.basename(path), mime)

        if result.notes_doc:
            with st.expander("notes.json preview"):
                st.json(json.loads(json.dumps(result.notes_doc.model_dump())))

    with tab5:
        st.markdown(
            "Edit the notes below to match your style, then submit. "
            "Future generations will retrieve from the knowledge DB."
        )
        edited = st.text_area(
            "Your edited notes",
            value=result.notes_md,
            height=400,
            key="feedback_editor",
        )
        if st.button("Submit feedback & re-index"):
            path = submit_feedback(edited, result.title, source_path=result.note_paths.get("md"))
            st.success(f"Saved and indexed: `{path}`")

st.divider()
with st.expander("Setup (free on M4)"):
    st.markdown("""
1. `brew install ollama ffmpeg tesseract && ollama pull mistral`
2. Optional VLM: `ollama pull moondream`
3. Scan notes → `python scripts/digitize_notes.py style_samples/scans/*.jpg`
4. Style guide → `python scripts/extract_style_guide.py`
5. Feedback CLI → `python scripts/submit_feedback.py my_edited_notes.md --in-place`
    """)
