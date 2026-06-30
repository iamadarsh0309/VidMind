"""Split ChromaDB RAG: style collection vs knowledge collection."""

# Must be set before google.protobuf / chromadb are imported (Python 3.13 + protobuf 4+).
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import glob

from core.primary_reference import (
    is_primary_source,
    load_primary_chunks,
    primary_extracted_dir,
    primary_reference_name,
)

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
STYLE_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "style_samples")
EXTRACTED_DIR = os.path.join(STYLE_SAMPLES_DIR, "extracted")
NOTES_DIR = os.path.join(os.path.dirname(__file__), "..", "notes")
STYLE_DIR = os.path.join(os.path.dirname(__file__), "..", "style")

STYLE_COLLECTION = "note_style"
KNOWLEDGE_COLLECTION = "note_knowledge"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_stores: dict[str, object] = {}


def _get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _load_documents_from_dir(directory: str, pattern: str = "*.md") -> list[dict]:
    if not os.path.isdir(directory):
        return []
    docs = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        name = os.path.basename(path)
        if name.lower() == "readme.md":
            continue
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            docs.append({"content": content, "source": os.path.abspath(path)})
    return docs


def _load_extracted_pdf_chunks() -> list[dict]:
    if not os.path.isdir(EXTRACTED_DIR):
        return []
    docs = []
    for path in sorted(glob.glob(os.path.join(EXTRACTED_DIR, "**", "*.md"), recursive=True)):
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            docs.append({
                "content": content,
                "source": os.path.abspath(path),
                "primary": is_primary_source(path),
            })
    return docs


def _load_primary_docs() -> list[dict]:
    return [
        {**c, "primary": True, "reference": primary_reference_name()}
        for c in load_primary_chunks()
    ]


def _get_store(collection_name: str):
    if collection_name in _stores:
        return _stores[collection_name]

    from langchain_chroma import Chroma

    os.makedirs(CHROMA_DIR, exist_ok=True)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    _stores[collection_name] = store
    return store


def _existing_sources(store) -> set[str]:
    try:
        data = store._collection.get(include=["metadatas"])
        return {m.get("source", "") for m in (data.get("metadatas") or [])}
    except Exception:
        return set()


def _index_docs(store, docs: list[dict]) -> int:
    from langchain_core.documents import Document

    if not docs:
        return 0
    existing = _existing_sources(store)
    new_docs = [d for d in docs if d["source"] not in existing]
    if not new_docs:
        return 0
    lc_docs = [
        Document(
            page_content=d["content"],
            metadata={
                "source": d["source"],
                "primary": d.get("primary", False),
                "reference": d.get("reference", ""),
            },
        )
        for d in new_docs
    ]
    store.add_documents(lc_docs)
    return len(lc_docs)


def index_style_collection() -> int:
    """
    Style DB: style guide, few-shot examples, AND primary reference chunks
    (canonical format/structure).
    """
    docs = _load_documents_from_dir(STYLE_DIR)
    docs += [
        d for d in _load_documents_from_dir(STYLE_SAMPLES_DIR)
        if "fewshot_" in os.path.basename(d["source"])
    ]
    docs += _load_primary_docs()
    store = _get_store(STYLE_COLLECTION)
    count = _index_docs(store, docs)
    if count:
        print(f"Indexed {count} document(s) into style collection")
    return count


def index_knowledge_collection() -> int:
    """
    Knowledge DB: topic content including primary reference (highest priority source).
    """
    docs = [
        d for d in _load_documents_from_dir(STYLE_SAMPLES_DIR)
        if "fewshot_" not in os.path.basename(d["source"])
        and not d["source"].endswith(".pdf")
    ]
    docs += _load_documents_from_dir(NOTES_DIR)
    docs += _load_extracted_pdf_chunks()
    store = _get_store(KNOWLEDGE_COLLECTION)
    count = _index_docs(store, docs)
    if count:
        print(f"Indexed {count} document(s) into knowledge collection")
    return count


def index_style_samples() -> int:
    return index_style_collection() + index_knowledge_collection()


def index_approved_note(path: str) -> int:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return 0

    from langchain_core.documents import Document

    doc = Document(page_content=content, metadata={"source": path, "approved": True})
    knowledge = _get_store(KNOWLEDGE_COLLECTION)
    knowledge.add_documents([doc])
    return 1


def ensure_indexed() -> None:
    try:
        style_store = _get_store(STYLE_COLLECTION)
        knowledge_store = _get_store(KNOWLEDGE_COLLECTION)
        if style_store._collection.count() == 0:
            index_style_collection()
        if knowledge_store._collection.count() == 0:
            index_knowledge_collection()
    except Exception as exc:
        print(f"ChromaDB indexing skipped ({exc}). Using disk-based primary reference.")


def _retrieve(collection: str, query: str, k: int, header: str) -> str:
    try:
        store = _get_store(collection)
        if store._collection.count() == 0:
            return ""
        results = store.similarity_search(query, k=k)
    except Exception:
        return ""

    if not results:
        return ""

    parts = []
    for i, doc in enumerate(results, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        tag = " [PRIMARY]" if doc.metadata.get("primary") else ""
        excerpt = doc.page_content[:1500]
        parts.append(f"### Reference {i}{tag} ({source})\n\n{excerpt}")

    return header + "\n\n" + "\n\n---\n\n".join(parts)


def retrieve_style_references(query: str, k: int = 2) -> str:
    ensure_indexed()
    chroma = _retrieve(
        STYLE_COLLECTION,
        query,
        k,
        "## Style references (match structure, tone, and formatting)",
    )
    if chroma:
        return chroma
    from core.primary_reference import retrieve_primary_references
    return retrieve_primary_references(query, k=k)


def retrieve_knowledge_references(query: str, k: int = 2) -> str:
    from core.primary_reference import retrieve_primary_references

    primary = retrieve_primary_references(query, k=k)
    ensure_indexed()
    chroma = _retrieve(
        KNOWLEDGE_COLLECTION,
        query,
        k,
        "## Additional topic references",
    )
    if primary and chroma:
        return primary + "\n\n" + chroma
    return primary or chroma
