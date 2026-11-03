"""RAG: extraction, chunking, embeddings, cosine retrieval with source attribution."""
import hashlib, io, json, math, re
from typing import Protocol
import numpy as np
from sqlalchemy.orm import Session
from ..config.settings import settings
from ..database.models import Chunk, Document


class Embedder(Protocol):
    name: str
    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Free, local, dependency-light. Lexical (unigram+bigram hashing), NOT semantic."""
    name = "hashing-768"
    dim = 768

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            toks = re.findall(r"[a-z0-9]+", t.lower())
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            counts: dict[str, int] = {}
            for f in feats:
                counts[f] = counts.get(f, 0) + 1
            for f, c in counts.items():
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                out[r, h % self.dim] += (1 if (h >> 64) & 1 else -1) * (1 + math.log(c))
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class APIEmbedder:
    """OpenAI-compatible /embeddings endpoint (e.g. Gemini text-embedding-004)."""
    def __init__(self, client=None, model: str | None = None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        self._c, self._m = client, model or settings.embedding_model
        self.name = f"api:{self._m}"

    def embed(self, texts):
        vecs = []
        for i in range(0, len(texts), 64):
            r = self._c.embeddings.create(model=self._m, input=texts[i:i + 64])
            vecs += [d.embedding for d in r.data]
        a = np.array(vecs, dtype=np.float32)
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.where(n == 0, 1, n)


def get_embedder() -> Embedder:
    if settings.embedding_model and settings.llm_api_key:
        return APIEmbedder()
    return HashingEmbedder()


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(data)).pages)
    else:
        text = data.decode("utf-8", errors="ignore")
    if not text.strip():
        raise ValueError("No extractable text (scanned PDFs need OCR, which is not supported)")
    return text


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    chunks, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):  # prefer to break at a sentence/paragraph boundary
            cut = max(text.rfind("\n", i, end), text.rfind(". ", i, end))
            end = cut + 1 if cut > i + size // 2 else end
        chunks.append(text[i:end].strip())
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return [c for c in chunks if c]


def ingest_document(db: Session, project_id: int, filename: str, data: bytes, embedder: Embedder) -> Document:
    chunks = chunk_text(extract_text(filename, data))
    vecs = embedder.embed(chunks)
    doc = Document(project_id=project_id, filename=filename, n_chunks=len(chunks))
    db.add(doc); db.flush()
    for i, (c, v) in enumerate(zip(chunks, vecs)):
        db.add(Chunk(document_id=doc.id, project_id=project_id, idx=i, text=c,
                     embedder=embedder.name, embedding_json=json.dumps(v.tolist())))
    db.commit()
    return doc


def retrieve(db: Session, project_id: int, query: str, embedder: Embedder, k: int = 5) -> list[dict]:
    rows = db.query(Chunk, Document.filename).join(Document, Document.id == Chunk.document_id).filter(
        Chunk.project_id == project_id, Chunk.embedder == embedder.name).all()
    if not rows:
        return []
    mat = np.array([json.loads(c.embedding_json) for c, _ in rows], dtype=np.float32)
    scores = mat @ embedder.embed([query])[0]
    order = np.argsort(-scores)[:k]
    return [{"text": rows[i][0].text, "source": rows[i][1], "chunk": rows[i][0].idx,
             "score": round(float(scores[i]), 4)} for i in order if scores[i] > 0.05]
