"""
Deep Researcher Agent with Local Embeddings
Single-file Python implementation
"""

import os
import argparse
import json
import sqlite3
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import pathlib
import subprocess

# ----------------------------- Safe Imports -------------------------------
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("❌ sentence-transformers not installed. Run: pip install sentence-transformers")

try:
    import faiss
except ImportError:
    raise ImportError("❌ faiss not installed. Run: pip install faiss-cpu OR pip install faiss-gpu")

try:
    from transformers import pipeline
except ImportError:
    raise ImportError("❌ transformers not installed. Run: pip install transformers")

try:
    import tiktoken
except ImportError:
    tiktoken = None  # optional, for token-aware chunking

# ----------------------------- Utilities ---------------------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def chunk_text(text: str, max_tokens: int = 200, tokenizer=None) -> List[str]:
    if tokenizer is None:
        words = text.split()
        return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]
    else:
        tokens = tokenizer.encode(text)
        return [tokenizer.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]

# --------------------------- Embedding Model ------------------------------
class LocalEmbedder:
    def __init__(self, model_dir_or_name: str = "all-MiniLM-L6-v2"):
        """
        Load local folder if exists, otherwise auto-download from HuggingFace.
        """
        try:
            # Force download if folder broken
            self.model = SentenceTransformer(model_dir_or_name, cache_folder=os.path.expanduser("~/.cache/huggingface"))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SentenceTransformer '{model_dir_or_name}': {e}.\n"
                f"Run: pip install sentence-transformers"
            )

    def embed_texts(self, texts):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

# --------------------------- Document Store -------------------------------
class DocumentStore:
    def __init__(self, path_prefix: str = "dr_index", dim: int = 384):
        os.makedirs(path_prefix, exist_ok=True)
        self.index_path = os.path.join(path_prefix, "faiss.index")
        self.meta_db = os.path.join(path_prefix, "meta.db")
        self.dim = dim
        self._init_sqlite()
        self._init_faiss()

    def _init_sqlite(self):
        self.conn = sqlite3.connect(self.meta_db)
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source TEXT,
                text TEXT,
                metadata TEXT
            )""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS id_map (
                idx INTEGER PRIMARY KEY,
                doc_id TEXT
            )""")
        self.conn.commit()

    def _init_faiss(self):
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception:
                print("⚠️ Warning: failed to load existing FAISS index, creating new one")

    def save(self):
        faiss.write_index(self.index, self.index_path)
        self.conn.commit()

    def add_documents(self, docs: List[Dict[str, Any]], vectors):
        import numpy as np
        ids = []
        cur = self.conn.cursor()
        for i, d in enumerate(docs):
            idx_val = int(int(d["id"][:15], 16) % (2**31 - 1))
            ids.append(idx_val)
            cur.execute(
                "INSERT OR REPLACE INTO documents (id, source, text, metadata) VALUES (?, ?, ?, ?)",
                (d["id"], d.get("source", ""), d.get("text", ""), json.dumps(d.get("metadata", {}))),
            )
            cur.execute("INSERT OR REPLACE INTO id_map (idx, doc_id) VALUES (?, ?)", (idx_val, d["id"]))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-9)
        self.index.add_with_ids(vectors.astype('float32'), np.array(ids, dtype='int64'))
        self.save()

    def search(self, vector, k=5) -> List[Tuple[float, Dict[str, Any]]]:
        import numpy as np
        v = vector.astype('float32')
        v = v / (np.linalg.norm(v) + 1e-9)
        D, I = self.index.search(v.reshape(1, -1), k)
        results = []
        cur = self.conn.cursor()
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            cur.execute("SELECT doc_id FROM id_map WHERE idx = ?", (int(idx),))
            row = cur.fetchone()
            if not row:
                continue
            doc_id = row[0]
            cur.execute("SELECT source, text, metadata FROM documents WHERE id = ?", (doc_id,))
            r = cur.fetchone()
            if not r:
                continue
            source, text, metadata = r
            results.append((float(score), {
                "id": doc_id, "source": source, "text": text, "metadata": json.loads(metadata)
            }))
        return results

# --------------------------- Ingestor -------------------------------------
class Ingestor:
    def __init__(self, embedder: LocalEmbedder, store: DocumentStore, tokenizer=None):
        self.embedder = embedder
        self.store = store
        self.tokenizer = tokenizer

    def read_file(self, path: str) -> str:
        p = pathlib.Path(path)
        suffix = p.suffix.lower()
        if suffix in ['.txt', '.md', '.html']:
            return p.read_text(encoding='utf-8', errors='ignore')
        elif suffix == '.pdf':
            try:
                import pdfplumber
                text = []
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        text.append(page.extract_text() or '')
                return '\n'.join(text)
            except Exception:
                raise RuntimeError("Install pdfplumber to ingest PDFs: pip install pdfplumber")
        return p.read_text(encoding='utf-8', errors='ignore')

    def ingest(self, path: str, chunk_size_tokens: int = 200, metadata: Dict[str, Any] = None):
        text = self.read_file(path)
        chunks = chunk_text(text, max_tokens=chunk_size_tokens, tokenizer=self.tokenizer)
        docs = []
        for i, c in enumerate(chunks):
            doc_id = sha1(path + str(i))
            docs.append({
                "id": doc_id,
                "source": path,
                "text": c,
                "metadata": {**(metadata or {}), "chunk": i}
            })
        vectors = self.embedder.embed_texts([d["text"] for d in docs])
        self.store.add_documents(docs, vectors)
        print(f"✅ Ingested {len(docs)} chunks from {path}")

    def ingest_folder(self, folder: str, recursive: bool = True, **kwargs):
        for root, _, files in os.walk(folder):
            for f in files:
                if f.startswith('.'): continue
                full = os.path.join(root, f)
                try:
                    self.ingest(full, **kwargs)
                except Exception as e:
                    print(f"⚠️ Skipped {full}: {e}")
            if not recursive: break

# ------------------------- Retriever & Reasoner ----------------------------
class Retriever:
    def __init__(self, embedder: LocalEmbedder, store: DocumentStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int = 5):
        vec = self.embedder.embed_texts([query])[0]
        return self.store.search(vec, k=k)

class Reasoner:
    def __init__(self, retriever: Retriever, synthesizer: 'Synthesizer'):
        self.retriever = retriever
        self.synthesizer = synthesizer

    def decompose(self, query: str) -> List[str]:
        if len(query.split()) < 10:
            return [query]
        separators = [' and ', ';', ',', ' or ']
        parts = [query]
        for s in separators:
            new = []
            for p in parts:
                new.extend([x.strip() for x in p.split(s) if x.strip()])
            parts = new
        uniq = []
        for p in parts:
            if p.lower() not in [u.lower() for u in uniq] and len(p.split()) >= 3:
                uniq.append(p)
        return uniq or [query]

    def answer(self, query: str, k: int = 5, explain: bool = True):
        subqueries = self.decompose(query)
        steps, combined = [], []
        for sq in subqueries:
            res = self.retriever.retrieve(sq, k=k)
            texts = [r[1]["text"] for r in res]
            synthesized = self.synthesizer.summarize(texts, prompt=sq)
            steps.append({"subquery": sq, "evidence": [r[1] for r in res], "synthesized": synthesized})
            combined.extend(texts)
        final = self.synthesizer.summarize(combined, prompt=query)
        out = {"query": query, "substeps": steps, "final": final}
        if explain:
            out["reasoning"] = {"decomposition": subqueries, "strategy": "decompose → retrieve → synthesize"}
        return out

# ------------------------- Synthesizer ------------------------------------
class Synthesizer:
    def __init__(self, model_name: Optional[str] = None, device: str = 'cpu'):
        self.pipeline = None
        if model_name:
            try:
                self.pipeline = pipeline("summarization", model=model_name,
                                         device=0 if device == 'cuda' else -1)
            except Exception as e:
                print(f"⚠️ Summarizer load failed ({e}), falling back to extractive summarizer.")

    def summarize(self, texts: List[str], prompt: Optional[str] = None, max_length: int = 256) -> str:
        joined = "\n\n".join(texts)
        if not joined.strip(): return ""
        if self.pipeline:
            try:
                input_text = (prompt + '\n\n' + joined) if prompt else joined
                if len(input_text) > 10000: input_text = input_text[:10000]
                out = self.pipeline(input_text, max_length=max_length, truncation=True)
                return out[0]["summary_text"]
            except Exception: pass
        # extractive fallback
        sentences = [s.strip() for t in texts for s in t.split(". ") if s.strip()]
        keywords = [w.lower() for w in (prompt.split() if prompt else []) if len(w) > 3][:10]
        def score(s): return len(s.split()) + sum(10 for k in keywords if k in s.lower())
        sentences = sorted(sentences, key=score, reverse=True)
        out, count = [], 0
        for s in sentences:
            out.append(s)
            count += len(s.split())
            if count > 180: break
        return ". ".join(out)

# ------------------------- Export Utilities -------------------------------
def export_markdown(output_path: str, result: Dict[str, Any]):
    md = [f"# Research Report\n", f"**Query:** {result['query']}\n"]
    if "reasoning" in result:
        md.append("## Reasoning\n")
        md.append(f"Decomposition: {result['reasoning'].get('decomposition')}\n")
        md.append(f"Strategy: {result['reasoning'].get('strategy')}\n")
    md.append("## Sub-answers\n")
    for s in result["substeps"]:
        md.append(f"### {s['subquery']}\n")
        md.append(f"**Synthesized:** {s['synthesized']}\n")
        md.append("**Evidence:**\n")
        for e in s["evidence"]:
            md.append(f"- Source: {e.get('source')} (chunk {e.get('metadata', {}).get('chunk')})\n  \n  {e.get('text')[:500]}\n")
    md.append("## Final Answer\n")
    md.append(result["final"])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return output_path

def export_pdf(markdown_path: str, pdf_path: str):
    subprocess.run(["pandoc", markdown_path, "-o", pdf_path], check=True)
    return pdf_path

# ------------------------- CLI --------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ingest', type=str, help='file or folder to ingest')
    parser.add_argument('--index', type=str, default='dr_index', help='index prefix folder')
    parser.add_argument('--query', type=str, help='query to run')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--summarizer', type=str, default=None, help='local summarization model name')
    parser.add_argument('--export-md', type=str, help='path to write Markdown report')
    parser.add_argument('--export-pdf', type=str, help='path to write PDF report (requires pandoc)')
    args = parser.parse_args()

    embedder = LocalEmbedder()
    dim = embedder.embed_texts(["hello"]).shape[1]
    store = DocumentStore(path_prefix=args.index, dim=dim)
    synth = Synthesizer(model_name=args.summarizer)
    retriever = Retriever(embedder, store)
    reasoner = Reasoner(retriever, synth)
    ingestor = Ingestor(embedder, store)

    if args.ingest:
        if os.path.isdir(args.ingest): 
            ingestor.ingest_folder(args.ingest)
        else: 
            ingestor.ingest(args.ingest)
        print("✅ Ingestion complete.")

    if args.query:
        out = reasoner.answer(args.query, k=args.k, explain=True)
        print("\n--- FINAL ANSWER ---\n")
        print(out['final'])
        if args.export_md:
            export_markdown(args.export_md, out)
        if args.export_pdf and args.export_md:
            export_pdf(args.export_md, args.export_pdf)

if __name__ == "__main__":
    main()
