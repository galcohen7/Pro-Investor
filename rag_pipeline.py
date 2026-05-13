"""
Advanced RAG (Retrieval-Augmented Generation) pipeline.

Stages:
  1. Semantic Chunking  — splits documents at sentence/paragraph boundaries
  2. Embedding          — Sentence-Transformers (all-MiniLM-L6-v2)
  3. Vector Storage     — ChromaDB with cosine similarity index
  4. Retrieval          — top-K candidates via cosine similarity
  5. Reranking          — Cross-Encoder (ms-marco-MiniLM-L-6-v2) for precision
"""

import re
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "financial_knowledge"
CHROMA_DB_PATH = "./chroma_db"


class SemanticChunker:
    """
    Splits financial text into overlapping semantic chunks using
    paragraph and sentence boundaries (no fixed token count).
    """

    def __init__(self, max_chunk_size: int = 512, overlap: int = 80):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str = "unknown") -> List[dict]:
        """
        Returns a list of chunk dicts:
          {"id": str, "text": str, "source": str, "chunk_index": int}
        """
        paragraphs = re.split(r"\n{2,}", text.strip())
        chunks: List[dict] = []
        current_chunk = ""
        chunk_id = 0

        for para in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) + 1 <= self.max_chunk_size:
                    current_chunk = (current_chunk + " " + sentence).strip()
                else:
                    if current_chunk:
                        chunks.append(
                            {
                                "id": f"{source}_chunk_{chunk_id}",
                                "text": current_chunk,
                                "source": source,
                                "chunk_index": chunk_id,
                            }
                        )
                        chunk_id += 1
                        # Carry overlap into next chunk
                        overlap_text = current_chunk[-self.overlap :]
                        current_chunk = (overlap_text + " " + sentence).strip()
                    else:
                        current_chunk = sentence

        if current_chunk:
            chunks.append(
                {
                    "id": f"{source}_chunk_{chunk_id}",
                    "text": current_chunk,
                    "source": source,
                    "chunk_index": chunk_id,
                }
            )

        return chunks


class RAGPipeline:
    """
    Full RAG pipeline: ingest → embed → store → retrieve → rerank.
    Persists the vector DB to disk so knowledge survives restarts.
    """

    def __init__(self):
        self.chunker = SemanticChunker()
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)

        # Use persistent storage locally; fall back to in-memory on cloud (no disk access)
        try:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        except Exception:
            self.chroma_client = chromadb.EphemeralClient()
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_document(self, text: str, source: str = "manual_input") -> int:
        """
        Chunks text, embeds each chunk, and upserts into ChromaDB.
        Returns the number of chunks stored.
        """
        chunks = self.chunker.chunk(text, source)
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True).tolist()

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 10) -> List[dict]:
        """
        Retrieves up to top_k candidates from ChromaDB using cosine similarity.
        Returns list of dicts with 'text', 'source', 'similarity_score'.
        """
        total_docs = self.collection.count()
        if total_docs == 0:
            return []

        query_embedding = self.embedder.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, total_docs),
        )

        retrieved = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                retrieved.append(
                    {
                        "text": doc,
                        "source": meta.get("source", "unknown"),
                        "similarity_score": round(1.0 - dist, 4),
                    }
                )
        return retrieved

    # ── Reranking ──────────────────────────────────────────────────────────────

    def rerank(self, query: str, candidates: List[dict], top_k: int = 3) -> List[dict]:
        """
        Reranks retrieval candidates with a Cross-Encoder for precision.
        Cross-encoders attend jointly to (query, passage) unlike bi-encoders.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve_and_rerank(
        self, query: str, top_k_retrieve: int = 10, top_k_rerank: int = 3
    ) -> List[dict]:
        """Convenience method: retrieve then rerank in one call."""
        candidates = self.retrieve(query, top_k=top_k_retrieve)
        return self.rerank(query, candidates, top_k=top_k_rerank)

    def get_context_string(self, query: str) -> str:
        """
        Returns a formatted context string ready for injection into an LLM prompt.
        """
        results = self.retrieve_and_rerank(query)
        if not results:
            return "No relevant financial context found in the knowledge base."

        parts = [f"[Source {i}: {r['source']}]\n{r['text']}" for i, r in enumerate(results, 1)]
        return "\n\n---\n\n".join(parts)

    def get_document_count(self) -> int:
        """Returns the total number of chunks currently stored."""
        return self.collection.count()
