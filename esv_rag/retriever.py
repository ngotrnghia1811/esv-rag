"""
Retrieval backend for ESV-RAG.

Wraps FlashRAG dense retrievers (E5-base, BGE) and BM25 under a single interface.
The RetrieverClient is used by RetrievalAugmented* variants of the ESV actions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import RetrieverConfig

logger = logging.getLogger(__name__)


class RetrieverClient:
    """
    Unified retrieval client.

    Parameters
    ----------
    config : RetrieverConfig
    """

    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.config = config or RetrieverConfig()
        self._retriever = None
        self._cache: Dict[str, List[str]] = {}
        self._cache_path = Path(self.config.cache_path)
        self._load_cache()
        self._build_retriever()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _build_retriever(self) -> None:
        method = self.config.method.lower()
        if method == "bm25":
            self._retriever = self._build_bm25()
        elif method in ("e5base", "e5"):
            self._retriever = self._build_dense("intfloat/e5-base-v2")
        elif method in ("bge", "bge-base"):
            self._retriever = self._build_dense("BAAI/bge-base-en-v1.5")
        else:
            logger.warning("Unknown retriever method '%s', using BM25 fallback", method)
            self._retriever = self._build_bm25()

    def _build_bm25(self):
        try:
            import bm25s
            corpus_path = self.config.corpus_path
            if not corpus_path:
                logger.warning("No corpus_path set for BM25; retrieval will return empty results")
                return None
            logger.info("Building BM25 index from %s", corpus_path)
            docs = self._load_corpus(corpus_path)
            tokenized = bm25s.tokenize([d["text"] for d in docs])
            idx = bm25s.BM25()
            idx.index(tokenized)
            return {"index": idx, "docs": docs, "backend": "bm25s"}
        except ImportError:
            logger.warning("bm25s not installed; retrieval unavailable")
            return None

    def _build_dense(self, model_path: str):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            import faiss

            corpus_path = self.config.corpus_path
            if not corpus_path:
                logger.warning("No corpus_path set; dense retrieval unavailable")
                return None

            index_path = Path(self.config.index_path) / f"{self.config.method}.index"
            docs = self._load_corpus(corpus_path)
            texts = [d["text"] for d in docs]

            model = SentenceTransformer(model_path or self.config.model_path)

            if index_path.exists():
                logger.info("Loading existing FAISS index from %s", index_path)
                idx = faiss.read_index(str(index_path))
            else:
                logger.info("Building FAISS index (%d passages)…", len(texts))
                embeddings = model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                )
                embeddings = embeddings.astype(np.float32)
                dim = embeddings.shape[1]
                idx = faiss.IndexFlatIP(dim)
                idx.add(embeddings)
                index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(idx, str(index_path))
                logger.info("Saved index to %s", index_path)

            return {"index": idx, "model": model, "docs": docs,
                    "backend": "dense", "model_path": model_path}
        except ImportError as exc:
            logger.warning("Dense retriever unavailable: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve the top-k passages for *query*.

        Returns a list of passage strings.
        """
        k = top_k or self.config.retrieval_topk

        if self.config.use_cache:
            cache_key = f"{query}||{k}"
            if cache_key in self._cache:
                return self._cache[cache_key]

        passages = self._retrieve_impl(query, k)

        if self.config.use_cache:
            self._cache[cache_key] = passages
            self._save_cache()

        return passages

    def retrieve_batch(self, queries: List[str], top_k: Optional[int] = None) -> List[List[str]]:
        """Retrieve passages for multiple queries."""
        return [self.retrieve(q, top_k) for q in queries]

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _retrieve_impl(self, query: str, k: int) -> List[str]:
        if self._retriever is None:
            logger.warning("No retriever available; returning empty results")
            return []

        backend = self._retriever.get("backend", "")
        try:
            if backend == "bm25s":
                return self._retrieve_bm25(query, k)
            elif backend == "dense":
                return self._retrieve_dense(query, k)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
        return []

    def _retrieve_bm25(self, query: str, k: int) -> List[str]:
        import bm25s
        idx   = self._retriever["index"]
        docs  = self._retriever["docs"]
        toks  = bm25s.tokenize([query])
        results, _ = idx.retrieve(toks, k=min(k, len(docs)))
        passages = []
        for hit_list in results:
            for hit in hit_list:
                doc = docs[hit]
                passages.append(doc.get("text", str(doc)))
        return passages[:k]

    def _retrieve_dense(self, query: str, k: int) -> List[str]:
        import numpy as np
        model = self._retriever["model"]
        idx   = self._retriever["index"]
        docs  = self._retriever["docs"]
        q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
        _, indices = idx.search(q_emb, k)
        return [docs[i].get("text", str(docs[i])) for i in indices[0] if i >= 0]

    # ------------------------------------------------------------------
    # Corpus / cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_corpus(path: str) -> List[Dict]:
        docs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    docs.append({"text": line})
        return docs

    def _load_cache(self) -> None:
        if self._cache_path.exists():
            try:
                with open(self._cache_path) as f:
                    self._cache = json.load(f)
                logger.debug("Loaded %d cached retrievals", len(self._cache))
            except Exception as exc:
                logger.warning("Could not load retrieval cache: %s", exc)

    def _save_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f)
        except Exception as exc:
            logger.warning("Could not save retrieval cache: %s", exc)
