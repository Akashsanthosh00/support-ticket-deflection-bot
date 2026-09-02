"""
Retrieval layer: hybrid (dense + BM25) search over FAQ docs + past resolved
tickets, followed by cross-encoder reranking.

Pipeline for a query:
  1. Dense leg   — embed the query, search Qdrant for the top HYBRID_CANDIDATE_K
                    nearest neighbors (cosine similarity).
  2. Sparse leg  — score the query against a BM25 index built over the same
                    corpus, take the top HYBRID_CANDIDATE_K by keyword overlap.
  3. Merge       — union the two candidate sets (dedup by item id).
  4. Rerank      — run a cross-encoder over (query, candidate_text) pairs and
                    sort by that score. This is what actually reaches the
                    abstention logic — cross-encoders are far better at
                    judging "does this text actually answer the question"
                    than embedding cosine similarity or BM25 alone.

The top (reranked) score is one of the key signals fed into the abstention
logic (app/abstention.py) — low similarity means "we don't have good
grounding for this question", which is a strong signal to escalate.
"""
import json
import math
import os
import pickle
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import settings


@dataclass
class RetrievedItem:
    source_id: str
    source_type: str  # 'faq' | 'ticket'
    title: str
    text: str
    score: float  # sigmoid-normalized cross-encoder rerank score, 0-1 (higher = more relevant)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class KnowledgeIndex:
    def __init__(self):
        self._embed_model: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None
        self._qdrant: QdrantClient | None = None
        self.bm25: BM25Okapi | None = None
        self.items: list[dict] = []  # parallel array; index == point id

        # Toggle for the ablation study (scripts/eval_ablation.py) — flip
        # this at runtime to compare hybrid+reranked retrieval against a
        # dense-only baseline with no cross-encoder step. Always True in
        # normal app usage.
        self.rerank_enabled: bool = True

    # --- lazy-loaded models (importing this module for unit tests should
    # never trigger a model download / GPU init) ---
    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._embed_model

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(settings.RERANKER_MODEL)
        return self._reranker

    @property
    def qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            if settings.QDRANT_URL:
                self._qdrant = QdrantClient(
                    url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None
                )
            else:
                os.makedirs(settings.QDRANT_PATH, exist_ok=True)
                self._qdrant = QdrantClient(path=settings.QDRANT_PATH)
        return self._qdrant

    # --- building ---
    def _load_items(self) -> list[dict]:
        items = []
        with open(settings.FAQ_PATH) as f:
            faqs = json.load(f)
        for faq in faqs:
            items.append(
                {
                    "source_id": faq["id"],
                    "source_type": "faq",
                    "title": faq["title"],
                    "text": f"{faq['title']}\n{faq['content']}",
                }
            )

        with open(settings.TICKETS_PATH) as f:
            tickets = json.load(f)
        for tkt in tickets:
            items.append(
                {
                    "source_id": tkt["id"],
                    "source_type": "ticket",
                    "title": tkt["subject"],
                    "text": f"{tkt['subject']}\n{tkt['question']}\n{tkt['resolution']}",
                }
            )
        return items

    def build(self):
        """Load FAQs + past tickets, embed them into Qdrant, and build the BM25 index."""
        self.items = self._load_items()
        texts = [it["text"] for it in self.items]

        # --- dense: Qdrant ---
        embeddings = self.embed_model.encode(texts, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        dim = embeddings.shape[1]

        if self.qdrant.collection_exists(settings.QDRANT_COLLECTION):
            self.qdrant.delete_collection(settings.QDRANT_COLLECTION)
        self.qdrant.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )
        self.qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=[
                qmodels.PointStruct(id=i, vector=embeddings[i].tolist(), payload={"idx": i})
                for i in range(len(self.items))
            ],
        )

        # --- sparse: BM25 ---
        tokenized_corpus = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        os.makedirs(settings.INDEX_DIR, exist_ok=True)
        with open(os.path.join(settings.INDEX_DIR, "items.json"), "w") as f:
            json.dump(self.items, f)
        with open(settings.BM25_INDEX_PATH, "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self) -> bool:
        """Load a previously built index from disk, if present."""
        items_path = os.path.join(settings.INDEX_DIR, "items.json")
        if not (os.path.exists(items_path) and os.path.exists(settings.BM25_INDEX_PATH)):
            return False
        if not self.qdrant.collection_exists(settings.QDRANT_COLLECTION):
            return False

        with open(items_path) as f:
            self.items = json.load(f)
        with open(settings.BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        return True

    # --- searching ---
    def search(self, query: str, top_k: int | None = None) -> list[RetrievedItem]:
        if self.bm25 is None or not self.items:
            raise RuntimeError("Index not built/loaded yet. Call build() or load() first.")
        top_k = top_k or settings.TOP_K_RESULTS
        candidate_k = max(settings.HYBRID_CANDIDATE_K, top_k)

        # --- dense leg (Qdrant) ---
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)[0]
        dense_response = self.qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vec.tolist(),
            limit=candidate_k,
        )
        dense_idxs = {hit.payload["idx"] for hit in dense_response.points}
        # Cosine similarity per candidate, from Qdrant directly — used as the
        # fallback score when reranking is disabled (see rerank_enabled).
        dense_score_map = {hit.payload["idx"]: hit.score for hit in dense_response.points}

        # --- sparse leg (BM25) ---
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        sparse_idxs = set(np.argsort(bm25_scores)[::-1][:candidate_k].tolist())

        # --- merge (union, dedup) ---
        candidate_idxs = dense_idxs | sparse_idxs
        if not candidate_idxs:
            return []

        candidates = [self.items[i] for i in candidate_idxs]

        if self.rerank_enabled:
            # --- rerank with cross-encoder ---
            pairs = [[query, c["text"]] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            scored = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
            scored = [(item, _sigmoid(float(s))) for item, s in scored]
        else:
            # --- ablation baseline: no reranking, just dense cosine similarity.
            # BM25-only candidates (not in the dense top-k) get no dense score
            # and rank last — without a reranker there's no principled way to
            # compare a dense-space score against a BM25 keyword score, which
            # is itself part of what a reranker buys you. ---
            scored = sorted(
                candidate_idxs, key=lambda i: dense_score_map.get(i, 0.0), reverse=True
            )
            scored = [
                (self.items[i], max(0.0, min(1.0, dense_score_map.get(i, 0.0))))
                for i in scored
            ]

        results = []
        for item, score in scored[:top_k]:
            results.append(
                RetrievedItem(
                    source_id=item["source_id"],
                    source_type=item["source_type"],
                    title=item["title"],
                    text=item["text"],
                    score=score,
                )
            )
        return results


# Singleton instance used by the FastAPI app
knowledge_index = KnowledgeIndex()


def ensure_index_ready():
    """Load the index from disk, or build it fresh if it doesn't exist yet."""
    if not knowledge_index.load():
        knowledge_index.build()