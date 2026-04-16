"""
Two-tier retriever:
  Tier 1 (Hot): ChromaDB semantic search over MCM-structured units
  Tier 2 (Cold): Full-context fallback to raw EventLog when confidence is low

This closes mem0's accuracy gap: we never throw away the raw log,
so we can always fall back to full-context retrieval when structured memory
doesn't have a high-confidence answer.
"""

import logging
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from ..data.event_log import EventLog
from ..data.memory_store import MemoryStore

logger = logging.getLogger(__name__)

# Confidence threshold below which we fall back to raw event log
FALLBACK_THRESHOLD = 0.65


class VectorIndex:
    """Thin wrapper around ChromaDB for per-user memory unit indexing."""

    def __init__(self, persist_dir: str = ".chromadb"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def _collection(self, user_id: str):
        return self.client.get_or_create_collection(
            name=f"user_{user_id.replace('-', '_')}",  # ChromaDB name constraints
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, unit_id: str, user_id: str, text: str, metadata: dict) -> None:
        # Enforce metadata field size — fix for mem0's metadata length failure
        safe_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, list):
                safe_metadata[k] = ",".join(str(x) for x in v)[:500]
            elif isinstance(v, str):
                safe_metadata[k] = v[:500]
            else:
                safe_metadata[k] = v

        coll = self._collection(user_id)
        coll.upsert(ids=[unit_id], documents=[text], metadatas=[safe_metadata])

    def delete(self, unit_ids: list[str], user_id: str) -> None:
        coll = self._collection(user_id)
        coll.delete(ids=unit_ids)

    def search(
        self,
        query: str,
        user_id: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        coll = self._collection(user_id)
        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": min(n_results, coll.count() or 1)}
        if where:
            kwargs["where"] = where

        results = coll.query(**kwargs)
        if not results["ids"][0]:
            return []

        return [
            {
                "id": uid,
                "text": doc,
                "score": 1 - dist,  # cosine similarity from cosine distance
                "metadata": meta,
            }
            for uid, doc, dist, meta in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]

    def delete_user(self, user_id: str) -> None:
        """Hard delete all vectors for a user (GDPR)."""
        try:
            self.client.delete_collection(f"user_{user_id.replace('-', '_')}")
        except Exception:
            pass


@dataclass
class RetrievalResult:
    unit_id: str
    text: str
    score: float
    unit_type: str
    confidence: float
    tags: list[str]
    source: str  # "structured" | "fallback"


class HybridRetriever:
    """
    Two-tier retriever.

    Tier 1: Semantic search over MCM-structured units in ChromaDB.
    Tier 2: If top-1 score < FALLBACK_THRESHOLD, also retrieve from raw event log
            using a simple keyword/sentence match and return with explicit source tag.

    This ensures we never lose critical information due to MCM extraction misses.
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        memory_store: MemoryStore,
        event_log: EventLog,
        fallback_threshold: float = FALLBACK_THRESHOLD,
    ):
        self.vector_index = vector_index
        self.memory_store = memory_store
        self.event_log = event_log
        self._fallback_threshold = fallback_threshold

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        unit_type: str | None = None,
        domain: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant memory for a query. Falls back to raw event log if needed.

        Args:
            query:          Natural language retrieval query
            user_id:        The user whose memory to search
            top_k:          Number of structured results to return
            unit_type:      Optional filter: fact | preference | task_summary | abstraction | open_thread
            domain:         Optional domain filter (e.g. "trading", "health")
            min_confidence: Only return units with confidence >= this value
        """
        # Build metadata filter for ChromaDB
        where: dict[str, Any] = {}
        if unit_type:
            where["type"] = unit_type
        if domain:
            where["domain"] = domain

        raw_results = self.vector_index.search(
            query=query,
            user_id=user_id,
            n_results=top_k,
            where=where if where else None,
        )

        # Enrich with structured store metadata
        results: list[RetrievalResult] = []
        for r in raw_results:
            unit = self.memory_store.get_unit_by_id(r["id"])
            if unit is None or unit["is_superseded"]:
                continue
            if unit["confidence"] < min_confidence:
                continue
            results.append(
                RetrievalResult(
                    unit_id=r["id"],
                    text=r["text"],
                    score=r["score"],
                    unit_type=unit["type"],
                    confidence=unit["confidence"],
                    tags=unit["tags"],
                    source="structured",
                )
            )

        # Tier 2 fallback: if best structured result is low-confidence, also search raw log
        top_score = results[0].score if results else 0.0
        if top_score < self._fallback_threshold:
            logger.info(
                f"Structured retrieval score {top_score:.2f} below threshold "
                f"({self._fallback_threshold}) — activating raw log fallback for user {user_id}"
            )
            fallback_results = self._fallback_search(query, user_id, top_k=3)
            results.extend(fallback_results)

        # Sort by score descending, deduplicate by text
        seen_texts: set[str] = set()
        final: list[RetrievalResult] = []
        for r in sorted(results, key=lambda x: x.score, reverse=True):
            if r.text not in seen_texts:
                seen_texts.add(r.text)
                final.append(r)
            if len(final) >= top_k + 3:  # allow a few extra for fallback mix
                break

        return final

    def _fallback_search(self, query: str, user_id: str, top_k: int = 3) -> list[RetrievalResult]:
        """
        Simple keyword-based search over the raw event log.
        Used when structured memory retrieval confidence is low.
        In production, replace with a BM25 or lightweight embedding over raw turns.
        """
        query_words = set(query.lower().split())
        all_turns = []

        # Collect turns from all sessions for this user — for a real system,
        # index sessions separately; this is intentionally simple for now
        with self.event_log._conn() as conn:
            rows = conn.execute(
                "SELECT content_gz, session_id, turn_index FROM events WHERE user_id = ? ORDER BY id DESC LIMIT 500",
                (user_id,),
            ).fetchall()

        import zlib
        scored = []
        for row in rows:
            content = zlib.decompress(row["content_gz"]).decode("utf-8")
            content_words = set(content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((overlap / len(query_words), content, row["session_id"]))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, content, _ in scored[:top_k]:
            results.append(
                RetrievalResult(
                    unit_id="raw",
                    text=content,
                    score=score * 0.5,  # discount raw results vs. structured
                    unit_type="raw_turn",
                    confidence=score,
                    tags=["fallback"],
                    source="fallback",
                )
            )
        return results

    def format_for_context(self, results: list[RetrievalResult]) -> str:
        """
        Format retrieval results as a context block for injection into the main LLM prompt.
        """
        if not results:
            return ""

        lines = ["## Retrieved Memory\n"]
        structured = [r for r in results if r.source == "structured"]
        fallback = [r for r in results if r.source == "fallback"]

        if structured:
            lines.append("### From memory store:")
            for r in structured:
                prefix = f"[{r.unit_type}]"
                if r.confidence < 0.8:
                    prefix += f" (confidence: {r.confidence:.0%})"
                lines.append(f"- {prefix} {r.text}")

        if fallback:
            lines.append("\n### From conversation history (low-confidence fallback):")
            for r in fallback:
                lines.append(f"- {r.text}")

        return "\n".join(lines)
