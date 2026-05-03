"""
Memory Pipeline — the single entry point for agent code.

Production design (addresses all mem0 limitations):
  - Writes are <1ms: turns go to an append-only EventLog, no LLM on the write path
  - Consolidation is async: MCM runs in a background worker, not inline
  - Never loses data: raw EventLog is never deleted → full-context fallback available
  - Contradiction-safe: memory units are versioned + immutable with supersession
  - GDPR-ready: delete_user() hard-deletes both EventLog and MemoryStore

Usage:
    pipeline = MemoryPipeline.create(
        user_id="user_123",
        mcm_model_path="checkpoints/mcm-v1",
    )
    pipeline.add_turn("user", "I prefer intraday trading", session_id="sess_1")
    results = pipeline.retrieve("what does the user prefer?")
    print(pipeline.format_context(results))
"""

import asyncio
from typing import Any

from ..data.event_log import EventLog, RawTurn
from ..data.memory_store import MemoryStore
from ..memory_model.mcm_inference import MCMInference
from ..retriever.hybrid_retriever import HybridRetriever, RetrievalResult, VectorIndex
from .consolidation_worker import ConsolidationConfig, ConsolidationWorker


class MemoryPipeline:
    """
    Single interface for an agent to interact with the memory system.

    Write path:  add_turn()  → EventLog (<1ms, no LLM)
    Async path:  worker consolidates in the background (MCM inference)
    Read path:   retrieve()  → HybridRetriever (structured + fallback)
    """

    def __init__(
        self,
        user_id: str,
        event_log: EventLog,
        memory_store: MemoryStore,
        vector_index: VectorIndex,
        retriever: HybridRetriever,
        worker: ConsolidationWorker,
    ):
        self.user_id = user_id
        self._event_log = event_log
        self._memory_store = memory_store
        self._vector_index = vector_index
        self._retriever = retriever
        self._worker = worker
        self._turn_index = 0

    @classmethod
    def create(
        cls,
        user_id: str,
        mcm_model_path: str,
        event_db: str = "data/events.db",
        memory_db: str = "data/memory.db",
        chroma_dir: str = "data/.chromadb",
        consolidation_config: ConsolidationConfig | None = None,
    ) -> "MemoryPipeline":
        """Factory method — wires up all components."""
        event_log = EventLog(event_db)
        memory_store = MemoryStore(memory_db)
        vector_index = VectorIndex(chroma_dir)
        retriever = HybridRetriever(vector_index, memory_store, event_log)
        mcm = MCMInference(mcm_model_path)
        worker = ConsolidationWorker(mcm, event_log, memory_store, vector_index, consolidation_config)
        return cls(user_id, event_log, memory_store, vector_index, retriever, worker)

    # ------------------------------------------------------------------
    # Write path — always fast, no LLM involved
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str, session_id: str = "default") -> int:
        """
        Append a turn to the event log. Returns event ID.
        This is the only method agents should call on the write path.
        Latency: <1ms.
        """
        turn = RawTurn(
            role=role,
            content=content,
            session_id=session_id,
            user_id=self.user_id,
            turn_index=self._turn_index,
        )
        self._turn_index += 1
        return self._event_log.append(turn)

    # ------------------------------------------------------------------
    # Consolidation — run async in background, or force for tests
    # ------------------------------------------------------------------

    async def consolidate_async(self) -> dict[str, Any]:
        """Trigger consolidation immediately (async). Use in tests or CLI."""
        return await self._worker.consolidate_now(self.user_id)

    def consolidate_sync(self) -> dict[str, Any]:
        """Blocking consolidation for non-async contexts."""
        return asyncio.run(self._worker.consolidate_now(self.user_id))

    # ------------------------------------------------------------------
    # Read path — two-tier retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        unit_type: str | None = None,
        domain: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant memory units for a query.
        Falls back to raw event log automatically if structured confidence is low.
        """
        return self._retriever.retrieve(
            query=query,
            user_id=self.user_id,
            top_k=top_k,
            unit_type=unit_type,
            domain=domain,
            min_confidence=min_confidence,
        )

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results as a context block for the main LLM prompt."""
        return self._retriever.format_for_context(results)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        memory_stats = self._memory_store.stats(self.user_id)
        pending = self._event_log.unconsolidated_count(self.user_id)
        return {**memory_stats, "pending_consolidation": pending}

    def delete_user(self) -> None:
        """GDPR right to be forgotten — hard deletes all data for this user."""
        self._memory_store.delete_user_memory(self.user_id)
        self._vector_index.delete_user(self.user_id)
        # EventLog entries are also deleted for completeness
        with self._event_log._conn() as conn:
            conn.execute("DELETE FROM events WHERE user_id = ?", (self.user_id,))
            conn.execute("DELETE FROM consolidation_offsets WHERE user_id = ?", (self.user_id,))

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic + structured retrieval over the memory store."""
        return self.retriever.search(query, top_k=top_k)

    def get_full_memory(self) -> dict[str, Any]:
        """Return the full structured memory store."""
        return self.retriever.get_all()

    def _format_log(self, turns: list[RawTurn]) -> str:
        lines = []
        for i, turn in enumerate(turns, 1):
            lines.append(f"Turn {i} [{turn.role}]: {turn.content}")
        return "\n".join(lines)

    def force_consolidate(self) -> dict[str, Any]:
        """Manually trigger consolidation regardless of threshold."""
        return self.consolidate()
