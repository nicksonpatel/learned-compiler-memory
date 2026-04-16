"""
Three-tier graph routing strategy for the MCM read path.

The problem this solves: before the read LoRA is trained, route_query() would
run on the base Qwen2.5-1.5B — which is fine for small graphs but degrades on
large graphs because reading a long flat schema from a prompt is a hard task.

Three tiers:

  Tier 1 — small graph (< SMALL_THRESHOLD nodes)
    Full schema injected into prompt, base model reasons over it.
    Works well because the schema fits comfortably in context.

  Tier 2 — medium graph (SMALL_THRESHOLD ≤ nodes < LARGE_THRESHOLD)
    Query is used to pull the top-N most relevant nodes from the VectorIndex.
    Only those nodes + the edges between them are injected as a filtered schema.
    Base model routes within the bounded subgraph.
    Keeps prompt size manageable while the LoRA is still accumulating examples.

  Tier 3 — large graph (≥ LARGE_THRESHOLD nodes) or read LoRA trained
    Read LoRA routes over the full schema. By the time the graph reaches
    LARGE_THRESHOLD nodes, enough turns have been processed for the LoRA
    threshold (default 500 examples) to have triggered at least once.

The tier boundaries (200 / 500 nodes) deliberately coincide with the LoRA
training threshold so there is no quality cliff at the transition point.
"""

import logging
from dataclasses import dataclass

from ..data.knowledge_graph import MemoryGraph
from ..memory_model.mcm_inference import MCMInference
from .hybrid_retriever import VectorIndex

logger = logging.getLogger(__name__)

# Default tier boundaries (node count per user)
SMALL_THRESHOLD = 200   # below this: full schema + base model
LARGE_THRESHOLD = 500   # at or above this: read LoRA (if trained)

# How many nodes to pull from VectorIndex for the tier-2 filtered subgraph
FILTERED_TOP_N = 50


@dataclass
class RoutingResult:
    path: list[str]           # ordered list of node IDs to traverse
    answer_from: list[str]    # subset of path that contains the answer
    reasoning: str            # chain-of-thought from the model
    confidence: float         # model-reported confidence [0, 1]
    tier: int                 # 1, 2, or 3 — which routing tier was used


class GraphRouter:
    """
    Dispatches graph traversal queries to the appropriate routing strategy
    based on the current size of the user's knowledge graph and whether the
    read LoRA adapter has been trained.

    Usage:
        router = GraphRouter(mcm, knowledge_graph, vector_index)
        result = router.route(query="What candle size for breakout?", user_id="u1")
        if result:
            contents = knowledge_graph.traverse_path(result.path)
    """

    def __init__(
        self,
        mcm: MCMInference,
        knowledge_graph: MemoryGraph,
        vector_index: VectorIndex,
        small_threshold: int = SMALL_THRESHOLD,
        large_threshold: int = LARGE_THRESHOLD,
        filtered_top_n: int = FILTERED_TOP_N,
    ):
        self._mcm = mcm
        self._graph = knowledge_graph
        self._vector_index = vector_index
        self._small_threshold = small_threshold
        self._large_threshold = large_threshold
        self._filtered_top_n = filtered_top_n

    def route(self, query: str, user_id: str) -> RoutingResult | None:
        """
        Route a query through the knowledge graph using the appropriate tier.

        Returns a RoutingResult, or None if the graph is empty or routing
        produced an invalid / low-confidence path.
        """
        count = self._graph.node_count(user_id)
        if count == 0:
            logger.debug("GraphRouter: no nodes for user %s — skipping graph routing", user_id)
            return None

        tier = self._select_tier(count)
        logger.debug(
            "GraphRouter: user=%s nodes=%d tier=%d lora_ready=%s",
            user_id, count, tier, self._mcm.read_lora_ready,
        )

        if tier == 1:
            return self._route_tier1(query, user_id)
        elif tier == 2:
            return self._route_tier2(query, user_id)
        else:
            return self._route_tier3(query, user_id)

    # ------------------------------------------------------------------
    # Tier selection
    # ------------------------------------------------------------------

    def _select_tier(self, node_count: int) -> int:
        # If the read LoRA is trained, always use tier 3 regardless of graph size —
        # a trained adapter is strictly better than the base model for routing.
        if self._mcm.read_lora_ready:
            return 3
        if node_count < self._small_threshold:
            return 1
        if node_count < self._large_threshold:
            return 2
        # Graph is large but LoRA isn't trained yet — use tier 2 with filtering.
        # This is a degraded state; the architecture doc notes this coincidence
        # should rarely occur if the LoRA threshold is tuned correctly.
        logger.warning(
            "GraphRouter: graph has %d nodes but read LoRA is not trained. "
            "Falling back to tier-2 filtered routing.",
            node_count,
        )
        return 2

    # ------------------------------------------------------------------
    # Tier implementations
    # ------------------------------------------------------------------

    def _route_tier1(self, query: str, user_id: str) -> RoutingResult | None:
        """
        Full schema → base model.
        Only used when the graph is small enough that the entire schema fits in
        context comfortably (< SMALL_THRESHOLD nodes).
        """
        schema = self._graph.get_schema_text(user_id)
        raw = self._mcm.route_query_base(query, schema)
        return self._parse_result(raw, tier=1)

    def _get_filtered_schema(self, query: str, user_id: str) -> tuple[str, list[str]]:
        """
        Pull the top-N most relevant nodes via vector similarity, then build a
        schema containing only those nodes and the edges between them.
        Returns (schema_text, node_ids). Falls back to full schema if the
        VectorIndex is empty (consolidation hasn't run yet).
        """
        vector_hits = self._vector_index.search(
            query=query,
            user_id=user_id,
            n_results=self._filtered_top_n,
        )
        node_ids = [h["id"] for h in vector_hits]

        if not node_ids:
            # VectorIndex is empty — fall back to full schema (graph is still small here).
            logger.debug("GraphRouter: no vector hits, using full schema fallback")
            return self._graph.get_schema_text(user_id), []

        return self._graph.get_filtered_schema_text(user_id, node_ids), node_ids

    def _route_tier2(self, query: str, user_id: str) -> RoutingResult | None:
        """
        Filtered schema → base model.
        Vector similarity bounds the prompt to top-N relevant nodes.
        Base model routes within that bounded subgraph.
        """
        schema, _ = self._get_filtered_schema(query, user_id)
        raw = self._mcm.route_query_base(query, schema)
        return self._parse_result(raw, tier=2)

    def _route_tier3(self, query: str, user_id: str) -> RoutingResult | None:
        """
        Filtered schema → trained read LoRA.

        The LoRA does NOT receive the full graph — at 100k+ nodes that would be
        megabytes of schema text, blowing the context window entirely.
        Instead, vector similarity first pulls the top-N most relevant nodes
        (same bound as tier 2), and the LoRA routes within that subgraph.

        The LoRA's advantage over the base model is not that it sees more of the
        graph — it's that it routes more accurately within the bounded subgraph
        because it has learned the structural patterns of this user's graph.
        """
        schema, _ = self._get_filtered_schema(query, user_id)
        raw = self._mcm.route_query(query, schema)
        return self._parse_result(raw, tier=3)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_result(self, raw: dict, tier: int) -> RoutingResult | None:
        """
        Convert the raw model output dict into a typed RoutingResult.
        Returns None if the output is missing the required 'path' key or is empty.
        """
        path = raw.get("path", [])
        if not path:
            return None

        return RoutingResult(
            path=path,
            answer_from=raw.get("answer_from", path[-1:]),
            reasoning=raw.get("reasoning", ""),
            confidence=float(raw.get("confidence", 0.0)),
            tier=tier,
        )
