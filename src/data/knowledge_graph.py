"""
Knowledge Graph for structured memory — stores memory units as nodes and
their relationships as typed edges.

Uses NetworkX for in-process operation (swap to Neo4j for production scale).
Serializable to/from JSON for persistence.

Why a graph (vs. flat ChromaDB):
  - The LoRA learns graph traversal paths, not just embedding similarity
  - Relations like "uses_strategy", "configured_with" are first-class
  - Multi-hop queries ("What candle size does the user prefer for breakout?")
    resolve to a 2-hop path, which the LoRA can learn directly
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx


class MemoryGraph:
    """
    In-memory knowledge graph over memory units.

    Nodes = memory units (facts, preferences, abstractions, etc.)
    Edges = typed relations between units (uses_strategy, configured_with, etc.)

    Serializable to disk for persistence between restarts.
    """

    def __init__(self, persist_path: str | None = None):
        self.graph = nx.DiGraph()
        self._persist_path = persist_path
        if persist_path and Path(persist_path).exists():
            self._load()

    # ------------------------------------------------------------------
    # Write operations (called by ConsolidationWorker)
    # ------------------------------------------------------------------

    def add_unit(self, unit_id: str, user_id: str, unit_type: str, content: str, **attrs) -> None:
        """Add a memory unit as a node."""
        self.graph.add_node(
            unit_id,
            user_id=user_id,
            type=unit_type,
            content=content,
            created_at=datetime.now(timezone.utc).isoformat(),
            **attrs,
        )

    def add_edge(self, from_id: str, to_id: str, relation: str, **attrs) -> None:
        """Add a typed relation between two memory units."""
        if from_id not in self.graph or to_id not in self.graph:
            return  # only link existing nodes
        self.graph.add_edge(from_id, to_id, relation=relation, **attrs)

    def remove_node(self, unit_id: str) -> None:
        if unit_id in self.graph:
            self.graph.remove_node(unit_id)

    # ------------------------------------------------------------------
    # Read operations (used by LoRA retrieval + fallback)
    # ------------------------------------------------------------------

    def get_neighbors(self, unit_id: str, relation: str | None = None) -> list[dict]:
        """
        Get all nodes reachable in 1 hop from unit_id.
        Optionally filter by relation type.
        """
        if unit_id not in self.graph:
            return []

        results = []
        for _, target, data in self.graph.out_edges(unit_id, data=True):
            if relation and data.get("relation") != relation:
                continue
            node_data = self.graph.nodes[target]
            results.append({"id": target, "relation": data.get("relation"), **node_data})

        # Also check incoming edges
        for source, _, data in self.graph.in_edges(unit_id, data=True):
            if relation and data.get("relation") != relation:
                continue
            node_data = self.graph.nodes[source]
            results.append({"id": source, "relation": data.get("relation"), "direction": "incoming", **node_data})

        return results

    def traverse_path(self, path: list[str]) -> list[dict]:
        """
        Follow an explicit path of node IDs through the graph.
        Returns the node data for each hop. Used when the LoRA outputs a traversal path.
        """
        results = []
        for node_id in path:
            if node_id in self.graph:
                data = self.graph.nodes[node_id]
                results.append({"id": node_id, **data})
        return results

    def find_paths(self, from_id: str, to_id: str, max_hops: int = 3) -> list[list[str]]:
        """Find all simple paths between two nodes up to max_hops."""
        if from_id not in self.graph or to_id not in self.graph:
            return []
        try:
            return list(nx.all_simple_paths(self.graph, from_id, to_id, cutoff=max_hops))
        except nx.NetworkXError:
            return []

    def get_subgraph_for_user(self, user_id: str) -> dict[str, Any]:
        """
        Return a serialized view of a user's memory graph.
        This is what gets fed to the LoRA read head as the "graph schema" input.
        """
        user_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("user_id") == user_id]
        subgraph = self.graph.subgraph(user_nodes)

        nodes = []
        for n, data in subgraph.nodes(data=True):
            nodes.append({
                "id": n,
                "type": data.get("type"),
                "content": data.get("content", "")[:100],  # truncate for prompt context
            })

        edges = []
        for u, v, data in subgraph.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "relation": data.get("relation"),
            })

        return {"nodes": nodes, "edges": edges}

    def node_count(self, user_id: str) -> int:
        """Return the number of active memory nodes for a user."""
        return sum(1 for _, d in self.graph.nodes(data=True) if d.get("user_id") == user_id)

    def get_schema_text(self, user_id: str) -> str:
        """
        Serialize the user's graph into a compact text format for LLM input.
        Example: [m1:fact:"trades NIFTY"] --uses_strategy--> [m2:fact:"breakout strategy"]
        """
        sub = self.get_subgraph_for_user(user_id)
        lines = []

        # Nodes
        for n in sub["nodes"]:
            lines.append(f'[{n["id"]}:{n["type"]}:"{n["content"]}"]')

        # Edges
        for e in sub["edges"]:
            lines.append(f'[{e["from"]}] --{e["relation"]}--> [{e["to"]}]')

        return "\n".join(lines)

    def get_filtered_schema_text(self, user_id: str, node_ids: list[str]) -> str:
        """
        Like get_schema_text but restricted to a specific set of node IDs.
        Includes edges between those nodes only. Used for tier-2 routing
        where the prompt is bounded by vector-similarity-selected nodes.
        """
        node_set = set(node_ids)
        lines = []

        for nid in node_ids:
            if nid not in self.graph:
                continue
            d = self.graph.nodes[nid]
            if d.get("user_id") != user_id:
                continue
            content = d.get("content", "")[:100]
            lines.append(f'[{nid}:{d.get("type", "unknown")}:"{content}"]')

        for u, v, data in self.graph.edges(data=True):
            if u in node_set and v in node_set:
                lines.append(f'[{u}] --{data.get("relation", "related_to")}--> [{v}]')

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Retrieval training data generation
    # ------------------------------------------------------------------

    def generate_retrieval_examples(self, user_id: str) -> list[dict]:
        """
        Auto-generate (query, path, answer) training triples from the graph.
        These become LoRA fine-tuning data for the read head.

        For each edge chain of length 1-3/2, generate a natural language question
        that would require traversing that path.
        """
        examples = []
        user_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("user_id") == user_id]

        for node in user_nodes:
            node_data = self.graph.nodes[node]

            # 1-hop examples: direct neighbors
            for _, target, edge_data in self.graph.out_edges(node, data=True):
                target_data = self.graph.nodes[target]
                relation = edge_data.get("relation", "related_to")

                examples.append({
                    "query_template": f"What is {relation.replace('_', ' ')} for {node_data.get('content', '')}?",
                    "path": [node, target],
                    "path_relations": [relation],
                    "answer_nodes": [target],
                    "answer_text": target_data.get("content", ""),
                    "hops": 1,
                })

            # 2-hop examples: node → middle → end
            for _, mid, edge1 in self.graph.out_edges(node, data=True):
                for _, end, edge2 in self.graph.out_edges(mid, data=True):
                    if end == node:
                        continue  # skip cycles
                    end_data = self.graph.nodes[end]
                    rel1 = edge1.get("relation", "related_to")
                    rel2 = edge2.get("relation", "related_to")

                    examples.append({
                        "query_template": f"What {rel2.replace('_', ' ')} is associated with {node_data.get('content', '')}?",
                        "path": [node, mid, end],
                        "path_relations": [rel1, rel2],
                        "answer_nodes": [end],
                        "answer_text": end_data.get("content", ""),
                        "hops": 2,
                    })

        return examples

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        if not self._persist_path:
            return
        persist = Path(self._persist_path)
        persist.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        # Write to a temp file then atomically replace — prevents corrupt JSON on kill
        tmp = persist.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.replace(persist)

    def _load(self) -> None:
        with open(self._persist_path) as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, directed=True)

    def clear_user(self, user_id: str) -> None:
        """Remove all nodes belonging to a user (GDPR)."""
        to_remove = [n for n, d in self.graph.nodes(data=True) if d.get("user_id") == user_id]
        self.graph.remove_nodes_from(to_remove)
