"""
Self-generating training pipeline for the MCM read head (retrieval LoRA).

Core insight: the write head produces graph edges during consolidation.
From those edges, we can auto-generate (query, graph_schema, traversal_path) triples.
These triples become the training data for the read head.

This means the system generates its OWN retrieval training data — no manual labeling.
Each consolidation cycle produces more training pairs.
When enough accumulate, trigger a LoRA fine-tune on the read adapter.
"""

import json
import random
from pathlib import Path
from typing import Any

from ..data.knowledge_graph import MemoryGraph


# Templates for generating natural language queries from graph paths
QUERY_TEMPLATES_1HOP = [
    "What {relation} does the user have for {source}?",
    "Tell me about {relation} related to {source}",
    "What is the {relation} of {source}?",
    "{relation} for {source}?",
]

QUERY_TEMPLATES_2HOP = [
    "What {rel2} is associated with the user's {source}?",
    "For {source}, what {rel2} applies?",
    "What {rel2} connects to {source} through {rel1}?",
]


def generate_read_training_data(
    graph: MemoryGraph,
    user_id: str,
) -> list[dict[str, Any]]:
    """
    Generate (query, graph_schema, expected_path, answer) training examples
    from the current state of a user's knowledge graph.

    Returns list of training examples in chat format ready for LoRA fine-tuning.
    """
    graph_schema = graph.get_schema_text(user_id)
    retrieval_examples = graph.generate_retrieval_examples(user_id)

    training_pairs = []
    for ex in retrieval_examples:
        # Generate multiple query phrasings per path for diversity
        if ex["hops"] == 1:
            templates = QUERY_TEMPLATES_1HOP
        else:
            templates = QUERY_TEMPLATES_2HOP

        template = random.choice(templates)

        # Fill template
        source_content = ""
        path = ex["path"]
        if path and path[0] in graph.graph:
            source_content = graph.graph.nodes[path[0]].get("content", "")

        relations = ex.get("path_relations", [])
        rel1 = relations[0].replace("_", " ") if relations else "related"
        rel2 = relations[-1].replace("_", " ") if relations else "related"

        query = template.format(
            relation=rel1,
            rel1=rel1,
            rel2=rel2,
            source=source_content[:50],
        )

        # Build the expected output
        expected_output = {
            "reasoning": f"The query asks about {rel2} for '{source_content[:30]}'. Following path: {' → '.join(path)}",
            "path": path,
            "answer_from": ex["answer_nodes"],
            "confidence": 0.9 if ex["hops"] == 1 else 0.8,
        }

        # Format as chat training example
        training_pairs.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a memory retrieval model. Given a user query and a knowledge graph schema, output the optimal traversal path to answer the query.\n\nOutput ONLY valid JSON:\n{\"reasoning\": \"...\", \"path\": [\"node_id_1\", \"node_id_2\"], \"answer_from\": [\"node_id_2\"], \"confidence\": 0.95}",
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nKnowledge Graph:\n{graph_schema}",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(expected_output),
                },
            ],
            "metadata": {
                "user_id": user_id,
                "hops": ex["hops"],
                "source_path": path,
            },
        })

    return training_pairs


class ReadTrainingAccumulator:
    """
    Accumulates retrieval training examples over time.
    When threshold is met, dumps to JSONL file for LoRA fine-tuning.

    The self-improving loop:
      1. Write head consolidates → produces graph edges
      2. This class generates training pairs from graph
      3. When enough pairs accumulate → triggers read LoRA fine-tune
      4. Read head gets better at traversal
      5. Repeat
    """

    def __init__(self, output_dir: str = "data/read_training", threshold: int = 500):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self._buffer: list[dict] = []

    def add_examples(self, examples: list[dict]) -> None:
        self._buffer.extend(examples)

    def should_trigger_training(self) -> bool:
        return len(self._buffer) >= self.threshold

    def flush_to_file(self) -> str | None:
        """Write accumulated examples to a JSONL file. Returns the file path."""
        if not self._buffer:
            return None

        # Deduplicate by query text
        seen = set()
        unique = []
        for ex in self._buffer:
            query = ex["messages"][1]["content"]
            if query not in seen:
                seen.add(query)
                unique.append(ex)

        # Split 90/10 train/val
        random.shuffle(unique)
        split = int(len(unique) * 0.9)
        train = unique[:split]
        val = unique[split:]

        train_path = self.output_dir / "train.jsonl"
        val_path = self.output_dir / "val.jsonl"

        for path, data in [(train_path, train), (val_path, val)]:
            with open(path, "w") as f:
                for ex in data:
                    f.write(json.dumps(ex) + "\n")

        count = len(unique)
        self._buffer.clear()
        return str(train_path)

    def stats(self) -> dict:
        return {
            "buffered_examples": len(self._buffer),
            "threshold": self.threshold,
            "ready_for_training": self.should_trigger_training(),
        }
