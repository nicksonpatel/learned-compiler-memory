# Memory Consolidation Model (MCM) — Architecture Document

> **Version:** 0.2  
> **Base model:** Qwen2.5-1.5B-Instruct  
> **Fine-tuning strategy:** QLoRA (4-bit NF4, dual LoRA adapters on shared base)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Principles](#2-architecture-principles)
3. [How MCM Differs from mem0](#3-how-mcm-differs-from-mem0)
4. [Component Catalog](#4-component-catalog)
5. [Storage Layer Architecture](#5-storage-layer-architecture)
6. [The Three Data Paths](#6-the-three-data-paths)
   - 6.1 [Write Path — Agent Turn Ingestion](#61-write-path--agent-turn-ingestion)
   - 6.2 [Consolidation Path — Background MCM Processing](#62-consolidation-path--background-mcm-processing)
   - 6.3 [Read Path — Retrieval](#63-read-path--retrieval)
7. [The Dual-Head MCM Model](#7-the-dual-head-mcm-model)
8. [The Self-Improving Training Loop](#8-the-self-improving-training-loop)
9. [Knowledge Graph Design](#9-knowledge-graph-design)
10. [GDPR and Data Deletion](#10-gdpr-and-data-deletion)
11. [mem0 Failure Mode Resolution](#11-mem0-failure-mode-resolution)
12. [Deployment Notes](#12-deployment-notes)

---

## 1. System Overview

MCM is a production memory system for AI agents built around one core idea: **memory formation and memory retrieval are learnable tasks that should be handled by a dedicated, continuously improving small model** — not by prompt-engineering a general-purpose large model on every agent turn.

At a high level the system has three moving parts:

```
┌─────────────────────────────────────────────────────────────────┐
│                         AGENT RUNTIME                           │
│                                                                 │
│  agent turn  ──►  MemoryPipeline.add_turn()                     │
│                       │  (<1ms, no LLM)                         │
│                       ▼                                         │
│                   EventLog (SQLite WAL)  ◄── source of truth    │
│                                                                 │
│  agent query  ──►  MemoryPipeline.retrieve()                    │
│                       │                                         │
│                       ▼                                         │
│                   HybridRetriever                               │
│                    ├── Tier 1: ChromaDB semantic search         │
│                    └── Tier 2: raw EventLog fallback            │
└─────────────────────────────────────────────────────────────────┘

                        (async, background)
┌─────────────────────────────────────────────────────────────────┐
│                    CONSOLIDATION WORKER                         │
│                                                                 │
│  EventLog  ──►  MCM write head  ──►  MemoryStore (structured)  │
│                              ├──────►  VectorIndex (semantic)  │
│                              └──────►  MemoryGraph (relational) │
│                                              │                  │
│                                              ▼                  │
│                              ReadTrainingAccumulator            │
│                                              │                  │
│                                (at threshold)▼                  │
│                              train_mcm.py --head read           │
│                              (read LoRA improves over time)     │
└─────────────────────────────────────────────────────────────────┘
```

The write path is intentionally decoupled from the LLM: agent turns are appended to the EventLog in under 1 ms. The MCM model runs asynchronously in a background worker, batching 100 turns per inference call. This is the fundamental fix to mem0's 20-second write latency.

---

## 2. Architecture Principles

| Principle | Rationale |
|---|---|
| **Never run LLM on the write path** | Latency of `add_turn()` must be <1ms for production. LLMs are only called during background consolidation. |
| **Never delete the raw log** | mem0 throws away source turns after extraction. Doing so permanently discards information and creates accuracy regressions. The EventLog is append-only and retained forever. |
| **Memory units are immutable** | Contradictions are resolved by creating a new unit that *supersedes* the old one. The old unit is preserved with `is_superseded=1`. Full audit trail is maintained. |
| **One LLM call per batch, not per turn** | The ConsolidationWorker batches up to 100 turns and makes a single MCM inference call. This is an N→1 reduction in cost over mem0's N+1 pattern. |
| **Structure and semantics are orthogonal** | Structured units live in MemoryStore + MemoryGraph. Semantic search lives in ChromaDB. Both are populated from the same MCM output. |
| **The model learns its own training data** | The write head auto-generates (query, path, answer) triples every consolidation cycle. These become training data for the read head. The system improves continuously without human labelling. |

---

## 3. How MCM Differs from mem0

The canonical open-source comparison point is [mem0](https://github.com/mem0ai/mem0). The architectural divergences are deliberate responses to reported production failures (mem0 issue #4573).

```
mem0 (baseline)                    MCM (this system)
────────────────────────────────────────────────────────────────────
add() blocks on LLM (~20s)         add_turn() writes to SQLite (<1ms)
One LLM call per turn (N+1)        One MCM call per 100 turns (1:N)
Raw turns discarded after extract  EventLog retained, used for fallback
Memory units mutated in place       Immutable units with supersession chain
Single-tier vector retrieval        Two-tier: ChromaDB + EventLog fallback
Metadata length limits cause bugs   Enforced 500-char field limit on write
No knowledge structure              MemoryGraph with typed relational edges
Static model, no improvement        Read LoRA trained on write-head output
```

---

## 4. Component Catalog

### 4.1 MemoryPipeline
**File:** `src/agent/memory_pipeline.py`  
**Role:** Single entry point for all agent code. Owns no logic — delegates to the six subsystems below.

| Method | Description |
|---|---|
| `MemoryPipeline.create(user_id, ...)` | Factory. Wires up EventLog, MemoryStore, MCMInference, VectorIndex, MemoryGraph, ConsolidationWorker. |
| `add_turn(role, content, session_id)` | Appends a RawTurn to the EventLog. <1ms. No LLM. |
| `consolidate_async()` | Fires the background consolidation worker (non-blocking). |
| `consolidate_sync()` | Blocks until consolidation for this user is complete (for tests/CLI). |
| `retrieve(query, top_k, unit_type, domain, min_confidence)` | Delegates to HybridRetriever. Returns list of RetrievalResult. |
| `format_context(results)` | Formats retrieval results into a markdown context block for LLM prompt injection. |
| `stats()` | Returns combined stats from EventLog + MemoryStore. |
| `delete_user()` | GDPR hard delete across all four storage layers. |

---

### 4.2 EventLog
**File:** `src/data/event_log.py`  
**Role:** Append-only source of truth for every raw agent interaction turn.

```
events table
  id             INTEGER PK AUTOINCREMENT
  user_id        TEXT NOT NULL
  session_id     TEXT NOT NULL
  turn_index     INTEGER NOT NULL
  role           TEXT NOT NULL         -- user | assistant | tool
  content_gz     BLOB NOT NULL         -- zlib-compressed (level 6)
  metadata_json  TEXT NOT NULL         -- tool name, output, etc. (max 4096 bytes)
  timestamp      TEXT NOT NULL

consolidation_offsets table
  user_id        TEXT PK
  last_event_id  INTEGER NOT NULL      -- high-water mark for the worker
  updated_at     TEXT NOT NULL
```

Key behaviors:
- SQLite WAL mode + `PRAGMA synchronous=NORMAL` → concurrent reads, single writer, crash-safe.
- `append()` returns the assigned event ID immediately after one SQLite insert.
- `get_unconsolidated(user_id, batch_size)` reads events `WHERE id > last_event_id`, returns up to `batch_size` turns.
- `mark_consolidated(user_id, last_event_id)` is an upsert — idempotent if the worker crashes and restarts.
- `get_session_log(user_id, session_id)` returns full uncompressed session for fallback retrieval.

---

### 4.3 MemoryStore
**File:** `src/data/memory_store.py`  
**Role:** Immutable, versioned persistence for MCM-extracted memory units.

```
memory_units table
  id              TEXT PK              -- uuid4
  user_id         TEXT NOT NULL
  type            TEXT NOT NULL        -- fact | preference | task_summary | abstraction | open_thread
  content         TEXT NOT NULL
  domain          TEXT                 -- e.g. "trading", "health", "coding"
  confidence      REAL NOT NULL        -- clamped to [0.0, 1.0]
  tags_json       TEXT NOT NULL        -- JSON array of string tags
  source_turns    TEXT NOT NULL        -- JSON array of event log IDs that generated this unit
  version         INTEGER NOT NULL     -- always 1 (versioning is via supersession chain)
  supersedes_json TEXT NOT NULL        -- JSON array of unit IDs this replaces
  is_superseded   INTEGER NOT NULL     -- 0=active, 1=superseded
  valid_from      TEXT NOT NULL        -- ISO timestamp when written
  valid_until     TEXT                 -- NULL if currently active
```

Supersession protocol when a contradiction is detected:
```
1. write_unit(..., supersedes=["old-uuid-1", "old-uuid-2"])
2.   → UPDATE memory_units SET is_superseded=1, valid_until=now WHERE id IN (old ids)
3.   → INSERT new unit with supersedes_json=["old-uuid-1", "old-uuid-2"]

Full history survives. get_history(unit_id) walks the chain oldest → newest.
```

---

### 4.4 VectorIndex
**File:** `src/retriever/hybrid_retriever.py` (`VectorIndex` class)  
**Role:** Per-user ChromaDB collections for semantic search over MCM-structured units.

- One collection per user: `user_{user_id}` (hyphens replaced to satisfy ChromaDB naming constraints).
- Cosine similarity space (`hnsw:space = cosine`); score = `1 - cosine_distance`.
- Metadata field values enforced ≤500 characters at write time (fixes mem0 silence-truncation bug).
- `upsert()` is idempotent — safe to re-index if the worker re-processes on restart.
- `delete_user()` drops the entire collection (GDPR).

---

### 4.5 MemoryGraph
**File:** `src/data/knowledge_graph.py`  
**Role:** Typed directed knowledge graph over memory units. Powers path-based retrieval and generates read-head training data.

```
Nodes  = memory units  (unit_id → {user_id, type, content, created_at})
Edges  = typed relations  (unit_id_a → unit_id_b, relation: str)

Example edge types:
  uses_strategy       -- fact "trades NIFTY" → fact "uses breakout strategy"
  configured_with     -- abstraction "breakout setup" → preference "15m candle"
  follows_from        -- task_summary "backtested RSI" → abstraction "RSI filter rule"
  contradicts         -- (implicit via supersession chain, not an edge)
  related_to          -- generic fallback
```

Backed by NetworkX `DiGraph` in-process (swap to Neo4j for >10M nodes). Serialized to JSON via `nx.node_link_data()` for persistence between restarts. Saves are atomic: data is written to a `.tmp` file and renamed via `os.replace()` (a single `rename(2)` syscall), so a mid-save kill cannot corrupt the live graph file.

Key methods:
- `node_count(user_id)` → number of active nodes for a user. Used by `GraphRouter` for tier selection.
- `get_schema_text(user_id)` → compact text representation: `[id:type:"content"] --relation--> [id]`. Used only for small graphs (tier-1 routing).
- `get_filtered_schema_text(user_id, node_ids)` → same format but restricted to a given set of node IDs and only edges between them. Used for tier-2 and tier-3 routing to keep the prompt bounded regardless of total graph size.
- `traverse_path(path: list[str])` → follows the router-output node ID list, returns content at each hop.
- `generate_retrieval_examples(user_id)` → 1-hop and 2-hop (query_template, path, answer) triples.
- `clear_user(user_id)` → removes all user nodes and their incident edges (GDPR).

---

### 4.6 MCMInference (Dual-Head)
**File:** `src/memory_model/mcm_inference.py`  
**Role:** The fine-tuned model. Qwen2.5-1.5B-Instruct with two LoRA adapters on a single loaded base.

Key methods beyond `consolidate()` and `route_query()`:

| Method / Property | Description |
|---|---|
| `read_lora_ready` | `True` if the read LoRA adapter is loaded. `GraphRouter` checks this to decide whether to use tier-3 routing. |
| `route_query(query, schema)` | Read head with LoRA active. Routes over a vector-filtered subgraph schema. |
| `route_query_base(query, schema)` | Same prompt, but always uses the frozen base model. Called by `GraphRouter` for tier-1 and tier-2 before the LoRA is trained. |

See [Section 7](#7-the-dual-head-mcm-model) for full design.

---

### 4.7 ConsolidationWorker
**File:** `src/agent/consolidation_worker.py`  
**Role:** Async background worker. Reads EventLog, calls MCM write head, writes to all three downstream storage layers, accumulates read-head training data.

Parameters (`ConsolidationConfig`):

| Config | Default | Effect |
|---|---|---|
| `turn_threshold` | 50 | Minimum unconsolidated turns before worker fires |
| `batch_size` | 100 | Maximum turns per single MCM inference call |
| `poll_interval_seconds` | 5.0 | How often the loop checks for new turns |
| `min_confidence` | 0.5 | Discard MCM units below this confidence |
| `read_training_threshold` | 500 | Trigger read-head LoRA re-train after this many accumulated examples |

---

### 4.8 HybridRetriever
**File:** `src/retriever/hybrid_retriever.py` (`HybridRetriever` class)  
**Role:** Two-tier retrieval with automatic fallback.

| Tier | Mechanism | Trigger |
|---|---|---|
| 1 (Hot) | ChromaDB cosine similarity over MCM-structured units | Always attempted first |
| 2 (Cold) | Keyword/sentence match over raw EventLog sessions | Top-1 score < `fallback_threshold` (default 0.65) |

`fallback_threshold` is a constructor parameter (default `0.65`) — it is not a hardcoded constant. Pass a different value if your embedding model has a different score distribution. The threshold determines when tier-2 raw-log fallback activates; calibrate it against known-good vs. known-bad retrievals for your embedding model and domain.

Tier 2 fallback is what allows MCM to match or beat mem0's retrieval accuracy despite the raw log never being thrown away.

---

### 4.9 GraphRouter
**File:** `src/retriever/graph_router.py`  
**Role:** Three-tier graph routing strategy. Dispatches graph traversal queries to the appropriate routing model based on the user's graph size and whether the read LoRA has been trained.

| Tier | Condition | Schema injected into prompt | Routing model |
|---|---|---|---|
| 1 | No LoRA, < 200 nodes | Full schema (`get_schema_text`) | Base model |
| 2 | No LoRA, ≥ 200 nodes | Filtered schema — top-50 nodes by vector similarity | Base model |
| 3 | LoRA trained (any graph size) | Filtered schema — top-50 nodes by vector similarity | Read LoRA |

**Critical design point:** both tier-2 and tier-3 inject the *same filtered schema* into the prompt. At 100k+ nodes, `get_schema_text()` would produce megabytes of text — far beyond any context window. The LoRA's advantage over the base model is not that it sees more of the graph; it is that it routes more accurately *within a bounded subgraph* because it has learned the structural patterns of this user's graph during training. Vector similarity pre-selects the relevant nodes; the LoRA decides the traversal path through them.

The tier boundaries (200 / 500 nodes) are intended to roughly coincide with the LoRA training threshold, but this alignment is not guaranteed. The two thresholds govern different things and must be tuned together:

- `large_threshold` (default 500) — node count above which prompt injection alone degrades.
- `read_training_threshold` in `ConsolidationConfig` (default 500) — example count before the first LoRA training run.

In sparse-graph domains (few edges per consolidation batch), the graph can reach 500 nodes long before 500 training examples have accumulated — because training examples are generated from graph *edges*, not nodes. A graph that grows wide but shallow (many isolated fact nodes, few relations) produces few 1-hop or 2-hop examples. In this case the system will be in tier-2 (filtered schema, base model) even after the graph is large. This is safe but suboptimal. If your domain is known to produce sparse graphs, lower `read_training_threshold` or raise `large_threshold` accordingly.

Parameters (`GraphRouter.__init__`):

| Config | Default | Effect |
|---|---|---|
| `small_threshold` | 200 | Node count below which full schema is safe for tier-1 |
| `large_threshold` | 500 | Node count at which tier-2 applies without a trained LoRA |
| `filtered_top_n` | 50 | Number of nodes pulled from VectorIndex for the filtered subgraph |

---

### 4.10 ReadTrainingAccumulator
**File:** `src/memory_model/read_training_generator.py`  
**Role:** Buffers (query, graph, path) training pairs produced by each consolidation cycle, writes train/val JSONL splits when the threshold is reached.

---

## 5. Storage Layer Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      STORAGE LAYERS                              │
│                                                                  │
│   Layer 1: EventLog (SQLite WAL)                                 │
│   ─────────────────────────────                                  │
│   • Append-only, zlib-compressed raw turns                       │
│   • Source of truth, never deleted                               │
│   • Consolidation offset tracking (high-water mark per user)     │
│   • File: events.db                                              │
│                                                                  │
│   Layer 2: MemoryStore (SQLite WAL)                              │
│   ─────────────────────────────────                              │
│   • Immutable versioned memory units (fact, preference, etc.)    │
│   • Supersession chain for contradiction resolution              │
│   • Filtered by is_superseded=0 for active view                  │
│   • File: memory.db                                              │
│                                                                  │
│   Layer 3: VectorIndex (ChromaDB)                                │
│   ────────────────────────────────                               │
│   • Per-user HNSW collections, cosine similarity                 │
│   • Fast semantic search (millisecond-range at 10k units)        │
│   • Directory: .chromadb/                                        │
│                                                                  │
│   Layer 4: MemoryGraph (NetworkX → JSON)                         │
│   ────────────────────────────────────                           │
│   • Typed directed graph (nodes=units, edges=relations)          │
│   • Powers multi-hop path-based retrieval                        │
│   • Source of read-head training data                            │
│   • File: memory_graph.json                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

Layers 2, 3, and 4 are all populated from the MCM write head output. They are redundant by design: Layer 2 is authoritative for metadata and supersession; Layer 3 is optimized for similarity search; Layer 4 is optimized for structured multi-hop reasoning.

---

## 6. The Three Data Paths

### 6.1 Write Path — Agent Turn Ingestion

```
Agent Code
   │
   │  pipeline.add_turn(role="user", content="...", session_id="s1")
   ▼
MemoryPipeline.add_turn()
   │
   │  1. Construct RawTurn(role, content, session_id, user_id, turn_index, timestamp)
   │
   ▼
EventLog.append(turn)
   │
   │  2. zlib.compress(content)
   │  3. SQLite INSERT INTO events ...
   │  4. Return event_id
   ▼
     ← returns to agent in <1ms (no LLM, no network, no vector index)
```

The ConsolidationWorker runs independently and reads from the EventLog at its own pace. The `consolidation_offsets` table tracks the high-water mark — the worker can handle crashes and restarts safely because `mark_consolidated()` is an upsert.

---

### 6.2 Consolidation Path — Background MCM Processing

```
ConsolidationWorker.run_forever(user_ids)
   │
   │  (every poll_interval_seconds, per user)
   ▼
event_log.unconsolidated_count(user_id)
   │  if count < turn_threshold → skip
   ▼
event_log.get_unconsolidated(user_id, batch_size=100)
   │  returns (turns: list[RawTurn], last_event_id: int)
   ▼
mcm.consolidate(raw_log_text)           ← ONE LLM call for the whole batch
   │
   │  Returns:
   │  {
   │    "memory": {
   │      "facts":                [...],
   │      "preferences":          [...],
   │      "task_summaries":       [...],
   │      "reusable_abstractions":[...],
   │      "open_threads":         [...],
   │      "corrections":          [...]
   │    },
   │    "graph_edges": [
   │      {"from_content": "...", "to_content": "...", "relation": "..."}
   │    ],
   │    "retrieval_examples": [...]
   │  }
   │
   ├─► _write_mcm_output()
   │       │
   │       ├── memory_store.write_unit()      for each unit above confidence threshold
   │       │       └── marks old unit is_superseded=1 if supersedes provided
   │       │
   │       ├── vector_index.upsert()          for each unit (semantic search)
   │       │
   │       └── knowledge_graph.add_unit()     for each unit (graph node)
   │
   ├─► _write_graph_edges()
   │       │
   │       └── knowledge_graph.add_edge()     resolves content snippets → unit IDs
   │
   ├─► generate_read_training_data(graph, user_id)
   │       │
   │       └── read_training_accumulator.add_examples()
   │
   ├─► knowledge_graph.save()
   │
   ├─► (if accumulator.should_trigger_training())
   │       └── accumulator.flush_to_file()    writes train.jsonl + val.jsonl
   │
   └─► event_log.mark_consolidated(user_id, last_event_id)
```

---

### 6.3 Read Path — Retrieval

```
Agent Code
   │
   │  pipeline.retrieve(query="What candle size for breakout?", top_k=5)
   ▼
HybridRetriever.retrieve()
   │
   │  Tier 1: Semantic Search
   ├─► vector_index.search(query, user_id, n_results=top_k, where={filters})
   │       └── ChromaDB cosine similarity query
   │
   │  Enrich with MemoryStore metadata + filter out superseded units
   ├─► memory_store.get_unit_by_id(unit_id) for each result
   │
   │  Check top-1 score
   ├─► if score < FALLBACK_THRESHOLD (0.65):
   │       │
   │       │  Tier 2: Raw Log Fallback
   │       └─► _fallback_search(query, user_id)
   │               └── event_log.get_session_log() + keyword match
   │
   └─► merge, deduplicate by text, sort by score descending
           │
           └── return list[RetrievalResult(unit_id, text, score, source, ...)]

                        ↓ (optional graph routing)
           GraphRouter.route(query, user_id)
               │
               │  count = knowledge_graph.node_count(user_id)
               │
               ├── Tier 1 (no LoRA, count < 200)
               │       full_schema = graph.get_schema_text(user_id)
               │       mcm.route_query_base(query, full_schema)
               │
               ├── Tier 2 (no LoRA, count ≥ 200)
               │       node_ids = vector_index.search(query, top_n=50)
               │       filtered_schema = graph.get_filtered_schema_text(user_id, node_ids)
               │       mcm.route_query_base(query, filtered_schema)
               │
               └── Tier 3 (read LoRA trained, any graph size)
                       node_ids = vector_index.search(query, top_n=50)
                       filtered_schema = graph.get_filtered_schema_text(user_id, node_ids)
                       mcm.route_query(query, filtered_schema)   ← LoRA active
               │
               └── RoutingResult(path, answer_from, confidence, tier)
           knowledge_graph.traverse_path(result.path)
               └── follows node IDs → returns content at each hop
```

The graph routing path (via `GraphRouter`) is called for complex multi-hop queries where the agent explicitly needs structured reasoning across related memory units. Simple factual lookups go directly through the hybrid retriever. Both tier-2 and tier-3 use vector similarity to pre-select the top-50 most relevant nodes before injecting the schema — the prompt size is always bounded regardless of total graph size.

---

## 7. The Dual-Head MCM Model

### Overview

Both heads share a single loaded instance of Qwen2.5-1.5B-Instruct with 4-bit NF4 quantization. Each head is a separate LoRA adapter loaded via PEFT. Adapter swapping is done in-process via `peft_model.set_adapter("write" | "read")`.

```
 Qwen2.5-1.5B-Instruct (frozen, 4-bit NF4)
  ├── LoRA adapter: write  (r=16, alpha=32, dropout=0.05)
  │     Training data: (conversation log → memory JSON + graph edges + retrieval examples)
  │     Source: scripts/generate_data.py → GPT-4o-mini/GPT-4o synthetic pairs
  │
  └── LoRA adapter: read   (r=16, alpha=32, dropout=0.05)
        Training data: (query + graph schema text → traversal path + answer node IDs)
        Source: AUTO-GENERATED by the write head during each consolidation cycle
```

### Write Head

**Input:** Raw interaction log (formatted as `"role: content\n"` blocks)

**System prompt:** Instructs the model to output a single JSON object with `memory`, `graph_edges`, and `retrieval_examples` keys.

**Output structure:**
```json
{
  "memory": {
    "facts":                [{"content": "...", "confidence": 0.9, "domain": "...", "tags": [...]}],
    "preferences":          [...],
    "task_summaries":       [...],
    "reusable_abstractions":[...],
    "open_threads":         [...],
    "corrections":          [{"original": "...", "corrected_to": "..."}]
  },
  "graph_edges": [
    {"from_content": "trades NIFTY", "to_content": "uses breakout strategy", "relation": "uses_strategy"}
  ],
  "retrieval_examples": [
    {"query": "What strategy does the user use for NIFTY?", "path_description": "fact→fact via uses_strategy", "answer_units": ["..."]}
  ]
}
```

**Fallback:** If JSON parsing fails, `_generate()` applies a regex `\{.*\}` extraction attempt before returning `{}`.

### Read Head

**Input:** Natural language query + compact graph schema text (from `MemoryGraph.get_schema_text()`)

**System prompt:** Instructs the model to output a JSON traversal path through the knowledge graph.

**Output structure:**
```json
{
  "reasoning":   "User asks about candle size for breakout. Node m3 (breakout setup) connects to m7 (15m candle) via configured_with.",
  "path":        ["m3", "m7"],
  "answer_from": ["m7"],
  "confidence":  0.92
}
```

**What the read LoRA learns:** The spatial structure of each user's memory graph — which node IDs connect via which relation types. This is fundamentally different from learning facts (which stay in the graph nodes). The LoRA memorizes the *map*, not the territory.

**What the read LoRA does not do:** See the full graph. At any scale, the router first uses vector similarity to pull the top-50 most relevant nodes and injects only that filtered subgraph into the prompt. The LoRA routes within that bounded context. Its advantage over the base model is routing *accuracy* within the subgraph — it has learned this user's graph patterns during training, so it produces fewer hallucinated paths through nodes that exist in the schema but are not actually connected.

**Stale node IDs:** The read LoRA is trained on historical paths. Nodes get superseded over time: when a contradiction is detected, the old unit is marked `is_superseded=1` but is *not removed* from the MemoryGraph (removal would break the audit trail). This means the LoRA may output a path containing the ID of a superseded node. `traverse_path()` handles this by silently skipping any node ID not present in the graph — it guards each hop with `if node_id in self.graph`. In practice, superseded nodes remain in the graph structure (they are never removed by the supersession process), so the returned content may be stale. Callers must cross-reference traversal results against `MemoryStore.get_unit_by_id()` and discard any unit where `is_superseded=1`. If the entire path resolves to an empty list (all IDs are missing or stale), the caller should fall back to vector retrieval results.

### Training Scripts

```bash
# Write head — trained on synthetic data generated by scripts/generate_data.py
python scripts/train_mcm.py --head write --data_dir data/synthetic --output_dir models/mcm_write

# Read head — trained on data generated by ConsolidationWorker at runtime
python scripts/train_mcm.py --head read --data_dir data/read_training --output_dir models/mcm_read
```

Both heads use the same QLoRA config: 4-bit NF4, r=16, alpha=32, dropout=0.05, target modules `q_proj v_proj`. Write head uses `max_seq_length=4096`, read head uses `max_seq_length=2048` (graph schema + short path output).

---

## 8. The Self-Improving Training Loop

This is the architectural feature that distinguishes MCM from all static-model memory systems. The write head auto-generates the training data for the read head as a side effect of every consolidation cycle.

```
Cycle 1 (consolidation):
  EventLog turns → MCM write head → MemoryStore + VectorIndex
                                  → MemoryGraph (new nodes + edges)
                                  → retrieval_examples (query, path, answer) triples
  ReadTrainingAccumulator: 0 → N examples

Cycle 2 (consolidation):
  ... same ...
  ReadTrainingAccumulator: N → 2N examples

...

At threshold (default 500 examples):
  ReadTrainingAccumulator.flush_to_file()
    → data/read_training/train.jsonl  (80%)
    → data/read_training/val.jsonl    (20%)
    → triggers: python scripts/train_mcm.py --head read

  Read LoRA is now fine-tuned on actual paths through this user's memory graph.
  Next retrieve() calls that hit mcm.route_query() benefit from the improved model.
```

The feedback loop:
- The more turns the agent processes → the richer the graph → the more training examples → the better the read head
- The better the read head → the more accurately it routes complex multi-hop queries → the more useful the agent memory context

No human labelling is required after the initial write-head synthetic data generation.

---

## 9. Knowledge Graph Design

### Why a Graph, Not Just Flat Vectors

Flat vector search answers "what is semantically similar to this query?" A knowledge graph answers "how does X relate to Y?" These are different retrieval modes for different query types:

| Query type | Best retrieval mechanism |
|---|---|
| "What's the user's preferred coding language?" | Vector similarity (single fact lookup) |
| "What candle size does the user use for their breakout strategy?" | 2-hop graph path: `preference→abstraction→configured_with→preference` |
| "What tasks are currently open in the trading domain?" | Metadata-filtered MemoryStore query |

The graph enables the third retrieval mode — traversal — which neither mem0 nor standard RAG can express.

### Node and Edge Structure

```
Node attributes:
  unit_id   : str          — UUID, links back to MemoryStore
  user_id   : str          — isolates users in shared graph
  type      : str          — fact | preference | task_summary | abstraction | open_thread
  content   : str          — unit text (full, not truncated in graph storage)
  created_at: str          — ISO timestamp

Edge attributes:
  relation  : str          — semantic relation type (uses_strategy, configured_with, follows_from, related_to, ...)
```

### Schema Text for LLM Input

Two methods produce schema text in the same format:

- `get_schema_text(user_id)` — full graph, used only for tiny graphs (tier-1 routing, < 200 nodes).
- `get_filtered_schema_text(user_id, node_ids)` — restricted to a given set of node IDs, with only edges between those nodes. Used for all routing at scale (tier-2 and tier-3).

Example output:
```
[m1:fact:"trades NIFTY options"]
[m2:fact:"uses breakout strategy"]
[m3:preference:"prefers 15m candle for entry"]
[m2:abstraction:"breakout setup rules"]
[m1] --uses_strategy--> [m2]
[m2] --configured_with--> [m3]
```

At 100k+ nodes, the full schema would be megabytes — far beyond any model's context window. `GraphRouter` always calls `vector_index.search()` first to select the top-50 most relevant nodes, then calls `get_filtered_schema_text()` with those IDs. The router (base model or LoRA) only sees the bounded subgraph.

---

## 10. GDPR and Data Deletion

`MemoryPipeline.delete_user()` performs a hard delete across all four storage layers atomically (best-effort; each layer is independent):

```python
event_log.delete_user(user_id)           # DELETE FROM events WHERE user_id=?
                                          # DELETE FROM consolidation_offsets WHERE user_id=?
memory_store.delete_user_memory(user_id) # DELETE FROM memory_units WHERE user_id=?
vector_index.delete_user(user_id)        # client.delete_collection(f"user_{user_id}")
knowledge_graph.clear_user(user_id)      # remove all nodes with user_id attribute
knowledge_graph.save()                   # persist the modified graph
```

After deletion: the user has no presence in any layer. New turns from the same user_id will start fresh.

---

## 11. mem0 Failure Mode Resolution

This table maps each reported production failure in mem0 (issue #4573) to the specific MCM design choice that prevents it.

| mem0 Failure Mode | MCM Design Choice | Evidence |
|---|---|---|
| **20s write latency** (LLM on every `add()`) | `add_turn()` appends to SQLite only. No LLM on write path. | `EventLog.append()` — single INSERT, no network call |
| **Accuracy loss** (66.9% vs 72.9%) due to raw log discarded | EventLog retained permanently. Tier-2 fallback activated when ChromaDB score < 0.65 | `HybridRetriever._fallback_search()` |
| **Staleness** (old contradicted facts not removed) | New units supersede old ones. `is_superseded=1` excludes stale results from all queries. | `MemoryStore.write_unit(supersedes=[...])` |
| **N+1 LLM cost** (one call per turn) | One MCM call per `batch_size` turns (default 100). | `ConsolidationWorker._consolidate_user()` |
| **Metadata length limit** causing silent data corruption | Metadata fields truncated to 500 chars at write time. EventLog metadata truncated to 4096 chars. | `VectorIndex.upsert()` + `EventLog.append()` |
| **Distributed write conflicts** | SQLite WAL mode allows concurrent readers + single writer. Large deployments should front with a queue. | `PRAGMA journal_mode=WAL` in both DBs |
| **No structured relationship support** | MemoryGraph captures typed edges. Multi-hop queries follow explicit paths. | `MemoryGraph.traverse_path()` |
| **Static model, no improvement** | Read LoRA auto-trained on write-head output. Improves continuously as graph grows. | `ReadTrainingAccumulator.flush_to_file()` |

---

## 12. Deployment Notes

### Minimal deployment (single process)

```python
from src.agent.memory_pipeline import MemoryPipeline

pipeline = MemoryPipeline.create(
    user_id="user-123",
    mcm_model_path="Qwen/Qwen2.5-1.5B-Instruct",       # no adapters = base model
    write_adapter_path="models/mcm_write",               # optional, defaults to base
    read_adapter_path="models/mcm_read",                 # optional, defaults to base
    event_log_path="data/events.db",
    memory_db_path="data/memory.db",
    chromadb_path="data/.chromadb",
    graph_path="data/memory_graph.json",
)

# On every agent turn:
pipeline.add_turn(role="user", content="...", session_id="session-1")
pipeline.add_turn(role="assistant", content="...", session_id="session-1")

# Background: fire consolidation (non-blocking)
pipeline.consolidate_async()

# On retrieval:
results = pipeline.retrieve("What does the user prefer for breakout entries?", top_k=5)
context = pipeline.format_context(results)
```

### Storage scaling

| Layer | Scale ceiling (single node) | Scale-out option |
|---|---|---|
| EventLog | ~100M events (SQLite) | PostgreSQL with WAL, Kafka for multi-writer |
| MemoryStore | ~10M units (SQLite) | PostgreSQL, partitioned by user_id |
| VectorIndex | ~1M units/collection (ChromaDB HNSW) | Milvus, Qdrant, Weaviate |
| MemoryGraph | ~10M nodes (NetworkX in-memory) | Neo4j, Amazon Neptune |

### GPU requirements

| Mode | Hardware | Notes |
|---|---|---|
| Inference only (4-bit) | 4GB VRAM | Qwen2.5-1.5B in NF4 fits on a T4 |
| QLoRA training (write head) | 16GB VRAM | 4096 seq len, batch=4 |
| QLoRA training (read head) | 8GB VRAM | 2048 seq len, shorter inputs |
| CPU-only inference | ~3GB RAM | Slow (~30s/call), acceptable for background consolidation |

### File layout

```
data/
  events.db               EventLog (SQLite)
  memory.db               MemoryStore (SQLite)
  .chromadb/              VectorIndex (ChromaDB)
  memory_graph.json       MemoryGraph (NetworkX serialized)
  read_training/
    train.jsonl           Auto-generated read-head training data
    val.jsonl

models/
  mcm_write/              Write LoRA adapter checkpoint
  mcm_read/               Read LoRA adapter checkpoint
```
