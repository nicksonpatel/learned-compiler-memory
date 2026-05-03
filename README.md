# Learned Memory Consolidation for LLM Agents

> **Core thesis:** Memory formation is a *learned transformation problem*, not a retrieval or prompting problem.

Instead of prompting a large LLM to summarize and store memory, we train a small dedicated model (~3B params) whose sole job is to transform raw agent interaction logs into structured, queryable memory units. This model is decoupled from the main reasoning LLM and acts as a portable, plug-and-play memory bank.

---

## Current Status

| Component | Status |
|---|---|
| Synthetic data generation | ✅ Complete — 4,926 samples (train/val/test) via MiniMax API |
| Write head training (`mcm-write-v1`) | ✅ Complete — 3 epochs, 372 steps, eval loss 1.28, token accuracy 66% |
| Write head checkpoint | ✅ `checkpoints/mcm-write-v1/` (base: Qwen2.5-3B-Instruct + QLoRA) |
| Read-training data accumulation | 🔄 In progress — requires write head deployed against live graph |
| Read head training (`mcm-read-v1`) | ⏳ Pending — waiting on read-training data |
| Exp 1: Memory quality eval | ⏳ Pending |
| Exp 2: Downstream task eval | ⏳ Pending |
| Exp 3: Efficiency benchmarks | ⏳ Pending |

---

## Architecture

The MCM system uses two separate LoRA adapters on the same base model (Qwen2.5-3B-Instruct), handling distinct tasks without interference:

```
Raw Interaction Logs (chat, tool calls, outcomes)
            │
            ▼
  ┌─────────────────────────────────────────┐
  │     MCM Write Head (mcm-write-v1)       │  ← QLoRA on Qwen2.5-3B-Instruct
  │  log → structured memory JSON           │    r=16, α=32, all-linear modules
  │       + graph edges                     │
  │       + retrieval training examples     │
  └─────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────┐
  │               Structured Memory Store            │
  │  MemoryStore (SQLite)  +  ChromaDB  +  MemoryGraph│
  └──────────────────────────────────────────────────┘
            │                    │
            │  (generates)       ▼
            │         Read-Training Accumulator
            │                    │ (at threshold)
            │                    ▼
            │     ┌─────────────────────────────────┐
            │     │   MCM Read Head (mcm-read-v1)   │  ← trained on write-head output
            │     │  query + graph → traversal path │    improves over time
            │     └─────────────────────────────────┘
            │                    │
      ┌─────┴────────────────────┘
      ▼
   HybridRetriever
    ├── Tier 1: ChromaDB semantic search
    └── Tier 2: raw EventLog fallback (<1ms, no LLM on write path)
            │
            ▼
   Main Agent LLM (any)  ← reasoning only; memory fully offloaded
```

**Three data paths:**
1. **Write path** — `add_turn()` appends to EventLog in <1 ms (no LLM on the critical path)
2. **Consolidation path** — background worker batches 100 turns → single MCM write-head call → MemoryStore + MemoryGraph + VectorIndex
3. **Read path** — HybridRetriever queries ChromaDB first, falls back to raw EventLog

---

## Key Contributions

1. **Dual-head MCM** — write adapter (log→memory) and read adapter (query→graph path) are separate LoRAs on the same base; trained sequentially, no interference
2. **Write path decoupled from LLM** — `add_turn()` is <1 ms; consolidation runs async in background
3. **Self-improving training loop** — write head auto-generates (query, path, answer) triples that become read-head training data; no human labelling required
4. **Immutable memory with supersession** — contradictions create new units that supersede old ones; full audit trail retained
5. **Standardized Memory Schema** — model-agnostic, cross-agent reusable (see `docs/memory_schema.md`)

---

## Model Details

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Fine-tuning strategy | QLoRA (4-bit NF4, bfloat16 compute) |
| LoRA rank / alpha | r=16 / α=32 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o projections + gate/up/down projections |
| Training epochs | 3 |
| Steps | 372 |
| Final eval loss | 1.28 |
| Final token accuracy | 66% |
| Frameworks | PEFT 0.19.1 · TRL 1.2.0 · Transformers 5.5.4 · PyTorch 2.5.1+cu121 |

---

## Project Structure

```
learned-compiler-memory/
├── docs/
│   ├── architecture.md        # Full system architecture (v0.2)
│   ├── improvement_plan.md    # Design review findings and next steps
│   ├── paper_outline.md       # Research paper draft outline
│   ├── memory_schema.md       # Memory unit schema specification
│   └── related_work.md        # Literature map
├── src/
│   ├── memory_model/          # MCM inference + read-training data generator
│   ├── data/                  # EventLog, MemoryStore, KnowledgeGraph
│   ├── retriever/             # HybridRetriever (ChromaDB + EventLog fallback)
│   └── agent/                 # ConsolidationWorker + MemoryPipeline
├── data/
│   └── synthetic/             # 4,926 generated (log → memory JSON) pairs
├── checkpoints/
│   └── mcm-write-v1/          # Trained write-head LoRA adapter
├── experiments/
│   ├── exp1_memory_quality/   # Intrinsic memory eval (compression, recall, proxy judge)
│   ├── exp2_downstream/       # Task performance eval
│   └── exp3_efficiency/       # Latency & cost benchmarks
├── scripts/                   # Training, data gen, and probe scripts
├── PLAN.md                    # Phased execution roadmap
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic training data (requires MINIMAX_API_KEY)
export MINIMAX_API_KEY=sk-cp-...
python scripts/generate_data.py --n_samples 5000 --output data/synthetic/ --workers 40

# Train write head
python scripts/train_mcm.py \
    --head write \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --data_dir data/synthetic \
    --output_dir checkpoints/mcm-write-v1

# Train read head (after accumulating read-training data from write head + graph)
python scripts/train_mcm.py \
    --head read \
    --data_dir data/read_training \
    --output_dir checkpoints/mcm-read-v1

# Run memory quality evaluation
python experiments/exp1_memory_quality/eval.py \
    --mcm_model checkpoints/mcm-write-v1 \
    --test_data data/synthetic/test.jsonl \
    --output results/exp1.json
```
