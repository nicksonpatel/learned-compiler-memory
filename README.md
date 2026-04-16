# Learned Memory Consolidation for LLM Agents

> **Core thesis:** Memory formation is a *learned transformation problem*, not a retrieval or prompting problem.

Instead of prompting a large LLM to summarize and store memory, we train a small dedicated model (1–3B params) whose sole job is to transform raw agent interaction logs into structured, queryable memory units. This model is decoupled from the main reasoning LLM and acts as a portable, plug-and-play memory bank.

---

## Architecture

```
Raw Interaction Logs (chat, tool calls, outcomes)
            │
            ▼
  ┌─────────────────────┐
  │  Memory Model (MCM) │  ← small pretrained + LoRA fine-tuned
  │  ~1–3B params       │
  └─────────────────────┘
            │
            ▼
  Structured Memory Store
  { facts, preferences, summaries, abstractions }
            │
      ┌─────┴─────┐
      │  Retriever │  ← semantic search over memory store
      └─────┬─────┘
            │
            ▼
   Main Agent LLM (any)  ← reasoning, not memory management
```

**Hybrid pipeline:**
1. New memories → RAG store (immediate, low-latency)
2. At threshold → Memory Consolidation Model refines + distills → structured store
3. User-specific adaptation → LoRA adapter (lightweight, no catastrophic forgetting)

---

## Key Contributions

1. **Learned Memory Consolidation Model (MCM)** — not prompt-engineered
2. **Standardized Memory Schema** — model-agnostic, cross-agent reusable
3. **Separation of Memory and Reasoning** — smaller model handles memory, main LLM reasons
4. **LoRA-based user adaptation** — portable adapters vs. full fine-tuning

---

## Project Structure

```
learned-compiler-memory/
├── docs/
│   ├── paper_outline.md       # Research paper draft outline
│   ├── memory_schema.md       # Memory unit schema specification
│   └── related_work.md        # Literature map
├── src/
│   ├── memory_model/          # Core MCM training & inference
│   ├── data/                  # Synthetic data generation pipeline
│   ├── retriever/             # Memory retrieval layer
│   └── agent/                 # Agent loop integration
├── experiments/
│   ├── exp1_memory_quality/   # Intrinsic memory eval
│   ├── exp2_downstream/       # Task performance eval
│   └── exp3_efficiency/       # Latency & cost benchmarks
├── scripts/                   # Training, eval, and data gen scripts
├── PLAN.md                    # Phased execution roadmap
└── requirements.txt
```

---

## Quick Start (after setup)

```bash
pip install -r requirements.txt

# Generate synthetic training data
python scripts/generate_data.py --n_samples 5000 --output data/synthetic/

# Train memory consolidation model
python scripts/train_mcm.py --base_model Qwen/Qwen2.5-1.5B --data data/synthetic/

# Run evaluation
python scripts/eval_memory_quality.py --model checkpoints/mcm-v1/
```
