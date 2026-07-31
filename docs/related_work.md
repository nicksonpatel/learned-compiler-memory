# Related Work Map

Quick reference for paper writing. For each system, note what they do and how MCM differs.

---

## Memory Systems for LLM Agents

| System | Method | Limitation vs. MCM |
|---|---|---|
| **MemGPT** (Packer et al., 2023) | Prompt-engineered memory management with hierarchical context | Not learned — relies on the main LLM to manage its own memory via prompting |
| **Generative Agents** (Park et al., 2023) | Reflection + retrieval over event streams | Summarization is prompt-based, not a trained function; no structured schema |
| **mem0** (Taranjeet et al., 2024) | RAG + LLM extraction pipeline | Memory formation is prompt-based extraction, not a fine-tuned dedicated model |
| **SuperMemory** | Retrieval-focused knowledge management | Retrieval system, not a memory formation system |
| **A-MEM** (2024) | Zettelkasten-inspired dynamic linking of memories | Note linking heuristic, not learned transformation |
| **Cognitive Architectures for LLMs** (Sumers et al., 2023) | Survey of memory types in LLM agents | Survey, not a system |

---

## Learned / RL-Trained Memory Management (2025–2026)

Newer than the systems above, and closer to what MCM is actually doing: instead of a large LLM prompted to manage memory, these train a dedicated policy for memory operations. Mostly via RL rather than SFT — this is the category MCM most needs to differentiate against.

| System | Method | Relation to MCM |
|---|---|---|
| **Memory-R1** (Yan et al., 2025) — arXiv:2508.19828 | RL (PPO/GRPO)-trained Memory Manager (ADD/UPDATE/DELETE/NOOP) + Answer Agent; reward = downstream QA correctness | Same write/read split intuition as MCM's dual-head design, but the policy is RL-trained end-to-end on outcome reward, not SFT-distilled from a teacher on synthetic (log→memory) pairs |
| **Mem-α** (2025) — arXiv:2509.25911 | RL for learning memory *construction* directly | Targets the same job as MCM's write head; worth comparing SFT-then-freeze (MCM) vs. RL fine-tune (Mem-α) for write-head quality |
| **LightMem** (Fang et al., ICLR 2026) — arXiv:2510.18866 | Three small LMs (Controller / Selector / Writer) for online memory ops; consolidation decoupled from the online path | Closest architectural cousin to MCM — independently validates the "no LLM on the critical path, dedicated small model per stage" design choice |
| **GraphRAG-Router** (2026) — arXiv:2604.16401 | RL-trained router choosing between GraphRAG and a plain LLM per query, optimizing cost/accuracy | Solves the same problem as MCM's read head (learned routing over graph-structured memory), but via RL against a cost/accuracy reward instead of SFT on self-generated path labels |
| **MemRouter** (2026) — arXiv:2605.00356 | Memory-as-embedding routing for long-term conversational agents | Alternative read-side design: routes via learned embeddings rather than a graph-traversal LoRA |
| **Agentic Memory** (2026) — arXiv:2601.01885 | Unified long-term/short-term memory management with learned prioritization and forgetting | Adjacent capability MCM does not yet have — current design has no learned decay/forgetting mechanism |

**Where MCM differs:** every RL-trained system above needs an environment that can score outcomes (QA correctness, routing cost/accuracy) to train against. MCM instead uses **teacher distillation + a self-generated supervision loop** — the write head (SFT on teacher-labeled log→memory pairs) produces structured memory over time, and that structure is mined for (query, path, answer) triples that train the read head, with no RL infrastructure or reward model required. This is cheaper to bootstrap, but the tradeoff — already flagged in `docs/improvement_plan.md` — is that self-generated, templated supervision may teach path completion rather than true discrimination between plausible paths. That is exactly the failure mode RL-based approaches (Memory-R1, Mem-α, GraphRAG-Router) sidestep by training directly against outcomes. This tension is worth stating explicitly in the paper's related work section, and worth testing empirically: does an RL fine-tuning pass on top of the SFT-warm-started read head close the gap that `improvement_plan.md` identifies?

---

## Continual Learning / Fine-tuning

| Paper | Key Insight | Relevance |
|---|---|---|
| **LoRA** (Hu et al., 2022) | Low-rank adaptation for efficient fine-tuning | Basis for our user-specific adapter approach |
| **QLoRA** (Dettmers et al., 2023) | 4-bit quantized LoRA — fine-tune 65B on 1 GPU | Training method we adopt |
| **Continual Learning Survey** (De Lange et al., 2022) | Overview of catastrophic forgetting mitigation | Motivates why we avoid full fine-tuning |

---

## Structured Memory / Knowledge Representation

| Paper | Key Insight | Relevance |
|---|---|---|
| **Think-on-Graph** (Sun et al., 2023) | LLM reasoning over knowledge graphs | Structured output retrieval pattern |
| **HippoRAG** (Guo et al., 2024) | RAG inspired by human hippocampal memory | Hybrid retrieval approach comparison |
| **KnowAgent** (Zhu et al., 2024) | Action knowledge graph for agent planning | Structured abstraction for reuse pattern |

---

## Knowledge Distillation (related to MCM training)

| Paper | Relevance |
|---|---|
| **Distilling the Knowledge in a Neural Network** (Hinton, 2015) | Teacher-student training — GPT-4o → MCM |
| **Self-Play Fine-Tuning (SPIN)** (Chen et al., 2024) | Can generate own training data iteratively |

---

## Benchmark Datasets

| Dataset | Use case |
|---|---|
| ShareGPT | Real human-agent conversations — can be converted to (log → memory) |
| WizardLM | Instruction following — scenarios for memory tasks |
| LoCoMo (Maharana et al., 2024) | Long-form conversation memory benchmark — key eval dataset |
| **LoCoMo is most relevant** — explicitly evaluates memory over long conversations |

---

## Key Papers to Read First (priority order)

1. MemGPT (Packer et al., 2023) — arXiv:2310.08560
2. Generative Agents (Park et al., 2023) — arXiv:2304.03442
3. LoCoMo benchmark (Maharana et al., 2024) — arXiv:2402.17753
4. QLoRA (Dettmers et al., 2023) — arXiv:2305.14314
5. Cognitive Architectures for LLMs (Sumers et al., 2023) — arXiv:2309.02427
6. Memory-R1 (Yan et al., 2025) — arXiv:2508.19828 — closest competing system; read before writing the paper's related-work/positioning section
7. LightMem (Fang et al., ICLR 2026) — arXiv:2510.18866 — closest architectural cousin (small models, decoupled online/offline path)
8. Mem-α (2025) — arXiv:2509.25911 — RL alternative to MCM's SFT write head; relevant to the read-head training-mismatch problem in `docs/improvement_plan.md`
