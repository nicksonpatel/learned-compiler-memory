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
