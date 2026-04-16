# Paper Outline: Learning to Remember

**Working title:** *"Learning to Remember: A Modular Memory Consolidation Model for LLM-based Agents"*

**Target venue:** EMNLP 2026 / NeurIPS 2026

---

## Abstract (draft)

Long-term memory remains a fundamental challenge for LLM-based agents. Existing approaches rely on retrieval-augmented generation (RAG) or prompt-based summarization — neither of which treats memory formation as a learnable function. We introduce the **Memory Consolidation Model (MCM)**, a small (1–3B parameter), dedicated language model trained to transform raw agent interaction histories into structured, queryable memory representations. MCM is decoupled from the main reasoning LLM, operates on a standardized memory schema enabling cross-agent reuse, and supports user-specific adaptation via lightweight LoRA adapters. We demonstrate that MCM outperforms RAG-only and prompt-summarization baselines in memory quality, downstream task performance, and inference efficiency. Our results suggest that treating memory formation as a learned transformation — rather than a retrieval or prompting problem — yields meaningful improvements in agent long-term coherence.

---

## 1. Introduction

- Hook: agents forget — this is a fundamental and underaddressed problem
- Current solutions and their limits (prompt engineering, RAG)
- Our claim: memory formation is a *learned* function
- Paper contributions (numbered, clear)
- Overview of results

**Contributions:**
1. MCM: a dedicated model for memory consolidation trained via supervised distillation
2. A standardized memory schema enabling plug-and-play cross-agent memory reuse
3. A hybrid pipeline (RAG + consolidation) with user-specific LoRA adapters
4. Evaluation metrics for memory quality and three empirical experiments
5. Evidence that separating memory from reasoning improves both

---

## 2. Related Work

- 2.1 Memory in LLM agents (MemGPT, Generative Agents, mem0)
- 2.2 Retrieval-augmented generation
- 2.3 Parameter-efficient fine-tuning (LoRA, QLoRA)
- 2.4 Continual learning and catastrophic forgetting
- 2.5 Knowledge distillation

**Key positioning sentence:** *"Unlike prior work that treats memory formation as a prompting or retrieval problem, MCM treats it as a learned transformation, enabling structured, reusable, and adaptable memory representations without modifying the main reasoning model."*

---

## 3. Method

### 3.1 Problem Formulation

Given a history of agent interactions $H = \{t_1, t_2, ..., t_n\}$, we want a function $f: H \rightarrow M$ where $M$ is a structured memory representation. We train $f$ as a small language model rather than engineering it as a prompt.

### 3.2 Memory Schema

- Define the schema (see `docs/memory_schema.md`)
- Justify each field type
- Explain merge/update strategy

### 3.3 MCM Architecture

- Base model selection (Qwen2.5-1.5B-Instruct)
- Training objective: next-token prediction over gold memory JSON
- System prompt design
- QLoRA training details

### 3.4 User-Specific Adaptation

- Why LoRA adapters vs. full fine-tuning
- Adapter training protocol
- How adapters are stored/loaded (portability)

### 3.5 Hybrid Pipeline

- Phase 1: RAG (immediate, low-latency)
- Phase 2: Consolidation trigger (threshold N turns or timer)
- Phase 3: MCM inference → structured store update
- Retrieval at query time: structured field lookup + semantic search

---

## 4. Training Data

### 4.1 Synthetic Data Generation

- GPT-4o as teacher: (scenario → log → gold memory)
- Diversity: 10 domains, 5 session lengths, 3 user personas
- Quality filtering pipeline

### 4.2 Real Data Adaptation

- ShareGPT → log conversion
- LoCoMo benchmark usage

### 4.3 Dataset Statistics

(fill in after data generation)

---

## 5. Experiments

### 5.1 Experiment 1: Memory Quality (Intrinsic)

- Setup, baselines, metrics
- Results table
- Human eval results

### 5.2 Experiment 2: Downstream Task Performance

- Tasks description
- Baselines comparison table
- Key findings

### 5.3 Experiment 3: Efficiency

- Latency, context compression, cost tables

### 5.4 Ablations

- MCM without pretraining stage
- MCM without LoRA adapter (base model only)
- MCM with different base model sizes (1B vs 3B vs 7B)
- Memory schema with fewer fields

---

## 6. Analysis

- Qualitative examples: MCM output vs. baselines
- Error analysis: what MCM gets wrong
- Adapter analysis: how much user-specific data is needed

---

## 7. Limitations

- Staleness window between consolidation cycles
- Synthetic training data may not generalize to all domains
- Privacy: memories in LoRA weights (harder to delete than DB entries)
- Compute cost of consolidation for very high-frequency interactions

---

## 8. Conclusion

- Restate the key claim
- Summarize results
- Future work: online consolidation, cross-user memory sharing, multimodal memory

---

## Figures Needed

1. System architecture diagram (pipeline figure)
2. Memory schema example (before/after: raw log → structured JSON)
3. Experiment 1 results bar chart
4. Experiment 2 results table
5. Efficiency comparison (latency + cost scatter plot)
6. Ablation results table
