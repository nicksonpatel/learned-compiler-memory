# Execution Plan: Learned Memory Consolidation Model (MCM)

**Goal:** Build and validate a small, specialized LLM that transforms raw agent interaction logs into structured memory — then publish as a research paper.

---

## Phase 0: Foundations (Week 1)

**Decisions to lock before coding anything:**

| Decision | Options | Recommendation |
|---|---|---|
| Base model | Qwen2.5-1.5B, Phi-3-mini, Llama-3.2-1B | **Qwen2.5-1.5B** — best instruction following at size |
| Fine-tuning strategy | Full fine-tune, LoRA, QLoRA | **QLoRA** — 4-bit, fits on single GPU, no catastrophic forgetting |
| Memory store | JSON files, SQLite, ChromaDB | **SQLite + ChromaDB** — structured fields + semantic search |
| Agent framework for integration | LangChain, raw, CrewAI | **Raw first** — avoid framework lock-in during research |

**Deliverables:**
- [ ] Lock all decisions above
- [ ] Read core papers (list in `docs/related_work.md`)
- [ ] Define memory schema v1 (see `docs/memory_schema.md`)
- [ ] Set up dev environment & repo

---

## Phase 1: Data Pipeline (Week 2–3)

The entire project lives or dies on training data quality. This is the most important phase.

### 1a. Define the training format

Each training example is a pair:
```
Input:  raw conversation/interaction log (N turns)
Output: structured memory JSON
```

### 1b. Synthetic data generation

Use GPT-4o or Claude to generate (log → memory) pairs at scale.

**Strategy:**
1. Generate diverse "agent session" scenarios (personal assistant, coding, trading, research, planning)
2. For each scenario, generate a realistic multi-turn log
3. Have the generator also produce the "ideal memory" output (gold label)
4. Human spot-check 10% for quality

**Target:** 5,000–10,000 high-quality pairs for initial training. 50,000+ for a stronger paper.

**Script:** `scripts/generate_data.py`

### 1c. Real data collection (if possible)

- Public conversation datasets: ShareGPT, WizardLM, Alpaca — can be converted
- Synthetic agent trajectories via self-play (agent acts, records logs)

**Deliverables:**
- [ ] `scripts/generate_data.py` working
- [ ] 5,000 synthetic training examples
- [ ] Data quality review + filtering pipeline
- [ ] Train/val/test split (80/10/10)

---

## Phase 2: Model Training (Week 3–4)

### 2a. Pretraining stage (optional but recommended)

If you have compute budget, do a short continued pretraining on memory-domain text:
- Memory psychology papers
- Agent interaction logs
- Knowledge graph serialization formats

This gives the model a prior for what "memory" looks like before fine-tuning.

> **Shortcut:** Skip pretraining v1, jump straight to instruction fine-tuning. Add pretraining as an ablation in the paper.

### 2b. Instruction fine-tuning with QLoRA

```
Base:   Qwen2.5-1.5B-Instruct
Method: QLoRA (4-bit NF4 quantization, r=16, alpha=32, dropout=0.05)
Epochs: 3–5
LR:     2e-4 with cosine decay
Batch:  4 (with gradient accumulation ×8 = effective 32)
```

**Script:** `scripts/train_mcm.py`

### 2c. User-specific LoRA adapters

After base MCM is trained:
- Each user/agent gets their own LoRA adapter (~50MB)
- Fine-tune adapter on user's own memory history
- Adapter is the "portable memory bank" — plug-and-play

**Deliverables:**
- [ ] Training script with QLoRA
- [ ] Base MCM checkpoint
- [ ] User adapter training script
- [ ] W&B training logs

---

## Phase 3: Agent Integration (Week 5)

Build the full pipeline connecting MCM to an agent loop.

```python
class MemoryConsolidationPipeline:
    def __init__(self, mcm_model, rag_store, threshold=50):
        self.mcm = mcm_model
        self.rag = rag_store          # immediate writes go here
        self.threshold = threshold    # turns before consolidation

    def add_interaction(self, turn):
        self.rag.add(turn)
        if len(self.rag) >= self.threshold:
            self.consolidate()

    def consolidate(self):
        raw_log = self.rag.get_recent(self.threshold)
        structured = self.mcm.generate(raw_log)   # MCM inference
        self.rag.update_structured(structured)
        self.rag.prune_raw()

    def retrieve(self, query):
        return self.rag.search(query)
```

**Deliverables:**
- [ ] `src/agent/memory_pipeline.py`
- [ ] Integration test with a simple LangChain/raw agent
- [ ] Demo notebook showing the full loop

---

## Phase 4: Experiments (Week 6–7)

Three layers of validation. Run all three for a credible paper.

### Experiment 1: Memory Quality (Intrinsic)

**Goal:** Does MCM produce better memory than naive baselines?

| Baseline | Method |
|---|---|
| Raw truncation | Just cut old context |
| Rolling summary | GPT prompted to summarize |
| RAG-only | Embedding-based retrieval |
| **MCM (yours)** | Learned consolidation |

**Metrics:**
- Human eval: usefulness / completeness / accuracy (1–5 scale, 100 examples, 3 annotators)
- Compression ratio: `tokens_before / tokens_after`
- Information retention: key fact recall rate (automatic, using a judge LLM)

**Script:** `experiments/exp1_memory_quality/eval.py`

### Experiment 2: Downstream Task Performance

**Goal:** Does better memory → better agent task success?

**Tasks (pick 2–3):**
1. Personalized assistant — agent must recall user preferences across sessions
2. Multi-step research — agent must remember prior findings and avoid re-querying
3. Coding continuity — agent recalls prior design decisions across sessions

**Metrics:**
- Task success rate
- Steps to completion
- Hallucination rate (judge-LLM scored)

**Script:** `experiments/exp2_downstream/eval.py`

### Experiment 3: Efficiency

**Goal:** Prove this is practical, not just accurate.

**Metrics:**
- Context size reduction (tokens saved per session)
- Latency: MCM inference time vs. prompt summarization latency
- Cost: MCM (local inference cost) vs. GPT-based summarization (API cost)

**Script:** `experiments/exp3_efficiency/eval.py`

**Deliverables:**
- [ ] All three experiments runnable end-to-end
- [ ] Results tables (match paper format)
- [ ] Ablation: MCM vs. MCM without pretraining / without LoRA adapter

---

## Phase 5: Paper Writing (Week 8–9)

**Target venues (in order of fit):**
1. EMNLP 2026 — strong fit (NLP + agents)
2. NeurIPS 2026 — broader, higher reach
3. ICLR 2027 — if results are very strong
4. arXiv preprint — ship ASAP regardless

**Paper structure:**
1. Abstract
2. Introduction — the memory-as-learned-transformation frame
3. Related Work — MemGPT, mem0, Generative Agents, RAG, PEFT
4. Method — MCM architecture, training, schema, hybrid pipeline
5. Experiments — 3 experiments above
6. Analysis & Ablations
7. Limitations — privacy, staleness window, compute cost
8. Conclusion

**Working title:** *"Learning to Remember: A Modular Memory Consolidation Model for LLM-based Agents"*

**Deliverables:**
- [ ] Full paper draft in LaTeX
- [ ] Camera-ready figures (architecture diagram, results tables)
- [ ] arXiv preprint submitted

---

## Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Synthetic data doesn't generalize | High | Validate on real ShareGPT data; add self-play agent data |
| MCM output quality poor | High | Start with GPT-4o as teacher, distill |
| No clear win over RAG baseline | Medium | Focus on structured output quality, not just recall |
| Catastrophic forgetting in fine-tuning | Low | Using QLoRA — base weights frozen |
| Privacy / GDPR (memories in weights) | Medium | Acknowledge in paper; LoRA adapters can be deleted |
| Reviewers say "just summarization" | Medium | Emphasize structured schema + cross-agent reuse + learnability |

---

## Compute Requirements

| Task | GPU | Time estimate |
|---|---|---|
| Synthetic data gen (10k samples) | CPU/API | ~4 hours |
| QLoRA fine-tune (Qwen2.5-1.5B) | 1x RTX 4090 or A100 | ~6–12 hours |
| User adapter fine-tune (per user) | 1x RTX 3090 | ~30 min |
| Experiments 1–3 | 1x A100 | ~8 hours total |

> RunPod or Lambda Labs for cheap A100 access; ~$50–100 total compute budget for MVP.

---

## MVP Success Criteria

The MVP is done when:
1. MCM can take a raw 50-turn conversation and output valid structured memory JSON
2. An agent using MCM memory outperforms an agent using RAG-only on at least 1 downstream task
3. MCM inference adds < 500ms latency per consolidation cycle
