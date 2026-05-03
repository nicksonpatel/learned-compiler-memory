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

### 2b. Instruction fine-tuning with LoRA — via Together AI Serverless

**Platform:** [Together AI Fine-Tuning](https://docs.together.ai/docs/fine-tuning-quickstart) — serverless, no GPU management needed.

```
Base model:   Qwen/Qwen2.5-1.5B-Instruct  (supported natively on Together AI)
Method:       LoRA (r=16, alpha=32, dropout=0.05, all-linear modules)
Epochs:       3
LR:           2e-4 with warmup_ratio=0.05
Batch size:   32 (max for this model on Together AI)
Max context:  32768 tokens
```

**Script:** `scripts/train_together.py`

**Steps:**
```bash
pip install together
export TOGETHER_API_KEY=<your-key>

# 1. Dry-run: upload data only, verify format
python scripts/train_together.py --data_dir data/synthetic --dry_run

# 2. Full run: upload + launch + poll to completion
python scripts/train_together.py \
    --data_dir data/synthetic \
    --epochs 3 \
    --suffix mcm-write-v1
```

Data format is already correct — our `messages` JSONL is what Together AI expects natively.

**Cost Estimate (Together AI LoRA, ≤16B, standard pricing):**

| Component | Tokens | Rate | Cost |
|---|---|---|---|
| Training (train.jsonl × 3 epochs) | ~85M | $0.48/M | ~$41 |
| Evaluation (val.jsonl × 10 evals) | ~35M | $0.48/M | ~$17 |
| **Total training** | ~120M | | **~$58** |
| Serverless inference (experiments) | ~5M | ~$0.10/M | ~$1 |
| Dedicated endpoint (optional, H100) | 2–4 hrs | $3.99/hr | ~$8–16 |
| **Total MVP budget** | | | **~$60–75** |

> Token estimate: train.jsonl=113MB ÷ 4 chars/token ≈ 28M tokens/epoch × 3 = 85M. Val.jsonl=14MB ÷ 4 ≈ 3.5M × 10 evals = 35M.

**Option B — Google Colab (free or cheap):**
`bitsandbytes` QLoRA works on Colab's CUDA GPUs. Qwen2.5-1.5B in 4-bit fits easily on a T4 (15GB).

| Tier | GPU | Cost | Time limit | Notes |
|---|---|---|---|---|
| Free | T4 16GB | $0 | ~4–6 hrs/session, disconnects | Need to save checkpoint frequently |
| Pro ($9.99/mo) | T4 or A100 | ~$10 | Priority access, longer sessions | Best free-ish option |
| Pro+ ($49.99/mo) | A100 40GB | ~$50 | Background running, no disconnect | Overkill for 1.5B model |

```python
# In a Colab cell — mount Drive for persistent storage:
from google.colab import drive
drive.mount('/content/drive')

# Clone repo + install deps:
!git clone https://github.com/your-user/learned-compiler-memory
!pip install -r learned-compiler-memory/requirements.txt

# Upload data, then train:
!python learned-compiler-memory/scripts/train_mcm.py \
    --head write \
    --data_dir /content/drive/MyDrive/mcm/data/synthetic \
    --output_dir /content/drive/MyDrive/mcm/checkpoints/mcm-write-v1 \
    --epochs 3
```

> **Tip:** Use Google Drive to persist checkpoints — Colab runtime resets wipe `/content/`. Free tier training estimate: ~4–6 hours on T4 (split across sessions if needed).

**Option C — local GPU fallback:**
Script `scripts/train_mcm.py` runs QLoRA (4-bit NF4) locally. Requires CUDA GPU.
Estimated cost: ~$10–15 on RunPod A100 40GB (~6–8 hrs × $1.50/hr).
> Note: `bitsandbytes` QLoRA does NOT support Apple Silicon.

### 2c. Inference strategy (per training option)

The trained model needs to run inference for: experiments (Exp 1–3), the agent integration demo, and eventual paper evaluation.

| Training path | Inference option | Latency | Cost | Setup effort |
|---|---|---|---|---|
| **Together AI (A)** | **Serverless API** (OpenAI-compatible, same key) | ~500ms–2s | ~$0.10/M tokens | ✅ Zero |
| Together AI (A) | Dedicated endpoint (H100) | ~100ms | $3.99/hr | Low |
| Colab (B) | Load adapter in Colab, batch inference | ~200ms/sample on T4 | $0 | Medium — must re-load each session |
| Colab (B) | Export to HuggingFace Hub → free Inference API | ~1–3s | $0 (rate-limited) | Medium |
| RunPod (C) | vLLM serving on same pod | ~100ms | ~$0.10/hr T4 | Medium |

**Recommended inference path per option:**

**If trained on Together AI →** inference is automatic. Call via their API:
```python
from together import Together
client = Together()
response = client.chat.completions.create(
    model="your-username/Qwen2.5-1.5B-Instruct-mcm-write-v1-xxxxxx",
    messages=[{"role": "user", "content": log_text}]
)
```

**If trained on Colab →** two sub-options:
1. **For experiments only** (batch, non-production): load the saved checkpoint each Colab session and run inference there. Fine for Exp 1–3 since experiments are scripted batch jobs.
2. **For the demo / agent integration**: push adapter to HuggingFace Hub (free), then load from there:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", ...)
model = PeftModel.from_pretrained(model, "your-hf-username/mcm-write-v1")
```

> **Bottom line:** Together AI serverless is the zero-friction path for inference. Colab-trained models can still run inference cheaply — just load the adapter from HuggingFace Hub. Both work for the experiments. Only Together AI dedicated endpoint or RunPod/vLLM makes sense for low-latency production use.

### 2d. User-specific LoRA adapters

After base MCM is trained:
- Each user/agent gets their own LoRA adapter (~50MB)
- Fine-tune adapter on user's own memory history
- Adapter is the "portable memory bank" — plug-and-play

**Deliverables:**
- [x] Training script for Together AI serverless (`scripts/train_together.py`)
- [x] Training script for local GPU fallback (`scripts/train_mcm.py`)
- [ ] Base MCM checkpoint (output of training job)
- [ ] User adapter training script
- [ ] W&B training logs

---

### 2e. ✅ Preferred Training Path: Lambda Labs or RunPod (SSH, no browser)

Colab is impractical for long runs — the browser tab must stay open and sessions time out. Lambda Labs and RunPod both provide true SSH access with persistent storage, so training runs in a `tmux` session that survives disconnect.

**Model upgrade:** Move from `Qwen2.5-1.5B` → **`Qwen2.5-3B-Instruct`**
- Same tokenizer and chat template (zero data-format changes)
- 2× parameters → meaningfully better JSON quality
- Fits comfortably in 4-bit QLoRA on any 24 GB GPU
- Could go 7B on an A100 40 GB if quality warrants

#### Platform Comparison

| | Lambda Labs | RunPod |
|---|---|---|
| **Best GPU option** | A10 24GB ($0.75/hr) or A100 40GB ($1.10/hr) | RTX 4090 24GB ($0.44/hr spot) or A100 40GB ($1.25/hr) |
| **Persistent storage** | Filesystem volumes ($0.20/GB/mo) | Network volumes ($0.07–0.10/GB/mo) |
| **SSH access** | ✅ direct | ✅ via `runpodctl` or standard SSH |
| **CLI** | `lambda ssh` | `runpodctl` |
| **Spot pricing** | ❌ on-demand only | ✅ spot ~30–50% cheaper |
| **UX** | Simpler, fewer options | More control, templates |
| **Cold start** | ~1 min | ~2–3 min |

**Recommendation:** Lambda Labs A10 for simplicity; RunPod RTX 4090 spot for cheapest option.

#### Time and Cost Estimate (3B QLoRA, 3940 train examples, 3 epochs)

| GPU | sec/step | steps/epoch | time/epoch | 3 epochs | cost |
|---|---|---|---|---|---|
| T4 16GB (Colab) | ~175s | 493 | ~24 hrs | ~72 hrs | $0 (impractical) |
| RTX 4090 24GB | ~12s | 493 | ~1.6 hrs | ~5 hrs | **~$2.20** |
| A10 24GB | ~18s | 493 | ~2.5 hrs | ~7.5 hrs | **~$5.60** |
| A100 40GB | ~8s | 493 | ~1.1 hrs | ~3.3 hrs | **~$3.60** |

> Step count: 3940 examples ÷ (batch=4 × grad_accum=8) = 493 steps/epoch at seq_len=2048.

#### Checkpoint Strategy

On fast GPUs, save every ~1 hour:
- RTX 4090 @ 12s/step → **`save_steps=300`** ≈ 60 min
- A10 @ 18s/step → **`save_steps=200`** ≈ 60 min
- A100 @ 8s/step → **`save_steps=450`** ≈ 60 min

Default in `train_mcm.py` is now `--save_steps 300 --save_total_limit 10` (keep last 10 checkpoints, ~10 hrs of history).

#### Lambda Labs Setup (step by step)

```bash
# 1. Create account → lambdalabs.com → add SSH key under Settings → SSH Keys
# 2. Launch instance: A10 24GB, "Ubuntu 22.04 LTS + CUDA 12.2" image
# 3. Create a persistent filesystem volume FIRST (Filesystems tab) — attach at launch
# 4. SSH in:
ssh ubuntu@<instance-ip>

# 5. Run setup script (one-time):
bash <(curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/learned-compiler-memory/main/scripts/setup_gpu_instance.sh)

# 6. Upload data (from your laptop — do once, stays in the persistent volume):
rsync -avz data/synthetic/ ubuntu@<instance-ip>:/home/ubuntu/mcm/data/synthetic/

# 7. Start training in tmux (survives disconnect):
tmux new -s train
python ~/mcm/scripts/train_mcm.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --data_dir ~/mcm/data/synthetic \
    --output_dir ~/mcm/checkpoints/mcm-write-v1 \
    --epochs 3 \
    --per_device_batch_size 4 \
    --grad_accumulation_steps 8 \
    --max_seq_length 2048 \
    --save_steps 200

# 8. Detach: Ctrl+B, D  — safe to close SSH
# 9. Re-attach later: tmux attach -t train
```

#### RunPod Setup (step by step)

```bash
# 1. Create account → runpod.io → add SSH public key under Settings → SSH Public Keys
# 2. Deploy pod: RTX 4090 Community Cloud, "RunPod PyTorch 2.x" template
#    - Add a Network Volume (persistent) and mount at /workspace
#    - Expose port 22 for SSH
# 3. SSH in (port shown in pod dashboard):
ssh root@<pod-host> -p <port>

# 4. Run setup script:
bash <(curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/learned-compiler-memory/main/scripts/setup_gpu_instance.sh)

# 5. Upload data from your laptop:
rsync -avz -e "ssh -p <port>" data/synthetic/ root@<pod-host>:/workspace/mcm/data/synthetic/

# 6. Train in tmux:
tmux new -s train
python /workspace/mcm/scripts/train_mcm.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --data_dir /workspace/mcm/data/synthetic \
    --output_dir /workspace/mcm/checkpoints/mcm-write-v1 \
    --epochs 3 \
    --per_device_batch_size 4 \
    --grad_accumulation_steps 8 \
    --max_seq_length 2048 \
    --save_steps 300
```

#### Resuming after disconnect

```bash
# Re-SSH, then:
tmux attach -t train
# If session died, resume from latest checkpoint:
python ~/mcm/scripts/train_mcm.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --data_dir ~/mcm/data/synthetic \
    --output_dir ~/mcm/checkpoints/mcm-write-v1 \
    --resume_from_checkpoint ~/mcm/checkpoints/mcm-write-v1/checkpoint-<N> \
    --epochs 3 --per_device_batch_size 4 --grad_accumulation_steps 8 \
    --max_seq_length 2048 --save_steps 200
```

#### Inference after training

Push adapter to HuggingFace Hub from the instance, then load anywhere:
```bash
huggingface-cli login
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
# load + push
model.push_to_hub('YOUR_HF_USERNAME/mcm-write-v1')
"
```

Or just `rsync` the checkpoint directory back to your laptop.

**Deliverables:**
- [x] `scripts/setup_gpu_instance.sh` — one-command environment setup for Lambda/RunPod
- [ ] Training run complete on Lambda or RunPod
- [ ] Adapter saved on HF Hub or local persistent volume

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

## Compute Requirements & Cost Estimates

| Task | Platform | Time estimate | Cost estimate |
|---|---|---|---|
| Synthetic data gen (5k samples) | MiniMax API (CPU) | ~8 hours (with quota limits) | ~$0 (free tier) |
| **LoRA fine-tune — Option A** | **Together AI serverless** | **~2–4 hours** | **~$58** |
| **LoRA fine-tune — Option B** | **Google Colab Free (T4)** | **~4–6 hrs (multi-session)** | **$0** |
| LoRA fine-tune — Option B+ | Google Colab Pro ($10/mo) | ~4–6 hrs (single session) | ~$10 |
| LoRA fine-tune — Option C | RunPod/Lambda A100 | ~6–8 hours | ~$10–15 |
| User adapter fine-tune (per user) | Colab Free or Together AI | ~1–2 hrs | $0–$5 |
| Experiments 1–3 (inference) | Together AI serverless | ~2 hours | ~$1–5 |
| Dedicated endpoint (optional serving) | Together AI H100 | hourly | $3.99/hr |

> **Recommended path:** Try **Colab Free** first (T4, $0). If sessions keep disconnecting, upgrade to **Colab Pro** (~$10) for a clean single run.
> Script: `scripts/train_mcm.py` (Colab/local QLoRA) or `scripts/train_together.py` (Together AI serverless).
>
> | Option | Train cost | Inference cost | Inference setup | Risk |
> |---|---|---|---|---|
> | Colab Free | $0 | $0 (batch in Colab) or $0 (HF Hub API) | Medium | Disconnects; must re-load adapter |
> | Colab Pro | ~$10/mo | Same as above | Medium | Low |
> | Together AI | ~$58 | ~$0.10/M tokens (serverless API) | ✅ Zero | None |
> | RunPod | ~$10–15 | ~$0.10/hr (keep pod running) | Medium | Low |

---

## MVP Success Criteria

The MVP is done when:
1. MCM can take a raw 50-turn conversation and output valid structured memory JSON
2. An agent using MCM memory outperforms an agent using RAG-only on at least 1 downstream task
3. MCM inference adds < 500ms latency per consolidation cycle
