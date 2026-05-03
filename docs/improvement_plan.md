# Improvement Plan for MCM Design

This document captures the main design findings from the architecture review and turns them into a concrete improvement plan. It focuses on what is likely to work, what is currently overstated, and what should be improved before making stronger research or product claims.

---

## Bottom Line

The current design is strongest as a **memory systems architecture** and weakest as a **learned read-routing story**.

What looks solid today:

- Async write path with no LLM on the critical path
- Append-only raw event log retained as source of truth
- Immutable memory units with intended supersession semantics
- Hybrid retrieval with structured memory plus raw-history fallback

What remains unproven or fragile:

- The claim that a read LoRA will consistently outperform strong prompt-based routing
- The quality of self-generated read-head supervision
- The robustness of graph edge grounding and stale-node handling
- The gap between documented behavior and implemented behavior in several places

The project should therefore position itself first as a strong hybrid memory pipeline, and only second as a learned graph-routing system pending stronger evidence.

This is not a verdict against building the read LoRA. It is a statement about where the burden of proof lies. The write-side architecture is credible enough to stand on its own as a research contribution. The read LoRA is a research bet that needs experimental validation before strong claims can be made about it.

---

## What Is Good in the Current Design

### 1. Write path architecture is directionally correct

Moving all heavy consolidation out of `add_turn()` is the right choice for a production memory system. It removes the worst latency failure mode of prompt-based memory systems and gives the architecture a clear operational advantage.

### 2. Keeping the raw log is the right safety mechanism

The decision to never discard the event log is important. It prevents irreversible information loss and gives the system a recovery path when structured extraction misses something.

### 3. Structured memory plus semantic search is a sensible hybrid

Combining a structured store, vector index, and graph layer is a reasonable decomposition. The three layers serve different retrieval patterns, and the graph layer is a plausible place to experiment with learned routing.

### 4. Immutable units are better than in-place mutation

The intended supersession model is much easier to audit and debug than directly mutating memory facts. This is a strong design decision and should stay.

---

## Main Problems in the Current Design

### 1. The read-head training task does not match inference-time conditions

The read head is trained on full graph schema text, but inference uses a filtered subgraph selected by vector similarity. That creates a train-inference mismatch. Even if the model learns useful graph patterns, it is not being optimized for the exact decision setting it will face at runtime.

### 2. Read-head supervision relies only on templated generation

Current read-training examples are generated from graph edges using a small set of fixed templates. This teaches basic structural regularities but the model may learn templated path completion rather than discrimination between plausible paths. This is a known limitation of synthetic supervision, not a fundamental flaw — it is partially addressable. Injecting real held-out user queries is not tractable because by definition you do not have them at training time. What is tractable is generating harder negatives (distractor nodes that are semantically close but not on the correct path) and applying paraphrase augmentation to the query side so the model sees more surface variation for the same underlying path.

### 3. The system can reinforce write-head errors

The graph used to generate read-head training data is itself produced by the write head. If the write head adds bad edges or poor abstractions, the read head will be trained on those mistakes. This is a known risk in any self-supervised system. The confidence gating already in the architecture partially mitigates it — low-confidence units are dropped before reaching the graph. That gating does not fully close the loop, but it is not as blind a propagation as it might appear. The real guard against this is validating write-head extraction quality early before the read head depends on it.

### 4. Edge grounding is brittle

Graph edges are currently linked through truncated content snippets and fuzzy overlap. That is workable for a prototype, but not reliable enough for high-confidence graph reasoning claims.

### 5. Supersession logic is not fully realized

The schema and architecture assume contradiction handling and retroactive correction, but the current fallback logic does not yet implement strong supersession matching. Without robust supersession, stale memory will accumulate and the graph will become less trustworthy over time.

### 6. Fallback retrieval is still heuristic

The raw-log fallback exists, which is good, but the current implementation is simple keyword overlap over a capped history window. That is a useful safety net, not a strong retrieval subsystem.

### 7. The self-improving loop stops at dataset generation, not online improvement

The design describes a loop where the read head gets better over time with no human labelling. What the code actually does is flush training examples to JSONL and log that files are ready. That is dataset generation, not self-improvement. The gap between "writes a file and logs readiness" and "triggers training, waits for completion, and hot-swaps the adapter" is the kind of gap that damages credibility in a research context. The architecture doc claims continuous improvement; the implementation delivers a one-way data pipe.

### 8. Integration bugs are a separate problem from architectural flaws

The pipeline has real implementation issues — duplicate method definitions, calls to missing attributes, and constructor mismatches. These are bugs, not design problems. The architecture itself is not unsound because of them. However, they do mean the system cannot produce any evaluation results in its current state, which makes them a blocker regardless of root cause.

### 9. Research claims are ahead of evidence

The paper framing claims wins on downstream quality and efficiency, but the experiment folders for downstream and efficiency evaluation are still empty. At this stage, the strongest claims are architectural, not empirical.

---

## What I Expect Will Work

These expectations are realistic for the current design if implemented cleanly:

- Better write-path latency than prompt-based per-turn memory extraction
- Better auditability than systems that overwrite or discard memory states
- Better recall safety than pure structured-memory systems because the raw log is preserved
- Better long-session stability than naive rolling summaries
- Useful structured memory units for facts, preferences, tasks, and open threads

I would expect the system to perform well as a hybrid memory manager even if the read LoRA adds only modest gains.

---

## What I Would Not Yet Claim

These claims are not yet well-supported by the current design and implementation:

- That the read LoRA is categorically better than a strong prompted base model
- That the system is already self-improving in production rather than merely capable of self-generated training
- That graph routing is robust enough to be the primary retrieval strategy at scale
- That downstream task improvements are already established

The safer claim is that MCM provides a better memory systems architecture and a plausible path toward learned retrieval improvements.

---

## Priority Improvements

## P0: Fix the problem formulation and evaluation story

- Reframe the main contribution around learned memory formation and robust hybrid retrieval, not around a decisive read-LoRA win.
- Define the baseline clearly: semantic retrieval only, semantic retrieval plus prompted routing, and semantic retrieval plus read LoRA.
- Add explicit failure metrics: stale-path rate, invalid-edge rate, supersession error rate, and fallback activation rate.

## P1: Align read-head training with inference

This is the strongest technical criticism in the full review. The model is trained on full schema text but is asked to route on a filtered subgraph selected by vector similarity. If the vector pre-selection misses a key intermediate node, the LoRA cannot recover because it was never trained to route in that partial-information setting.

- Generate all read-head training examples from filtered subgraphs only, matching exactly what `GraphRouter` injects at inference time. This is a targeted change to `read_training_generator.py` and is the single highest-return fix available.
- Make the training input match runtime exactly: query, top-N candidate nodes selected by vector similarity, local edges between those nodes, then expected path or answer node.
- Include hard negatives — distractor nodes that are semantically similar to the correct path but not actually connected — so the model learns discrimination rather than pattern completion.
- Apply paraphrase augmentation on the query side for surface variation without requiring real user traffic.

## P2: Make graph supervision more trustworthy

- Replace content-snippet edge resolution with stable unit IDs emitted by the write head.
- Add confidence scoring for each predicted edge and drop low-confidence relations.
- Validate generated edges before they become read-head supervision.

## P3: Strengthen supersession and freshness handling

- Implement actual contradiction detection and semantic deduplication.
- Ensure retrieval and graph traversal consistently filter superseded units.
- Introduce freshness-aware traversal so stale nodes do not silently survive in route outputs.

## P4: Make fallback retrieval a real subsystem

- Replace keyword overlap with BM25 or raw-turn embedding retrieval.
- Retrieve over the full event history or session-aware shards rather than a small capped window.
- Measure how often fallback saves the system and what types of misses it covers.

## P5: Consider narrowing the read-head task (tradeoff, not a clear win)

Reranking candidate paths rather than generating them free-form is worth exploring, but it is not an obvious improvement. For single-hop queries, reranking a small candidate set is likely more stable than generation. For multi-hop paths of 3+ hops, the candidate space grows combinatorially and reranking becomes expensive or requires aggressive pruning. The decision depends on what hop depth your graphs actually reach in practice.

- Evaluate the median and 90th-percentile hop depth in your generated graphs before committing to a generation vs. reranking strategy.
- For single-hop answer retrieval, reranking is likely better.
- For multi-hop chains, constrained beam search over the local subgraph may be a middle ground: generation, but pruned to edges that exist in the injected schema.

## P6: Add a two-stage recall-then-rerank retrieval pipeline

The current retrieval flow asks the embedding model to be both the recall stage and the precision stage. That is asking too much of a single similarity score. A cross-encoder reranker, placed between the candidate pool and the final selection, separates those concerns cleanly:

```
embedding → top-100 candidates (recall, not precision)
reranker  → top-5 / top-50 (precision, actual relevance signal)
threshold check on reranker score, not embedding distance
```

This fits cleanly into the existing architecture because the reranker only sits on the read path. Write latency is completely unaffected.

**The most important connection in this system is not retrieval quality in isolation — it is training data quality for the read head.** If the vector pre-selection feeding into `GraphRouter` improves, the subgraph the LoRA gets trained on improves. That directly addresses the main criticism in P1 (train-inference mismatch) from a different angle: instead of only changing what schema format training examples use, you also improve which nodes end up in that schema in the first place.

Specific integration points:

- `HybridRetriever.retrieve()` — increase initial `top_k` to 50–100, pass candidate texts through the reranker, re-sort by reranker score, then apply the existing fallback threshold against reranker score instead of embedding distance.
- `GraphRouter._get_filtered_schema()` — replace the current direct VectorIndex top-50 with a reranker-filtered top-50 drawn from a larger embedding pool (e.g. top-150 → rerank → top-50). The subgraph nodes the LoRA sees at both training and inference time are then higher quality.
- `ReadTrainingAccumulator` / `generate_read_training_data()` — generate training examples from reranker-filtered subgraphs only. This ensures training and inference see the same node selection process, which is the core fix for the train-inference mismatch identified in P1.

**Real trade-offs to weigh before building this:**

- Latency. Cross-encoder reranking over 100 candidates is not free. A small (1–3B) open-source reranker running on-device will add 100–400ms per read query depending on batch size and hardware. That is noticeable for interactive agents. Batch scoring and caching repeated query-document pairs can help, but cache hit rates for memory retrieval queries tend to be low because queries are contextual.
- The feedback loop concern raised in the proposal (training on reranked data + using same reranker at inference = bias) is real but mislabeled — it is not overfitting, it is training-inference correlation, which is actually what you want. The actual risk is different: if you ever swap the reranker model, your LoRA training data is suddenly drawn from a different distribution than what the new reranker will produce. Keep the reranker stable once you start training the read head on its outputs.
- The proposal claims this is the "highest ROI upgrade available right now." That is too strong. Fixing the pipeline integration bugs (P8 in current problems) is higher ROI because nothing can be measured without it. Fixing the training distribution mismatch (P1) is arguably equal or higher ROI because it directly repairs the main modeling flaw. The reranker is a worthwhile improvement with the clearest impact on retrieval precision, but it is a medium-term upgrade, not the immediate unblock.
- Do not add the reranker until the baseline system is running end-to-end. Adding another model and failure point before you have a working evaluation loop will slow you down.

**Added experiments if the reranker is built:**

- Embedding-only vs. embedding + reranker on held-out memory retrieval queries (precision at K)
- Fallback activation rate before and after reranker (expect a significant drop)
- GraphRouter path accuracy with embedding-selected subgraph vs. reranker-selected subgraph
- Read LoRA accuracy trained on reranker-filtered examples vs. embedding-filtered examples

---

## Immediate Actions (Do These Before Anything Else)

The review identifies many things to improve. Most of them are not urgent without a working system. These three unblock everything else:

**1. ~~Fix the train-inference mismatch.~~ ✓ Done**
`read_training_generator.py` now accepts a `vector_index` argument and constructs each training example from a query-filtered subgraph (top-N nodes via vector similarity), matching exactly what `GraphRouter._get_filtered_schema()` injects at inference time. `ConsolidationWorker` passes its live `VectorIndex` instance on every call. The full-schema fallback is retained for early runs before the index is populated.

**2. ~~Fix the pipeline integration bugs.~~ ✓ Done**
All four integration bugs in `memory_pipeline.py` are resolved:
- `MemoryPipeline.create()` now instantiates `MemoryGraph` and passes it to `ConsolidationWorker` with the correct argument order.
- The duplicate `retrieve` method (which called the non-existent `self.retriever.search()`) is removed.
- `get_full_memory()` now delegates to `self._memory_store.get_active_units()`.
- `force_consolidate()` now calls `self.consolidate_sync()` instead of the non-existent `self.consolidate()`.

**3. Run the baseline comparison.**
On a fixed held-out eval set: measure plain vector retrieval, prompted base model routing on filtered subgraph, and read LoRA routing on filtered subgraph. If the LoRA wins even modestly on multi-hop queries, that is a real result worth reporting. If it does not, you have an honest finding and the write-side architecture becomes the main contribution instead. Either outcome is defensible. Not running the comparison is not.

---

## Recommended Research Positioning

If the goal is a paper, the current strongest paper claim is:

> Memory formation is a learned transformation and should be decoupled from the main reasoning model.

That claim is credible with the current architecture and does not require the read LoRA to win anything. The write-side system — async consolidation, immutable memory units, hybrid retrieval with raw-log fallback — is a legitimate research and engineering contribution on its own.

The weaker and riskier paper claim is:

> A read LoRA trained on self-generated graph paths consistently beats strong prompted routing baselines.

This may still turn out to be true. But the burden of proof lies with that claim, and you do not currently have the experiments to meet it. Do not let the architecture doc continue asserting it as established. Run the baseline comparison (see Immediate Actions above). If the LoRA wins, promote the claim. If it does not, the write-side contribution carries the paper.

---

## Suggested Near-Term Milestones

### Milestone 1: Stabilize the write-side system

- Validate memory extraction quality
- Validate contradiction handling
- Validate raw-log fallback behavior
- Establish strong baselines for write-side memory quality

### Milestone 2: Fix read-head supervision distribution

- Move from full-schema training to filtered-subgraph training (matches `GraphRouter` inference exactly)
- Add hard negative distractor nodes to each training example
- Apply query paraphrase augmentation for surface variation
- Evaluate answer-node accuracy before measuring full path accuracy
- If the reranker (P6) is ready: generate training examples from reranker-filtered subgraphs rather than embedding-only subgraphs

### Milestone 3: Run the missing experiments

- Downstream task evaluation against prompt-only and RAG-only baselines
- Efficiency evaluation on retrieval latency and context size
- Ablations for no-graph, no-LoRA, and no-fallback settings

### Milestone 4: Tighten paper claims

- Promote only the claims supported by measured results
- Separate architectural contributions from speculative model improvements
- Show where the read LoRA helps, and just as importantly, where it does not

---

## Success Criteria for the Next Iteration

The next version of the design should meet the following bar:

- Read-head training distribution matches inference distribution
- Graph edges are linked by stable identifiers, not fuzzy text overlap
- Supersession is operational and measurable
- Fallback retrieval is strong enough to serve as a serious safety layer
- Experiments exist for downstream quality and efficiency
- Claims in the paper match what the implementation and results actually support
- If a reranker is added: it is measured against embedding-only baseline, its latency impact is documented, and the LoRA is trained on reranker-filtered examples rather than embedding-filtered examples

If those conditions are met, the project becomes much more compelling both as a research artifact and as an applied system.