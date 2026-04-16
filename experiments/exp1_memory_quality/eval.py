"""
Experiment 1: Memory Quality (Intrinsic Evaluation)

Compares MCM against baselines on three metrics:
  1. Compression ratio (tokens before / tokens after)
  2. Information retention (key fact recall via judge LLM)
  3. Human eval proxy (GPT-4o as proxy judge: usefulness, completeness, accuracy)

Usage:
    python experiments/exp1_memory_quality/eval.py \
        --mcm_model checkpoints/mcm-v1 \
        --test_data data/synthetic/test.jsonl \
        --output results/exp1.json
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI

client = OpenAI()


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_truncation(log: str, max_tokens: int = 512) -> str:
    """Naive truncation baseline: keep the last N tokens worth of text."""
    words = log.split()
    # rough approximation: 1 word ≈ 1.3 tokens
    keep = int(max_tokens / 1.3)
    return " ".join(words[-keep:])


def baseline_rolling_summary(log: str) -> str:
    """Prompt GPT-4o-mini to summarize the log (prompt-engineering baseline)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the following conversation, preserving key facts, preferences, and decisions."},
            {"role": "user", "content": log},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compression_ratio(original: str, compressed: str) -> float:
    orig_tokens = len(original.split())
    comp_tokens = len(compressed.split())
    return orig_tokens / max(comp_tokens, 1)


def information_retention_score(original_log: str, memory_output: str, gold_facts: list[str]) -> float:
    """
    Ask a judge LLM: for each gold fact, can it be recovered from the memory output?
    Returns proportion of facts recoverable (0.0–1.0).
    """
    if not gold_facts:
        return 1.0

    recoverable = 0
    for fact in gold_facts:
        prompt = f"""Given this memory representation:

{memory_output}

Can the following fact be recovered or inferred from it?
Fact: "{fact}"

Answer with only YES or NO."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().upper()
        if answer.startswith("YES"):
            recoverable += 1

    return recoverable / len(gold_facts)


def proxy_human_eval(log: str, memory_output: str) -> dict[str, float]:
    """
    Use GPT-4o as a proxy human evaluator.
    Scores usefulness, completeness, accuracy on a 1–5 scale.
    """
    prompt = f"""You are evaluating a memory extraction system. Given the original conversation and the extracted memory, rate the memory on three dimensions:

Original conversation:
{log[:3000]}  # truncate for context window

Extracted memory:
{memory_output}

Rate on a 1-5 scale:
- usefulness: How useful would this memory be for a future agent session?
- completeness: Are important facts/preferences/decisions captured?
- accuracy: Is the extracted information factually correct vs. the conversation?

Output JSON only: {{"usefulness": N, "completeness": N, "accuracy": N}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(mcm_model_path: str, test_data_path: str, output_path: str):
    from src.memory_model.mcm_inference import MCMInference

    mcm = MCMInference(mcm_model_path)

    test_examples = []
    with open(test_data_path) as f:
        for line in f:
            test_examples.append(json.loads(line))

    results = {
        "mcm": [],
        "truncation": [],
        "rolling_summary": [],
    }

    for i, example in enumerate(test_examples):
        messages = example["messages"]
        log = messages[1]["content"].replace("Interaction log:\n\n", "")
        gold_memory_str = messages[2]["content"]
        gold_memory = json.loads(gold_memory_str)

        # Extract gold facts for retention scoring
        gold_facts = [f["content"] for f in gold_memory.get("memory", {}).get("facts", [])]

        print(f"Evaluating example {i+1}/{len(test_examples)}...")

        # MCM
        mcm_output = mcm.generate(log)
        mcm_output_str = json.dumps(mcm_output)

        # Baselines
        trunc_output = baseline_truncation(log)
        summary_output = baseline_rolling_summary(log)

        for name, output in [("mcm", mcm_output_str), ("truncation", trunc_output), ("rolling_summary", summary_output)]:
            cr = compression_ratio(log, output)
            ir = information_retention_score(log, output, gold_facts)
            proxy = proxy_human_eval(log, output)
            results[name].append({"compression_ratio": cr, "info_retention": ir, **proxy})

    # Aggregate
    aggregated = {}
    for method, scores in results.items():
        if not scores:
            continue
        keys = scores[0].keys()
        aggregated[method] = {k: sum(s[k] for s in scores) / len(scores) for k in keys}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"per_example": results, "aggregated": aggregated}, f, indent=2)

    print("\n=== Results ===")
    for method, scores in aggregated.items():
        print(f"\n{method}:")
        for k, v in scores.items():
            print(f"  {k}: {v:.3f}")

    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcm_model", required=True)
    parser.add_argument("--test_data", default="data/synthetic/test.jsonl")
    parser.add_argument("--output", default="results/exp1.json")
    args = parser.parse_args()

    run_evaluation(args.mcm_model, args.test_data, args.output)
