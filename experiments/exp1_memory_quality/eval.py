"""
Experiment 1: Memory Quality (Intrinsic Evaluation)

Compares MCM against baselines on three metrics:
  1. Compression ratio (tokens before / tokens after)
  2. Information retention (key fact recall via judge LLM)
  3. Human eval proxy (MiniMax-M2.7 as proxy judge: usefulness, completeness, accuracy)

Requires:  MINIMAX_API_KEY environment variable.
Usage:
    python experiments/exp1_memory_quality/eval.py \
        --mcm_model checkpoints/mcm-v1 \
        --test_data data/synthetic/test.jsonl \
        --output results/exp1.json
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import openai
from openai import OpenAI

# ---------------------------------------------------------------------------
# MiniMax client — OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
MINIMAX_BASE_URL = "https://api.minimax.io/v1"

def _make_client() -> OpenAI:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MINIMAX_API_KEY environment variable not set.\n"
            "Run: export MINIMAX_API_KEY=sk-cp-..."
        )
    return OpenAI(base_url=MINIMAX_BASE_URL, api_key=api_key)

client = _make_client()

# Models available under the Token Plan (highspeed variants NOT included)
JUDGE_FAST   = "MiniMax-M2.7"   # all judge calls use M2.7
JUDGE_STRONG = "MiniMax-M2.7"   # strongest, for scored proxy eval

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks, including unclosed ones from truncated responses."""
    text = _THINK_RE.sub("", text)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()

def _api_call_with_retry(fn, retries: int = 6, base_delay: float = 60.0):
    """Call fn(); on 429/529 errors, wait with exponential backoff and retry."""
    for attempt in range(retries):
        try:
            return fn()
        except (openai.RateLimitError, openai.InternalServerError) as e:
            if attempt == retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"  [API error {type(e).__name__}] waiting {wait:.0f}s before retry {attempt+1}/{retries-1}...")
            time.sleep(wait)


def _extract_json(text: str) -> dict:
    """Return parsed JSON from model output, stripping markdown fences and think-tags."""
    text = _strip_think(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON in model output: {text[:200]}")


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
    """Prompt MiniMax-M2.1 to summarize the log (prompt-engineering baseline)."""
    response = _api_call_with_retry(lambda: client.chat.completions.create(
        model=JUDGE_FAST,
        messages=[
            {"role": "system", "content": "Summarize the following conversation, preserving key facts, preferences, and decisions."},
            {"role": "user", "content": log},
        ],
        temperature=0.2,
        max_tokens=1024,
    ))
    return _strip_think(response.choices[0].message.content)


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
        response = _api_call_with_retry(lambda: client.chat.completions.create(
            model=JUDGE_FAST,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        ))
        answer = _strip_think(response.choices[0].message.content).strip().upper()
        if answer.startswith("YES"):
            recoverable += 1

    return recoverable / len(gold_facts)


def proxy_human_eval(log: str, memory_output: str) -> dict[str, float]:
    """
    Use MiniMax-M2.7 as a proxy human evaluator.
    Scores usefulness, completeness, accuracy on a 1–5 scale.
    """
    prompt = f"""You are evaluating a memory extraction system. Given the original conversation and the extracted memory, rate the memory on three dimensions:

Original conversation:
{log[:3000]}

Extracted memory:
{memory_output}

Rate on a 1-5 scale:
- usefulness: How useful would this memory be for a future agent session?
- completeness: Are important facts/preferences/decisions captured?
- accuracy: Is the extracted information factually correct vs. the conversation?

Output JSON only: {{"usefulness": N, "completeness": N, "accuracy": N}}"""

    try:
        response = _api_call_with_retry(lambda: client.chat.completions.create(
            model=JUDGE_STRONG,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2048,
        ))
        return _extract_json(response.choices[0].message.content)
    except Exception as e:
        print(f"  [proxy_human_eval] skipped: {e}")
        return {"usefulness": None, "completeness": None, "accuracy": None}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(mcm_model_path: str, test_data_path: str, output_path: str,
                   base_model: str = "Qwen/Qwen2.5-3B-Instruct",
                   max_examples: int | None = None):
    from src.memory_model.mcm_inference import MCMInference

    mcm = MCMInference(base_model, write_adapter_path=mcm_model_path)

    test_examples = []
    with open(test_data_path) as f:
        for line in f:
            test_examples.append(json.loads(line))
    if max_examples is not None:
        test_examples = test_examples[:max_examples]
    print(f"Evaluating {len(test_examples)} examples...")

    # Resume from checkpoint if output file exists
    checkpoint_path = Path(output_path + ".ckpt.json")
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        done = len(results.get("mcm", []))
        test_examples = test_examples[done:]
        print(f"Resuming from example {done+1} ({len(test_examples)} remaining)...")
    else:
        results = {"mcm": [], "truncation": [], "rolling_summary": []}
        done = 0

    def _save_checkpoint():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(results, f)

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
            try:
                cr = compression_ratio(log, output)
                ir = information_retention_score(log, output, gold_facts)
                proxy = proxy_human_eval(log, output)
                results[name].append({"compression_ratio": cr, "info_retention": ir, **proxy})
            except Exception as e:
                print(f"  [{name}] example {i+1} skipped: {e}")
        _save_checkpoint()

    # Aggregate (skip None values from failed API calls)
    aggregated = {}
    for method, scores in results.items():
        if not scores:
            continue
        keys = scores[0].keys()
        aggregated[method] = {
            k: sum(s[k] for s in scores if s.get(k) is not None) /
               max(sum(1 for s in scores if s.get(k) is not None), 1)
            for k in keys
        }

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
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--test_data", default="data/synthetic/test.jsonl")
    parser.add_argument("--output", default="results/exp1.json")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    run_evaluation(args.mcm_model, args.test_data, args.output, args.base_model, args.max_examples)
