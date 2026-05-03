"""
Data generation pipeline for MCM training data — parallel async version.

Calls MiniMax API (OpenAI-compatible) with N concurrent workers to generate
(interaction_log → structured_memory) pairs at maximum throughput.

Rate limit: 100 req/min. Each sample = 2 API calls (log + gold memory).
With --workers 40: ~40 in-flight calls, hitting ~80 req/min safely.

Requires:  MINIMAX_API_KEY environment variable.
Usage:
    export MINIMAX_API_KEY=sk-cp-...
    python scripts/generate_data.py --n_samples 5000 --output data/synthetic --max_hours 5 --workers 40
"""

import asyncio
import json
import os
import re
import random
import time
import argparse
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# MiniMax async client
# ---------------------------------------------------------------------------
MINIMAX_BASE_URL = "https://api.minimax.io/v1"

def _make_client() -> AsyncOpenAI:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MINIMAX_API_KEY environment variable not set.\n"
            "Run: export MINIMAX_API_KEY=sk-cp-..."
        )
    return AsyncOpenAI(base_url=MINIMAX_BASE_URL, api_key=api_key)

# Models available under the Token Plan (highspeed variants NOT included)
LOG_MODEL  = "MiniMax-M2.1"   # cheaper, fast enough for log generation
GOLD_MODEL = "MiniMax-M2.7"   # strongest, used for gold label extraction

MIN_LOG_CHARS = 200            # discard empty / truncated logs

# ---------------------------------------------------------------------------
# Think-tag stripping and JSON extraction
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()

def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON found in model output: {text[:200]}")

# ---------------------------------------------------------------------------
# Prompts and scenarios
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a memory consolidation model. Given a raw agent interaction log, extract and structure the key memories into the standard JSON schema.

Be:
- Precise: only store what was clearly stated or strongly implied
- Conservative: prefer omitting over fabricating
- Structured: always output valid JSON matching the schema exactly

Output ONLY valid JSON, no explanation."""

MEMORY_SCHEMA_EXAMPLE = """{
  "memory": {
    "facts": [{"content": "...", "confidence": 0.95, "source_turn": 3, "tags": ["..."]}],
    "preferences": [{"domain": "...", "preference": "...", "confidence": 0.9, "tags": ["..."]}],
    "task_summaries": [{"task": "...", "status": "completed|in_progress|pending", "outcome": "...", "key_decisions": ["..."], "tags": ["..."]}],
    "reusable_abstractions": [{"name": "...", "description": "...", "tags": ["..."]}],
    "open_threads": [{"topic": "...", "status": "pending", "tags": ["..."]}],
    "corrections": [{"original": "...", "corrected_to": "...", "source_turn": 15, "tags": ["..."]}]
  }
}"""

SCENARIO_DOMAINS = [
    "personal_assistant", "software_engineering", "financial_trading",
    "research_assistant", "creative_writing", "language_learning",
    "health_coaching", "project_management", "data_analysis", "customer_support",
]

SCENARIO_LENGTHS = [10, 20, 30, 50, 75]  # turns


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """Async token-bucket: allows `rate` API calls per 60 seconds.

    Starts with an empty bucket so all workers stagger at startup rather
    than bursting all `rate` tokens simultaneously.
    """
    def __init__(self, rate: int):
        self._rate = rate
        self._tokens = 0.0          # start empty — prevents burst on startup
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate / 60.0)
            self._last_refill = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / (self._rate / 60.0)
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


# ---------------------------------------------------------------------------
# API wrappers with exponential back-off retry
# ---------------------------------------------------------------------------
async def _call_with_retry(
    client: AsyncOpenAI,
    limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    max_retries: int = 6,
    **kwargs,
) -> str:
    """Rate-limited + concurrency-capped API call with exponential back-off."""
    for attempt in range(max_retries):
        await limiter.acquire()                 # respect calls-per-minute
        async with semaphore:                   # cap concurrent in-flight calls
            try:
                resp = await client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                code = getattr(e, "status_code", None)
                msg  = str(e)
                if code in (429, 500, 529) or "529" in msg or "overloaded" in msg.lower() or "rate" in msg.lower():
                    wait = min(2 ** attempt, 60)
                    print(f"    [retry {attempt+1}/{max_retries}] {code or 'err'} — wait {wait}s: {msg[:60]}", flush=True)
                    await asyncio.sleep(wait)
                    continue
                raise
    raise RuntimeError(f"Gave up after {max_retries} retries")


async def _gen_log(client, limiter, sem, domain, n_turns):
    prompt = (
        f"Generate a realistic {n_turns}-turn conversation between a user and an AI assistant "
        f"in the domain of: {domain}.\n\n"
        "The conversation should:\n"
        "- Feel natural with real user needs and preferences\n"
        "- Include follow-up questions, corrections, and decisions\n"
        "- Contain memorable facts the user reveals about themselves\n"
        "- Have at least one task that is completed and one open question\n\n"
        "Format:\nTurn 1 [User]: ...\nTurn 1 [Assistant]: ...\nTurn 2 [User]: ...\n...\n\n"
        "Output ONLY the conversation, no preamble."
    )
    raw = await _call_with_retry(
        client, limiter, sem,
        model=LOG_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=8192,
    )
    return _strip_think(raw)


async def _gen_gold(client, limiter, sem, log):
    prompt = (
        f"Here is a raw agent interaction log:\n\n<log>\n{log}\n</log>\n\n"
        f"Extract the structured memory following this schema:\n{MEMORY_SCHEMA_EXAMPLE}\n\n"
        "Output ONLY valid JSON."
    )
    raw = await _call_with_retry(
        client, limiter, sem,
        model=GOLD_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    return _extract_json(_strip_think(raw))


def _build_example(log, gold):
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Interaction log:\n\n{log}"},
            {"role": "assistant", "content": json.dumps(gold, indent=2)},
        ]
    }


# ---------------------------------------------------------------------------
# Per-worker coroutine
# ---------------------------------------------------------------------------
async def worker(
    wid: int,
    queue: asyncio.Queue,
    results: list,
    lock: asyncio.Lock,
    ckpt_file: Path,
    checkpoint_every: int,
    client: AsyncOpenAI,
    limiter: RateLimiter,
    sem: asyncio.Semaphore,
    counters: dict,
    deadline: float,
    max_turns: int | None,
):
    while True:
        try:
            sample_idx = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        if time.time() >= deadline:
            queue.task_done()
            return

        domain  = random.choice(SCENARIO_DOMAINS)
        n_turns = random.choice(SCENARIO_LENGTHS)
        if max_turns and n_turns > max_turns:
            n_turns = max_turns

        t0 = time.monotonic()
        try:
            log = await _gen_log(client, limiter, sem, domain, n_turns)
            if len(log) < MIN_LOG_CHARS:
                raise ValueError(f"Log too short ({len(log)} chars)")

            gold    = await _gen_gold(client, limiter, sem, log)
            example = _build_example(log, gold)

            async with lock:
                results.append(example)
                counters["ok"] += 1
                total     = len(results)
                elapsed_m = (time.time() - counters["start"]) / 60
                rate      = total / elapsed_m if elapsed_m > 0 else 0
                print(
                    f"  [w{wid:02d}] #{sample_idx+1:04d} {domain}/{n_turns}t "
                    f"{len(log)}c  {time.monotonic()-t0:.0f}s  "
                    f"total={total}  {rate:.1f}/min",
                    flush=True,
                )
                if counters["ok"] % checkpoint_every == 0:
                    with open(ckpt_file, "w") as f:
                        for ex in results:
                            f.write(json.dumps(ex) + "\n")
                    print(f"  [ckpt] saved {total} examples  ({elapsed_m:.1f}m, {counters['skip']} skipped)", flush=True)

        except Exception as e:
            async with lock:
                counters["skip"] += 1
                print(f"  [w{wid:02d}] SKIP #{sample_idx+1}: {e}", flush=True)
        finally:
            queue.task_done()


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------
async def run(n_samples, output_path, max_hours, workers, rate_per_min, max_concurrent, checkpoint_every, max_turns):
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = output_path / "checkpoint.jsonl"

    results: list[dict] = []
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            for line in f:
                s = line.strip()
                if s:
                    results.append(json.loads(s))
        print(f"Resumed: {len(results)} examples already in checkpoint.")

    remaining = n_samples - len(results)
    if remaining <= 0:
        print(f"Already at target ({len(results)} examples). Nothing to do.")
        return

    print(
        f"Generating {remaining} more  (target={n_samples}, have={len(results)}, "
        f"workers={workers}, concurrent={max_concurrent}, rate={rate_per_min}/min, max={max_hours}h)",
        flush=True,
    )

    client  = _make_client()
    limiter = RateLimiter(rate_per_min)
    sem     = asyncio.Semaphore(max_concurrent)   # hard cap on simultaneous in-flight calls

    queue = asyncio.Queue()
    for i in range(len(results), n_samples):
        queue.put_nowait(i)

    lock     = asyncio.Lock()
    deadline = time.time() + max_hours * 3600
    counters = {"ok": 0, "skip": 0, "start": time.time()}

    worker_tasks = [
        asyncio.create_task(worker(
            wid, queue, results, lock,
            ckpt_file, checkpoint_every,
            client, limiter, sem, counters, deadline, max_turns,
        ))
        for wid in range(1, workers + 1)
    ]

    await asyncio.gather(*worker_tasks)

    # Final save
    with open(ckpt_file, "w") as f:
        for ex in results:
            f.write(json.dumps(ex) + "\n")

    total     = len(results)
    elapsed_m = (time.time() - counters["start"]) / 60
    print(
        f"\nFinished. total={total}  ok={counters['ok']}  skipped={counters['skip']}  "
        f"elapsed={elapsed_m:.1f}m  avg_rate={counters['ok']/elapsed_m:.1f}/min",
        flush=True,
    )

    if total == 0:
        return

    # Train / val / test split
    random.shuffle(results)
    n_train = int(total * 0.8)
    n_val   = int(total * 0.1)
    for split, chunk in [
        ("train", results[:n_train]),
        ("val",   results[n_train:n_train+n_val]),
        ("test",  results[n_train+n_val:]),
    ]:
        out = output_path / f"{split}.jsonl"
        with open(out, "w") as f:
            for ex in chunk:
                f.write(json.dumps(ex) + "\n")
        print(f"  {split}: {len(chunk)} examples → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Generate MCM training data (parallel async)")
    p.add_argument("--n_samples",        type=int,   default=5000)
    p.add_argument("--output",           type=str,   default="data/synthetic")
    p.add_argument("--max_hours",        type=float, default=5.0)
    p.add_argument("--workers",          type=int,   default=40,
                   help="Concurrent async workers (default 40)")
    p.add_argument("--rate_per_min",     type=int,   default=90,
                   help="API calls per minute cap (default 90, limit=100)")
    p.add_argument("--max_concurrent",   type=int,   default=20,
                   help="Max simultaneous in-flight API calls (default 20)")
    p.add_argument("--checkpoint_every", type=int,   default=10)
    p.add_argument("--max_turns",        type=int,   default=None,
                   help="Cap conversation turns (quick test mode)")
    args = p.parse_args()

    asyncio.run(run(
        n_samples=args.n_samples,
        output_path=Path(args.output),
        max_hours=args.max_hours,
        workers=args.workers,
        rate_per_min=args.rate_per_min,
        max_concurrent=args.max_concurrent,
        checkpoint_every=args.checkpoint_every,
        max_turns=args.max_turns,
    ))


if __name__ == "__main__":
    main()
