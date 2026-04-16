"""
Data generation pipeline for MCM training data.

Calls GPT-4o (or Claude) to generate (interaction_log → structured_memory) pairs.
Output: JSONL files in the format ready for QLoRA fine-tuning.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Any

# pip install openai
from openai import OpenAI

client = OpenAI()

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
    "personal_assistant",
    "software_engineering",
    "financial_trading",
    "research_assistant",
    "creative_writing",
    "language_learning",
    "health_coaching",
    "project_management",
    "data_analysis",
    "customer_support",
]

SCENARIO_LENGTHS = [10, 20, 30, 50, 75]  # turns


def generate_interaction_log(domain: str, n_turns: int) -> str:
    """Generate a synthetic interaction log for a given domain and length."""
    prompt = f"""Generate a realistic {n_turns}-turn conversation between a user and an AI assistant in the domain of: {domain}.

The conversation should:
- Feel natural with real user needs and preferences
- Include follow-up questions, corrections, and decisions
- Contain memorable facts the user reveals about themselves
- Have at least one task that is completed and one open question

Format:
Turn 1 [User]: ...
Turn 1 [Assistant]: ...
Turn 2 [User]: ...
...

Output ONLY the conversation, no preamble."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheaper for log generation
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    return response.choices[0].message.content


def generate_gold_memory(interaction_log: str) -> dict[str, Any]:
    """Use GPT-4o to generate the gold-standard structured memory for a log."""
    prompt = f"""Here is a raw agent interaction log:

<log>
{interaction_log}
</log>

Extract the structured memory following this schema:
{MEMORY_SCHEMA_EXAMPLE}

Output ONLY valid JSON."""

    response = client.chat.completions.create(
        model="gpt-4o",  # use the stronger model for gold labels
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def build_training_example(interaction_log: str, gold_memory: dict) -> dict:
    """Format a (log, memory) pair into the fine-tuning format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Interaction log:\n\n{interaction_log}"},
            {"role": "assistant", "content": json.dumps(gold_memory, indent=2)},
        ]
    }


def generate_dataset(n_samples: int, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": int(n_samples * 0.8),
        "val": int(n_samples * 0.1),
        "test": int(n_samples * 0.1),
    }

    all_examples = []

    print(f"Generating {n_samples} samples...")
    for i in range(n_samples):
        domain = random.choice(SCENARIO_DOMAINS)
        n_turns = random.choice(SCENARIO_LENGTHS)

        try:
            log = generate_interaction_log(domain, n_turns)
            memory = generate_gold_memory(log)
            example = build_training_example(log, memory)
            all_examples.append(example)

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{n_samples}")
        except Exception as e:
            print(f"  Warning: skipped sample {i} — {e}")
            continue

    # Write splits
    random.shuffle(all_examples)
    idx = 0
    for split, count in splits.items():
        split_examples = all_examples[idx : idx + count]
        idx += count
        out_file = output_path / f"{split}.jsonl"
        with open(out_file, "w") as f:
            for ex in split_examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(split_examples)} examples to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate MCM training data")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/synthetic")
    args = parser.parse_args()

    generate_dataset(args.n_samples, Path(args.output))
    print("Done.")


if __name__ == "__main__":
    main()
