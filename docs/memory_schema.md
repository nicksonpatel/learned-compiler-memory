# Memory Schema Specification v1

The Memory Consolidation Model (MCM) outputs a structured JSON object per consolidation cycle. This schema is the "standardized interface" contribution of the paper.

---

## Schema

```json
{
  "session_id": "string",
  "created_at": "ISO8601 timestamp",
  "source_turns": 42,
  "memory": {
    "facts": [
      {
        "content": "User prefers intraday trading over swing trading",
        "confidence": 0.95,
        "source_turn": 12,
        "tags": ["preference", "trading"]
      }
    ],
    "preferences": [
      {
        "domain": "communication",
        "preference": "Prefers concise bullet-point answers",
        "confidence": 0.88,
        "tags": ["style"]
      }
    ],
    "task_summaries": [
      {
        "task": "Build RSI-based breakout strategy",
        "status": "completed",
        "outcome": "Strategy implemented, backtest showed 23% annual return on NIFTY",
        "key_decisions": ["RSI threshold set to 30/70", "Used 15-min candles"],
        "tags": ["trading", "strategy"]
      }
    ],
    "reusable_abstractions": [
      {
        "name": "User risk profile",
        "description": "Conservative — avoids overnight positions, max 2% drawdown per trade",
        "tags": ["risk", "trading"]
      }
    ],
    "open_threads": [
      {
        "topic": "Explore options hedging for existing portfolio",
        "status": "pending",
        "tags": ["trading", "options"]
      }
    ],
    "corrections": [
      {
        "original": "User said they use 1H candles",
        "corrected_to": "User later clarified they use 15-min candles",
        "source_turn": 38,
        "tags": ["correction"]
      }
    ]
  }
}
```

---

## Field Definitions

| Field | Type | Description |
|---|---|---|
| `facts` | list | Discrete, verifiable statements about the user or domain |
| `preferences` | list | Style, behavior, or domain preferences expressed by the user |
| `task_summaries` | list | Completed or ongoing tasks with outcomes and key decisions |
| `reusable_abstractions` | list | Higher-order patterns the agent can reference repeatedly |
| `open_threads` | list | Topics mentioned but not resolved — follow up on these |
| `corrections` | list | Cases where earlier memory was explicitly corrected |

---

## Design Principles

1. **Flat and queryable** — each item in each list is independently retrievable via embedding search
2. **Confidence-scored** — MCM outputs a confidence for inferred facts (not stated facts)
3. **Tagged** — every item has `tags` for domain-based filtering without embedding search
4. **Append-friendly** — new consolidation cycles merge with existing schema, they don't replace
5. **Source-traced** — facts reference the source turn for auditability

---

## Merge Strategy (for consolidation cycles)

When a new cycle runs on top of existing memory:

```
new_facts → deduplicate against existing facts (semantic similarity > 0.92)
corrections → apply retroactively to relevant existing facts
open_threads → check if resolved in new turns → move to task_summaries
reusable_abstractions → update confidence score if reinforced
```

---

## Training Format

Each training example for the MCM looks like:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a memory consolidation model. Given a raw agent interaction log, extract and structure the key memories into the standard schema. Be precise, factual, and conservative — only store what was clearly stated or strongly implied."
    },
    {
      "role": "user",
      "content": "<raw interaction log of N turns>"
    },
    {
      "role": "assistant",
      "content": "<structured memory JSON>"
    }
  ]
}
```
