"""Probe which MiniMax models are available under the current token plan."""
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.minimax.io/v1",
    api_key=os.environ["MINIMAX_API_KEY"],
)

MODELS = [
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
    "MiniMax-M2.1",
    "MiniMax-M2.1-highspeed",
    "MiniMax-M2",
    "MiniMax-Text-01",
    "abab6.5s-chat",
]

for model in MODELS:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK."}],
            max_tokens=5,
        )
        reply = resp.choices[0].message.content.strip()
        print(f"  OK    {model}: {reply!r}")
    except Exception as e:
        short = str(e)[:130]
        print(f"  FAIL  {model}: {short}")
