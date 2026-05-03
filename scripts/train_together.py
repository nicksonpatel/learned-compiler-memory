"""
Together AI serverless LoRA fine-tuning for MCM.

Uploads train/val data to Together AI, launches a LoRA fine-tuning job on
Qwen2.5-1.5B-Instruct, and polls until completion.

Pricing (as of Apr 2026): $0.48/M tokens for LoRA on models ≤16B.
Estimated cost for this dataset: ~$40-60 (3 epochs, ~85M train tokens).

Usage:
    pip install together
    export TOGETHER_API_KEY=<your-key>

    python scripts/train_together.py \
        --data_dir data/synthetic \
        --epochs 3 \
        --suffix mcm-write-v1

    # Dry-run (just upload files, don't train):
    python scripts/train_together.py --data_dir data/synthetic --dry_run
"""

import argparse
import os
import sys
import time
from pathlib import Path

from together import Together


def upload_file(client: Together, path: Path, purpose: str = "fine-tune") -> str:
    print(f"Uploading {path.name} ({path.stat().st_size / 1e6:.1f} MB)...")
    with open(path, "rb") as f:
        response = client.files.upload(file=(path.name, f), purpose=purpose)
    file_id = response.id
    print(f"  → file_id: {file_id}")
    return file_id


def create_job(client: Together, train_id: str, val_id: str, args) -> str:
    print("\nCreating fine-tuning job...")
    ft = client.fine_tuning.create(
        training_file=train_id,
        validation_file=val_id,
        model="Qwen/Qwen2.5-1.5B-Instruct",
        n_epochs=args.epochs,
        n_evals=10,              # evaluate 10× over training
        lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_trainable_modules="all-linear",
        learning_rate=2e-4,
        warmup_ratio=0.05,
        batch_size=32,           # max for this model
        suffix=args.suffix,
        wandb_api_key=os.environ.get("WANDB_API_KEY"),  # optional — logs to W&B if set
    )
    job_id = ft.id
    output_name = getattr(ft, "output_name", "")
    print(f"  → job_id:     {job_id}")
    print(f"  → model name: {output_name}")
    return job_id


def poll_job(client: Together, job_id: str, poll_interval: int = 30):
    print(f"\nPolling job {job_id} every {poll_interval}s...")
    last_status = None
    while True:
        job = client.fine_tuning.retrieve(job_id)
        status = job.status
        if status != last_status:
            ts = time.strftime("%H:%M:%S")
            events = getattr(job, "events", [])
            latest_event = events[-1].message if events else ""
            print(f"[{ts}] status={status}  {latest_event}")
            last_status = status
        if status in ("completed", "failed", "error", "cancelled"):
            break
        time.sleep(poll_interval)

    if status == "completed":
        output_name = getattr(job, "output_name", "")
        print(f"\n✅ Training complete!")
        print(f"   Model: {output_name}")
        print(f"\nTo run inference:")
        print(f"   from together import Together")
        print(f"   client = Together()")
        print(f"   r = client.chat.completions.create(")
        print(f"       model='{output_name}',")
        print(f"       messages=[...])")
    else:
        print(f"\n❌ Job ended with status: {status}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--suffix", default="mcm-write-v1",
                        help="Suffix appended to the output model name")
    parser.add_argument("--dry_run", action="store_true",
                        help="Upload files only, don't launch training job")
    parser.add_argument("--poll_interval", type=int, default=30,
                        help="Seconds between status polls")
    args = parser.parse_args()

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("Error: TOGETHER_API_KEY environment variable not set.")
        sys.exit(1)

    client = Together(api_key=api_key)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    val_path   = data_dir / "val.jsonl"

    for p in [train_path, val_path]:
        if not p.exists():
            print(f"Error: {p} not found.")
            sys.exit(1)

    # Upload
    train_id = upload_file(client, train_path)
    val_id   = upload_file(client, val_path)

    if args.dry_run:
        print("\nDry run — skipping job creation.")
        print(f"train_file_id: {train_id}")
        print(f"val_file_id:   {val_id}")
        return

    # Launch
    job_id = create_job(client, train_id, val_id, args)

    # Poll
    poll_job(client, job_id, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
