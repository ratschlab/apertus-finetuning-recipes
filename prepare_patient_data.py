#!/usr/bin/env python3
"""Prepare patient-level HF Dataset from clinical JSONL for DP-LoRA training.

Groups episodes by patient_id and concatenates text into a single sequence
per patient. This ensures privacy_unit = patient for Opacus DP training.

Input: JSONL file where each line has at minimum:
    {"text": "...", "patient_id": "PAT_00001", ...}

If no patient_id field is present, each episode is treated as its own
privacy unit (episode-level DP).

Output: HF Dataset saved to disk (load with datasets.load_from_disk()).

Usage:
    # Patient-level grouping (from LLM-generated data)
    python prepare_patient_data.py \
        --input ../dp-megatron-dev/synthetic_data/synthetic_clinical.jsonl \
        --output data/patient_dataset \
        --group-by-patient

    # Episode-level (each line = one privacy unit)
    python prepare_patient_data.py \
        --input ../dp-megatron-dev/synthetic_data/synthetic_episodes.jsonl \
        --output data/episode_dataset
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file, return list of records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {line_num}: {e}", file=sys.stderr)
    return records


def group_by_patient(records: list[dict], separator: str = "\n\n---\n\n") -> list[dict]:
    """Group episodes by patient_id, concatenate text.

    Returns one record per patient with concatenated text.
    """
    patient_episodes = defaultdict(list)
    patient_order = []  # Preserve first-seen order

    for record in records:
        pid = record.get("patient_id", record.get("episode_id", "unknown"))
        if pid not in patient_episodes:
            patient_order.append(pid)
        patient_episodes[pid].append(record)

    grouped = []
    for pid in patient_order:
        episodes = patient_episodes[pid]
        # Sort by episode_id if available for consistent ordering
        episodes.sort(key=lambda r: r.get("episode_id", ""))

        text = separator.join(ep["text"] for ep in episodes)
        grouped.append({
            "text": text,
            "patient_id": pid,
            "num_episodes": len(episodes),
            "total_chars": len(text),
        })

    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Prepare patient-level HF Dataset from clinical JSONL"
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for HF Dataset")
    parser.add_argument(
        "--group-by-patient",
        action="store_true",
        help="Group episodes by patient_id (default: each line = one privacy unit)",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="\n\n---\n\n",
        help="Separator between episodes when grouping",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Truncate patient text to this many characters (approximate token control)",
    )
    args = parser.parse_args()

    # Load
    print(f"Loading {args.input}...")
    records = load_jsonl(args.input)
    print(f"  Loaded {len(records)} records")

    # Group if requested
    if args.group_by_patient:
        print("Grouping by patient_id...")
        data = group_by_patient(records, args.separator)
        print(f"  {len(records)} episodes → {len(data)} patients")
    else:
        data = [
            {
                "text": r["text"],
                "patient_id": r.get("patient_id", r.get("episode_id", f"unit_{i}")),
                "num_episodes": 1,
                "total_chars": len(r["text"]),
            }
            for i, r in enumerate(records)
        ]

    # Optional truncation
    if args.max_chars:
        for item in data:
            if len(item["text"]) > args.max_chars:
                item["text"] = item["text"][: args.max_chars]
                item["total_chars"] = args.max_chars

    # Build HF Dataset
    dataset = Dataset.from_dict(
        {
            "text": [d["text"] for d in data],
            "patient_id": [d["patient_id"] for d in data],
            "num_episodes": [d["num_episodes"] for d in data],
        }
    )

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(args.output)

    # Stats
    lengths = [len(d["text"]) for d in data]
    ep_counts = [d["num_episodes"] for d in data]
    print(f"\nDataset saved to {args.output}")
    print(f"  N (privacy units): {len(data)}")
    print(f"  Text length (chars): min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.0f}")
    if args.group_by_patient:
        print(f"  Episodes per patient: min={min(ep_counts)}, max={max(ep_counts)}, "
              f"mean={sum(ep_counts)/len(ep_counts):.1f}")


if __name__ == "__main__":
    main()
