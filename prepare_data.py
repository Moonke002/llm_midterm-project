"""Create a small, balanced, prompt-aligned LLM response dataset."""

import argparse
import random
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


MODELS = {
    "gemma3-4b": "Gemma",
    "olmo2-7b": "OLMo",
    "qwen3-no_reasoning-4b": "Qwen",
}
DATASET_NAME = "allenai/Dolci-DPO-Model-Response-Pool"
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)")


def clean_row(row):
    """Return safe normalized fields, or None for excluded rows."""
    # Read and normalize each response record
    prompt = str(row.get("prompt", "")).strip()
    response = str(row.get("response", "")).strip()
    source = str(row.get("dataset", "unknown")).strip()
    source_lower = source.lower()

    # Keep only verified permissive source rows
    if source != "IF_sft_data_verified_permissive":
        return None
    if "wildchat" in source_lower or "filtered_wc" in source_lower:
        return None
    if len(prompt) < 8 or len(response) < 40:
        return None
    if EMAIL_RE.search(prompt + " " + response):
        return None
    if PHONE_RE.search(prompt + " " + response):
        return None
    if "name_" in (prompt + " " + response).lower():
        return None

    return {
        "custom_id": str(row.get("custom_id", "")),
        "source_dataset": source,
        "prompt": prompt,
        "response": response,
    }


def read_model(config, family, max_scan):
    """Stream one model configuration and keep eligible records."""
    # Stream a limited number of model records
    stream = load_dataset(DATASET_NAME, config, split="train", streaming=True)
    rows = {}
    scanned = 0
    for row in stream:
        scanned += 1
        item = clean_row(row)
        if item and item["custom_id"]:
            rows.setdefault(item["custom_id"], item)
        if scanned >= max_scan:
            break

    print(f"{config}: kept {len(rows)} rows from {scanned} scanned")
    return family, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-model", type=int, default=600)
    parser.add_argument("--max-scan", type=int, default=10000)
    parser.add_argument("--min-shared", type=int, default=100)
    parser.add_argument("--output", default="data/llm_responses.csv")
    args = parser.parse_args()

    collected = {}
    for config, family in MODELS.items():
        collected[config] = read_model(config, family, args.max_scan)[1]

    shared_ids = set.intersection(*(set(rows) for rows in collected.values()))
    reference = collected[next(iter(MODELS))]
    shared_ids = {
        custom_id for custom_id in shared_ids
        if all(
            collected[config][custom_id]["prompt"]
            == reference[custom_id]["prompt"]
            and collected[config][custom_id]["source_dataset"]
            == reference[custom_id]["source_dataset"]
            for config in MODELS
        )
    }
    if len(shared_ids) < args.min_shared:
        raise RuntimeError(
            f"Only {len(shared_ids)} shared prompts found. "
            "Increase --max-scan and run again."
        )

    # Match prompts and source labels across models
    selected_ids = random.Random(42).sample(
        sorted(shared_ids), min(args.per_model, len(shared_ids))
    )
    records = []
    for config, family in MODELS.items():
        for custom_id in selected_ids:
            item = collected[config][custom_id]
            records.append(
                {
                    "custom_id": custom_id,
                    "LLM_name": family,
                    "model_config": config,
                    "source_dataset": item["source_dataset"],
                    "LLM_Input": item["prompt"],
                    "LLM_output": item["response"],
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"Saved {len(records)} balanced rows to {output}")
    print(pd.Series([r["LLM_name"] for r in records]).value_counts())


if __name__ == "__main__":
    main()
