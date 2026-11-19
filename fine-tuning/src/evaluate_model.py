from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm

from config import DEFAULT_CONFIG_PATH, load_config
from data import prompt_completion, read_jsonl
from metrics import aggregate, score
from model import load_base_model, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from CombustionAgent import parameters  # noqa: F401,E402

# This module evaluates a base Qwen model and an optional QLoRA adapter on a dataset.

def generate_batch(model, tokenizer, rows, max_new_tokens: int) -> list[str]:
    prompts = [prompt_completion(row, tokenizer)["prompt"] for row in rows]
    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True)
        for sequence in output
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate base Qwen and QLoRA adapter")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--run-name", default=None, help="Evaluate a named training run; defaults to latest")
    args = parser.parse_args()
    config = load_config(args.config)
    model_cfg = dict(config.section("model"))
    model_cfg["name_or_path"] = config.model_name_or_path()
    quant_cfg = config.section("quantization")
    rows = read_jsonl(config.validation_file if args.split == "validation" else config.test_file)
    tokenizer = load_tokenizer(
        model_cfg["name_or_path"], bool(model_cfg.get("trust_remote_code", False))
    )
    tokenizer.padding_side = "left"
    model = load_base_model(model_cfg, quant_cfg)
    adapter = None
    selected_run_name = args.run_name
    if not args.base_only:
        selected_run_name, adapter = config.resolve_adapter(args.run_name)
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    scored = []
    eval_cfg = config.section("evaluation")
    batch_size = int(eval_cfg.get("batch_size", 4))
    if batch_size < 1:
        raise ValueError("evaluation.batch_size must be at least 1")
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 512))
    for start in tqdm(range(0, len(rows), batch_size), desc=f"Generating {args.split}"):
        batch_rows = rows[start:start + batch_size]
        predictions = generate_batch(
            model, tokenizer, batch_rows, max_new_tokens
        )
        scored.extend(score(prediction, row) for prediction, row in zip(predictions, batch_rows))
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_family[row["application_family"]].append(row)
    report = {
        "model": model_cfg["name_or_path"],
        "adapter": str(adapter) if adapter else None,
        "split": args.split,
        "generated_at": datetime.now().astimezone().isoformat(),
        "overall": aggregate(scored),
        "by_application_family": {
            family: aggregate(items) for family, items in sorted(by_family.items())
        },
        "per_example": scored,
    }
    report_dir = config.path(config.section("output").get("report_dir", "outputs/evaluation"))
    report_dir = report_dir / (selected_run_name or "base")
    report_dir.mkdir(parents=True, exist_ok=True)
    label = "base" if args.base_only else "adapter"
    output_path = report_dir / f"{args.split}-{label}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {output_path}")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
