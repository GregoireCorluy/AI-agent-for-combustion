from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEFAULT_CONFIG_PATH, load_config

# save merged model and tokenizer to disk, so that it can be loaded with AutoModelForCausalLM.from_pretrained()

def main():
    parser = argparse.ArgumentParser(description="Optional: merge a LoRA adapter into the base model")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--run-name", default=None, help="Merge a named training run; defaults to latest")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    model_cfg = dict(config.section("model"))
    model_cfg["name_or_path"] = config.model_name_or_path()
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    run_name, adapter_dir = config.resolve_adapter(args.run_name)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    if args.output:
        output = config.path(args.output)
    elif run_name:
        output = adapter_dir.parent / "merged"
    else:
        output = config.path("outputs/qwen3-8b-merged")
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_cfg["name_or_path"]).save_pretrained(output)
    print(f"Merged model written to {output}")


if __name__ == "__main__":
    main()
