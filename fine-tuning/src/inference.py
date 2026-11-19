from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel

from config import DEFAULT_CONFIG_PATH, load_config
from data import build_prompt, build_prompt_from_prepared_system, read_jsonl
from model import load_base_model, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# This module generates InputParameters for a single combustion case using a fine-tuned Qwen model with an optional QLoRA adapter.
# It reads the configuration, loads the model and tokenizer, and generates the output based on the provided description.

def main():
    parser = argparse.ArgumentParser(description="Generate InputParameters for one combustion case")
    parser.add_argument("description")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--run-name", default=None, help="Load a named training run; defaults to latest")
    args = parser.parse_args()
    config = load_config(args.config)
    model_cfg = dict(config.section("model"))
    model_cfg["name_or_path"] = config.model_name_or_path()
    selected_run_name, adapter = config.resolve_adapter(args.run_name)
    saved_prompt = adapter / "system_prompt.txt"
    if args.system_prompt is not None: # Use a system prompt from the command line, overriding any saved prompt
        prompt_builder = lambda user: build_prompt(tokenizer, args.system_prompt, user)
    elif saved_prompt.is_file(): # Use a saved system prompt from the adapter directory, if available
        prepared_system_prompt = saved_prompt.read_text(encoding="utf-8").rstrip("\n")
        prompt_builder = lambda user: build_prompt_from_prepared_system(
            tokenizer, prepared_system_prompt, user
        )
    else: # Use the system prompt from the first example in the training dataset, if no other prompt is provided
        first = read_jsonl(config.train_file)[0]
        system_prompt = next(
            m["content"] for m in first["messages"] if m["role"] == "system"
        )
        prompt_builder = lambda user: build_prompt(tokenizer, system_prompt, user)
    tokenizer = load_tokenizer(
        model_cfg["name_or_path"], bool(model_cfg.get("trust_remote_code", False))
    )
    model = load_base_model(model_cfg, config.section("quantization"))
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    prompt = prompt_builder(args.description)
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(config.section("evaluation").get("max_new_tokens", 512)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    print(f"\nAdapter run: {selected_run_name or adapter}")


if __name__ == "__main__":
    main()
