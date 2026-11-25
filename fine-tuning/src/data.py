from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

# This module provides utilities for reading and processing JSONL datasets for fine-tuning.

OUTPUT_INSTRUCTION = (
    "Return exactly one valid JSON object with only the top-level key input_parameters. "
    "The value must contain all InputParameters fields, using null when a value cannot "
    "be inferred responsibly. Do not output provenance, explicit anchors, explanations, "
    "or dataset metadata."
)


def prepare_system_prompt(prompt: str) -> str:
    prompt = prompt.rstrip()
    if OUTPUT_INSTRUCTION not in prompt:
        prompt += "\n\n" + OUTPUT_INSTRUCTION
    return prompt


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)
    return records


def build_prompt(tokenizer, system: str, user: str) -> str:
    return build_prompt_from_prepared_system(
        tokenizer, prepare_system_prompt(system), user
    )


def build_prompt_from_prepared_system(tokenizer, system: str, user: str) -> str:
    """Render a prompt when the system message already includes training instructions."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prompt_completion(
    record: dict[str, Any],
    tokenizer,
    output_key: str = "input_parameters",
) -> dict[str, str]:
    """Return a list of messages for the model, with the assistant message containing only the specified output key."""
    
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Each record must have a messages list")
    roles = {role: [m for m in messages if isinstance(m, dict) and m.get("role") == role]
             for role in ("system", "user", "assistant")}
    if any(len(items) != 1 for items in roles.values()):
        raise ValueError("Each record must contain exactly one system, user, and assistant message")
    parsed = json.loads(roles["assistant"][0]["content"])
    if not isinstance(parsed, dict) or output_key not in parsed:
        raise ValueError(f"Assistant JSON must contain {output_key!r}")
    # Train only on InputParameters. Keep metadata in the source JSONL for scoring.
    answer = json.dumps({output_key: parsed[output_key]}, ensure_ascii=False, separators=(",", ":"))
    eos = tokenizer.eos_token
    if not eos:
        raise ValueError("Tokenizer must define an EOS token for completion-only SFT")
    return {
        "prompt": build_prompt(
            tokenizer, roles["system"][0]["content"], roles["user"][0]["content"]
        ),
        "completion": answer + eos,
    }


def to_dataset(records: list[dict[str, Any]], tokenizer, output_key: str) -> Dataset:
    return Dataset.from_list(
        [prompt_completion(row, tokenizer, output_key) for row in records]
    )


def load_train_validation(config, tokenizer):
    key = config.section("data").get("assistant_output_key", "input_parameters")
    return (
        to_dataset(read_jsonl(config.train_file), tokenizer, key),
        to_dataset(read_jsonl(config.validation_file), tokenizer, key),
    )
