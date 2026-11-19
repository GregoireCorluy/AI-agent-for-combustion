from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from config import DEFAULT_CONFIG_PATH, load_config
from data import prompt_completion, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from CombustionAgent.parameters import InputParameters  # noqa: E402

# This module validates the dataset schema and checks for split isolation.

def main():
    parser = argparse.ArgumentParser(description="Validate dataset schema and split isolation")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    config = load_config(args.config)
    key = config.section("data").get("assistant_output_key", "input_parameters")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path(),
        trust_remote_code=bool(config.section("model").get("trust_remote_code", False)),
    )
    split_paths = {
        "train": config.train_file,
        "validation": config.validation_file,
        "test": config.test_file,
    }
    group_to_split: dict[str, str] = {}
    prompt_to_split: dict[str, str] = {}
    lengths = []
    completion_lengths = []
    too_long = []
    max_seq_length = int(config.section("training").get("max_seq_length", 2048))
    max_new_tokens = int(config.section("evaluation").get("max_new_tokens", 512))
    completion_too_long = []
    counts = {}
    preview_printed = False
    for split, path in split_paths.items():
        rows = read_jsonl(path)
        counts[split] = len(rows)
        if not rows:
            raise ValueError(f"{split} dataset is empty: {path}")
        for index, row in enumerate(rows, 1):
            example = prompt_completion(row, tokenizer, key)
            params = json.loads(
                next(m["content"] for m in row["messages"] if m["role"] == "assistant")
            )[key]
            expected = set(InputParameters.model_fields)
            if set(params) != expected:
                raise ValueError(
                    f"{split}:{index}: schema keys differ; "
                    f"missing={expected - set(params)}, extra={set(params) - expected}"
                )
            InputParameters.model_validate(params)
            metadata = row.get("metadata", {})
            if metadata.get("split") != split:
                raise ValueError(
                    f"{split}:{index}: metadata.split is {metadata.get('split')!r}, "
                    f"expected {split!r}"
                )
            group = metadata.get("scenario_group")
            if not group:
                raise ValueError(f"{split}:{index}: missing metadata.scenario_group")
            prior = group_to_split.setdefault(group, split) # Record the first split seen for this scenario_group
            if prior != split:
                raise ValueError(f"scenario_group {group!r} leaks across {prior} and {split}")
            user_messages = [
                message for message in row["messages"]
                if isinstance(message, dict) and message.get("role") == "user"
            ]
            if len(user_messages) != 1 or not isinstance(user_messages[0].get("content"), str):
                raise ValueError(f"{split}:{index}: expected exactly one text user message")
            normalized_prompt = " ".join(user_messages[0]["content"].split())
            prior = prompt_to_split.setdefault(normalized_prompt, split)
            if prior != split:
                example_id = metadata.get("example_id", index)
                raise ValueError(
                    f"user prompt from {example_id!r} in {split} is identical after "
                    f"whitespace normalization to a prompt in {prior}"
                )
            prompt_ids = tokenizer(
                example["prompt"], add_special_tokens=False
            )["input_ids"]
            completion_ids = tokenizer(
                example["completion"], add_special_tokens=False
            )["input_ids"]
            eos_id = tokenizer.eos_token_id
            if not completion_ids or completion_ids[-1] != eos_id:
                raise ValueError(f"{split}:{index}: completion does not end with tokenizer EOS")
            if completion_ids.count(eos_id) != 1:
                raise ValueError(f"{split}:{index}: expected exactly one EOS token in completion")
            completion_length = len(completion_ids)
            completion_lengths.append(completion_length)
            if completion_length > max_new_tokens:
                metadata = row.get("metadata", {})
                completion_too_long.append(
                    f"{split}:{metadata.get('example_id', index)} "
                    f"(completion={completion_length})"
                )
            total_tokens = len(prompt_ids) + completion_length
            lengths.append(total_tokens)
            if total_tokens > max_seq_length:
                metadata = row.get("metadata", {})
                too_long.append(
                    f"{split}:{metadata.get('example_id', index)} "
                    f"(prompt={len(prompt_ids)}, completion={len(completion_ids)}, "
                    f"total={total_tokens})"
                )
            if not preview_printed:
                decoded = tokenizer.decode(
                    prompt_ids + completion_ids, skip_special_tokens=False
                )
                decoded_completion = tokenizer.decode(
                    completion_ids, skip_special_tokens=False
                )
                print(f"First formatted example: {decoded!r}")
                print(f"Decoded completion ending: {decoded_completion[-80:]!r}")
                preview_printed = True
    lengths.sort()

    def pct(values, q):
        """ Return the q-th quantile of the token lengths."""

        return values[min(int((len(values) - 1) * q), len(values) - 1)]

    print(f"Counts: {counts}")
    print(f"Unique scenario groups: {len(group_to_split)}; no cross-split leakage found")
    print(
        f"Prompt + completion tokens: min={lengths[0]}, p50={pct(lengths, .50)}, "
        f"p90={pct(lengths, .90)}, p95={pct(lengths, .95)}, max={lengths[-1]}"
    )
    print(
        f"Completion tokens (including EOS): min={completion_lengths[0]}, "
        f"p50={pct(completion_lengths, .50)}, p90={pct(completion_lengths, .90)}, "
        f"p95={pct(completion_lengths, .95)}, max={completion_lengths[-1]}"
    )
    for split, path in split_paths.items():
        rows = read_jsonl(path)
        family_counts = Counter(r.get("metadata", {}).get("application_family", "unknown") for r in rows)
        print(f"{split} by application_family: {dict(family_counts)}")
    errors = []
    if too_long:
        shown = "\n".join(f"  - {item}" for item in too_long[:20])
        remainder = len(too_long) - 20
        if remainder > 0:
            shown += f"\n  - ... and {remainder} more"
        errors.append(
            f"{len(too_long)} examples exceed training.max_seq_length="
            f"{max_seq_length}. Increase the limit or shorten the prompt/target:\n{shown}"
        )
    if completion_too_long:
        shown = "\n".join(f"  - {item}" for item in completion_too_long[:20])
        remainder = len(completion_too_long) - 20
        if remainder > 0:
            shown += f"\n  - ... and {remainder} more"
        errors.append(
            f"{len(completion_too_long)} assistant completions exceed "
            f"evaluation.max_new_tokens={max_new_tokens}. Increase that limit:\n{shown}"
        )
    if errors:
        raise ValueError("\n\n".join(errors))


if __name__ == "__main__":
    main()
