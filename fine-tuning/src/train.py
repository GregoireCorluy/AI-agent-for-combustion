from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch.distributed as dist
from accelerate.utils import wait_for_everyone
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from config import DEFAULT_CONFIG_PATH, load_config
from data import load_train_validation, prepare_system_prompt, read_jsonl
from model import load_tokenizer, load_train_model

# This module implements the training script for QLoRA supervised fine-tuning.

def main():
    parser = argparse.ArgumentParser(description="QLoRA supervised fine-tuning")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--run-name", default=None, help="Optional unique name for this training run")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.run_name is None and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError(
            "For multi-GPU launch, provide --run-name; scripts/run_train.sh does this automatically."
        )
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("--run-name must be a single directory name")
    output_cfg = config.section("output")
    run_root = config.path(output_cfg.get("run_root", "outputs/runs"))
    run_dir = run_root / run_name
    is_main_process = int(os.environ.get("RANK", "0")) == 0
    if is_main_process:
        # Only rank 0 may claim/create a run directory. Other ranks must not
        # race it or make an existing run look like a newly created one.
        run_dir.mkdir(parents=True, exist_ok=False)
    wait_for_everyone()
    started_at = datetime.now(timezone.utc)
    if is_main_process:
        shutil.copy2(config.file, run_dir / "qlora.yaml")
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "status": "running",
                    "started_at": started_at.isoformat(),
                    "base_model": config.model_name_or_path(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    model_cfg = dict(config.section("model"))
    model_cfg["name_or_path"] = config.model_name_or_path()
    train_cfg = config.section("training")
    set_seed(int(train_cfg.get("seed", 42)))
    tokenizer = load_tokenizer(
        model_cfg["name_or_path"], bool(model_cfg.get("trust_remote_code", False))
    )
    train_data, validation_data = load_train_validation(config, tokenizer)
    model = load_train_model(
        model_cfg, config.section("quantization"), config.section("lora")
    )
    checkpoint_dir = run_dir / "checkpoints"
    per_device_batch_size = int(train_cfg.get("per_device_train_batch_size", 1))
    gradient_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 8))
    num_train_epochs = float(train_cfg.get("num_train_epochs", 3))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    sft_config_parameters = inspect.signature(SFTConfig).parameters
    if "warmup_ratio" in sft_config_parameters:
        warmup_args = {"warmup_ratio": warmup_ratio}
    elif "warmup_steps" in sft_config_parameters:
        # Newer TRL releases expose warmup_steps instead. Estimate total
        # optimizer updates from dataset size and distributed batch settings.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        steps_per_epoch = math.ceil(
            len(train_data) / (per_device_batch_size * world_size * gradient_accumulation_steps)
        )
        total_steps = math.ceil(steps_per_epoch * num_train_epochs)
        warmup_steps = math.ceil(warmup_ratio * total_steps)
        warmup_args = {"warmup_steps": warmup_steps}
        print(f"Optimizer steps: {total_steps}, warmup steps: {warmup_steps}")
    else:
        raise RuntimeError(
            "Installed TRL SFTConfig exposes neither warmup_ratio nor warmup_steps; "
            "check the installed TRL version."
        )
    logging_dir = run_dir / "tensorboard"
    logging_args = (
        {"logging_dir": str(logging_dir)}
        if "logging_dir" in sft_config_parameters
        else {}
    )
    training_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=int(train_cfg.get("max_seq_length", 2048)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy=train_cfg.get("eval_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(train_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(train_cfg.get("greater_is_better", False)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 2)),
        report_to=train_cfg.get("report_to", "none"),
        run_name=run_name,
        completion_only_loss=bool(train_cfg.get("completion_only_loss", True)),
        assistant_only_loss=False,
        packing=bool(train_cfg.get("packing", False)),
        **warmup_args,
        **logging_args,
    )
    # Some TRL/Transformers combinations remove logging_dir from the config
    # constructor although TensorBoard integrations still read this attribute.
    if "logging_dir" not in sft_config_parameters:
        training_args.logging_dir = str(logging_dir)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
    )
    trainer.train()
    adapter_dir = run_dir / "adapter"
    if trainer.is_world_process_zero(): # Only rank 0 saves the adapter and tokenizer to avoid race conditions
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(adapter_dir)
        training_records = read_jsonl(config.train_file)
        if not training_records:
            raise ValueError("Cannot save system prompt: training dataset is empty")
        source_system_prompt = next(
            message["content"]
            for message in training_records[0]["messages"]
            if message.get("role") == "system"
        )
        (adapter_dir / "system_prompt.txt").write_text(
            prepare_system_prompt(source_system_prompt) + "\n", encoding="utf-8"
        )
        manifest = {
            "run_name": run_name,
            "status": "completed",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "base_model": model_cfg["name_or_path"],
            "adapter_dir": str(adapter_dir),
            "system_prompt_file": str(adapter_dir / "system_prompt.txt"),
        }
        (run_dir / "run.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        latest_file = config.path(
            output_cfg.get("latest_run_file", "outputs/latest_run.json")
        )
        latest_file.parent.mkdir(parents=True, exist_ok=True)
        latest_file.write_text(
            json.dumps({"run_name": run_name}, indent=2), encoding="utf-8"
        )
        print(f"Best adapter saved to {adapter_dir}")
    wait_for_everyone()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
