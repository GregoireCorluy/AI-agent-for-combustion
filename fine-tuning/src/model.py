from __future__ import annotations

import os

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# This module provides utilities for loading and configuring models for fine-tuning.

def torch_dtype(name: str):
    supported = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in supported:
        raise ValueError(f"Unsupported compute dtype {name!r}")
    return supported[name]


def load_tokenizer(model_name: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        clean_up_tokenization_spaces=False, # could add a "add_special_tokens=False" option to avoid problems with other models which add speical tokens with the tokenizer
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _quantization(quant_config: dict):
    return BitsAndBytesConfig(
        load_in_4bit=bool(quant_config.get("load_in_4bit", True)),
        bnb_4bit_quant_type=quant_config.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(quant_config.get("double_quant", True)),
        bnb_4bit_compute_dtype=torch_dtype(quant_config.get("compute_dtype", "bfloat16")),
    )


def load_train_model(model_config: dict, quant_config: dict, lora_config: dict):
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires CUDA and a CUDA-enabled PyTorch installation")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dtype = torch_dtype(quant_config.get("compute_dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        quantization_config=_quantization(quant_config),
        dtype=dtype,
        device_map={"": local_rank},
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_config.get("r", 16)),
            lora_alpha=int(lora_config.get("alpha", 32)),
            lora_dropout=float(lora_config.get("dropout", 0.05)),
            target_modules=lora_config.get("target_modules", "all-linear"),
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "CAUSAL_LM"),
        ),
    )
    model.print_trainable_parameters()
    return model


def load_base_model(model_config: dict, quant_config: dict):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dtype = torch_dtype(quant_config.get("compute_dtype", "bfloat16"))
    return AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        quantization_config=_quantization(quant_config),
        dtype=dtype,
        device_map={"": local_rank},
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
