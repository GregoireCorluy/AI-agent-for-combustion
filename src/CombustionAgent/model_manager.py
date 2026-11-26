"""Load and serve role-specific local Hugging Face models."""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    multimodal: bool


class ModelManager:
    """Registry of model roles. Models in the initial configuration load eagerly."""

    def __init__(self, models: dict[str, dict[str, Any]] | None = None):
        self.specs: dict[str, dict[str, Any]] = {}
        self.loaded: dict[str, LoadedModel] = {}
        for name, spec in (models or {}).items():
            self.register_model(name, spec, load=True)

    def register_model(self, name: str, spec: dict[str, Any], load: bool = False) -> None:
        """Register another role and optionally load it immediately."""
        self.specs[name] = dict(spec)
        if load:
            self.load_model(name)

    def load_model(self, name: str) -> LoadedModel:
        if name in self.loaded:
            return self.loaded[name]
        if name not in self.specs:
            raise KeyError(f"No model configuration registered for role {name!r}")

        spec = self.specs[name]
        alias_of = spec.get("alias_of")
        if alias_of:
            self.loaded[name] = self.load_model(alias_of)
            return self.loaded[name]
        model_id = spec["model"]
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=bool(spec.get("trust_remote_code", False))
        )
        # Qwen3.8 is a multimodal checkpoint. Use its AutoProcessor and multimodal
        # Auto class rather than the AutoModelForCausalLM path used by the old loader.
        multimodal = bool(spec.get("multimodal", False)) or hasattr(config, "vision_config") or "qwen3_5" in getattr(config, "model_type", "")
        processor = (
            AutoProcessor.from_pretrained(model_id, trust_remote_code=bool(spec.get("trust_remote_code", False)))
            if multimodal
            else AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False,
                                               trust_remote_code=bool(spec.get("trust_remote_code", False)))
        )

        quantization = spec.get("quantization", "4bit")
        kwargs: dict[str, Any] = {"device_map": spec.get("device_map", "auto"),
                                  "trust_remote_code": bool(spec.get("trust_remote_code", False))}
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=spec.get("quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(spec.get("double_quant", True)),
                bnb_4bit_compute_dtype=self._dtype(spec.get("compute_dtype", "bfloat16")),
            )
        elif quantization in {"bfloat16", "float16", "float32"}:
            kwargs["dtype"] = self._dtype(quantization)
        elif quantization != "none":
            raise ValueError(f"Unsupported quantization {quantization!r} for model {name!r}")

        if multimodal:
            from transformers import AutoModelForMultimodalLM
            model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

        adapter = spec.get("adapter")
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        if hasattr(processor, "pad_token") and processor.pad_token is None:
            processor.pad_token = processor.eos_token

        loaded = LoadedModel(model=model, processor=processor, multimodal=multimodal)
        self.loaded[name] = loaded
        return loaded

    @staticmethod
    def _dtype(name: str) -> torch.dtype:
        try:
            return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                    "float32": torch.float32}[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported torch dtype {name!r}") from exc

    def generate(self, role: str, messages: list[dict[str, str]], *, max_new_tokens: int = 1024,
                 do_sample: bool = False, enable_thinking: bool = False) -> str:
        loaded = self.loaded.get(role) or self.load_model(role)
        processor, model = loaded.processor, loaded.model
        tokenizer = getattr(processor, "tokenizer", processor)
        template_kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
        if loaded.multimodal:
            template_kwargs["return_dict"] = True
        try:
            inputs = processor.apply_chat_template(
                messages, enable_thinking=enable_thinking, **template_kwargs
            )
        except TypeError:
            inputs = processor.apply_chat_template(messages, **template_kwargs)
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        else:
            inputs = {key: value.to(model.device) if hasattr(value, "to") else value
                      for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=do_sample,
                                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                                    eos_token_id=getattr(tokenizer, "eos_token_id", None))
        input_ids = inputs["input_ids"]
        new_ids = output[0][input_ids.shape[-1]:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=False)
        if "</think>" in raw:
            raw = raw.rsplit("</think>", 1)[1]
            return tokenizer.decode(tokenizer.encode(raw, add_special_tokens=False),
                                     skip_special_tokens=True).strip()
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def unload_model(self, name: str | None = None) -> None:
        names = [name] if name else list(self.loaded)
        for role in names:
            self.loaded.pop(role, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
