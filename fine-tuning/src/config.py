from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

FINE_TUNING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = FINE_TUNING_ROOT / "configs" / "qlora.yaml"

# This module provides a simple configuration loader for YAML files.
# It defines a Config dataclass that holds the raw configuration data,
# the path to the configuration file, and the root directory for resolving relative paths.
# The load_config function reads a YAML file and returns a Config instance.

@dataclass(frozen=True)
class Config:
    """A configuration for the fine-tuning process."""
    raw: dict[str, Any]
    file: Path
    root: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section {name!r} must be a mapping")
        return value

    def path(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        return (self.root / candidate).resolve() if not candidate.is_absolute() else candidate

    def model_name_or_path(self) -> str:
        """Resolve a local model path relative to this config, or retain a Hub ID."""
        configured = self.section("model")["name_or_path"]
        candidate = self.path(configured).resolve()
        return str(candidate) if candidate.exists() else configured

    def resolve_adapter(self, run_name: str | None = None) -> tuple[str | None, Path]:
        """Resolve a named/latest run adapter, with support for the previous fixed adapter path."""
        output = self.section("output")
        if run_name is None:
            latest_file = self.path(
                output.get("latest_run_file", "outputs/latest_run.json")
            )
            if latest_file.exists():
                import json

                run_name = json.loads(latest_file.read_text(encoding="utf-8"))["run_name"]
        if run_name is not None:
            if Path(run_name).name != run_name or run_name in {".", ".."}:
                raise ValueError("run name must be a single directory name")
            run_root = self.path(output.get("run_root", "outputs/runs"))
            adapter = run_root / run_name / "adapter"
            if not adapter.is_dir():
                raise FileNotFoundError(f"No adapter found for run {run_name!r}: {adapter}")
            return run_name, adapter
        legacy = output.get("adapter_dir")
        if legacy:
            adapter = self.path(legacy)
            if adapter.is_dir():
                return None, adapter
        raise FileNotFoundError(
            "No trained adapter is available. Train a run first or pass --run-name."
        )

    @property
    def train_file(self) -> Path:
        return self.path(self.section("data")["train_file"])

    @property
    def validation_file(self) -> Path:
        return self.path(self.section("data")["validation_file"])

    @property
    def test_file(self) -> Path:
        return self.path(self.section("data")["test_file"])


def load_config(path: str | Path) -> Config:
    """Load a YAML configuration file and return a Config instance."""
    
    config_file = Path(path).expanduser().resolve()
    with config_file.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_file}")
    # qlora.yaml lives in configs/, so project-relative paths start at fine-tuning/.
    return Config(raw=raw, file=config_file, root=config_file.parent.parent)
