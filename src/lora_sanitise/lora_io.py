"""LoRA adapter loading and grouping helpers.

These helpers operate on adapter files only. They do not load a base model or
instantiate any Transformers classes.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ADAPTER_ID = "BackdoorLLM/Jailbreak_Llama2-7B_BadNets"
DEFAULT_ADAPTER_SNAPSHOT = Path(
    "/home/huggingface/hub/"
    "models--BackdoorLLM--Jailbreak_Llama2-7B_BadNets/"
    "snapshots/408295cd17df70e5164e7692e2aa3c5b9e2e4f3b"
)


@dataclass(frozen=True)
class LoraTensorInfo:
    key: str
    kind: str
    module_name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class LoraPairInfo:
    module_name: str
    a_key: str
    b_key: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    a_dtype: str
    b_dtype: str

    @property
    def rank(self) -> int | None:
        if len(self.a_shape) != 2 or len(self.b_shape) != 2:
            return None
        a_rank = self.a_shape[0]
        b_rank = self.b_shape[1]
        return int(a_rank) if a_rank == b_rank else None

    @property
    def target_module(self) -> str:
        return self.module_name.split(".")[-1]

    @property
    def layer_id(self) -> int | None:
        parts = self.module_name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part == "layers":
                try:
                    return int(parts[index + 1])
                except ValueError:
                    return None
        return None


def hf_hub_cache_roots(extra_roots: list[str] | None = None) -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    if os.environ.get("TRANSFORMERS_CACHE"):
        roots.append(Path(os.environ["TRANSFORMERS_CACHE"]).expanduser())
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    roots.extend([hf_home / "hub", hf_home])
    for root in extra_roots or []:
        roots.append(Path(root).expanduser())

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def repo_cache_dirname(repo_id: str, repo_type: str = "model") -> str:
    prefix = {"model": "models", "dataset": "datasets", "space": "spaces"}[repo_type]
    return f"{prefix}--{repo_id.replace('/', '--')}"


def locate_adapter_snapshot(
    adapter_path: str | Path | None = None,
    adapter_id: str = ADAPTER_ID,
    cache_roots: list[str] | None = None,
) -> Path:
    """Locate a cached adapter snapshot without network access."""
    if adapter_path:
        path = Path(adapter_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {path}")
        return path

    if DEFAULT_ADAPTER_SNAPSHOT.exists():
        return DEFAULT_ADAPTER_SNAPSHOT

    dirname = repo_cache_dirname(adapter_id, "model")
    candidates: list[Path] = []
    for root in hf_hub_cache_roots(cache_roots):
        snapshots = root / dirname / "snapshots"
        if snapshots.exists():
            candidates.extend(path for path in snapshots.iterdir() if path.is_dir())

    if not candidates:
        raise FileNotFoundError(
            f"No cached snapshots found for {adapter_id}. Pass --adapter-path explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def adapter_config_path(snapshot_path: str | Path) -> Path:
    return Path(snapshot_path).expanduser() / "adapter_config.json"


def adapter_model_path(snapshot_path: str | Path) -> Path:
    return Path(snapshot_path).expanduser() / "adapter_model.safetensors"


def load_adapter_config(snapshot_path: str | Path) -> dict[str, Any]:
    path = adapter_config_path(snapshot_path)
    return json.loads(path.read_text(encoding="utf-8"))


def classify_lora_tensor_key(key: str) -> tuple[str | None, str | None]:
    suffixes = [
        (".lora_A.weight", "A"),
        (".lora_B.weight", "B"),
        (".lora_embedding_A.weight", "A"),
        (".lora_embedding_B.weight", "B"),
    ]
    for suffix, kind in suffixes:
        if key.endswith(suffix):
            return kind, key[: -len(suffix)]

    markers = [
        (".lora_A.", "A"),
        (".lora_B.", "B"),
        (".lora_embedding_A.", "A"),
        (".lora_embedding_B.", "B"),
    ]
    for marker, kind in markers:
        if marker in key:
            return kind, key.split(marker, 1)[0]
    return None, None


def safetensors_metadata(model_path: str | Path) -> list[LoraTensorInfo]:
    from safetensors import safe_open

    rows: list[LoraTensorInfo] = []
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            kind, module_name = classify_lora_tensor_key(key)
            if kind is None or module_name is None:
                continue
            tensor_slice = handle.get_slice(key)
            rows.append(
                LoraTensorInfo(
                    key=key,
                    kind=kind,
                    module_name=module_name,
                    shape=tuple(int(dim) for dim in tensor_slice.get_shape()),
                    dtype=str(tensor_slice.get_dtype()),
                )
            )
    return rows


def group_lora_ab_pairs(model_path: str | Path) -> tuple[list[LoraPairInfo], list[str]]:
    grouped: dict[str, dict[str, LoraTensorInfo]] = defaultdict(dict)
    for info in safetensors_metadata(model_path):
        grouped[info.module_name][info.kind] = info

    pairs: list[LoraPairInfo] = []
    warnings: list[str] = []
    for module_name in sorted(grouped):
        module = grouped[module_name]
        a_info = module.get("A")
        b_info = module.get("B")
        if a_info is None or b_info is None:
            warnings.append(f"Incomplete LoRA pair for {module_name}")
            continue
        pair = LoraPairInfo(
            module_name=module_name,
            a_key=a_info.key,
            b_key=b_info.key,
            a_shape=a_info.shape,
            b_shape=b_info.shape,
            a_dtype=a_info.dtype,
            b_dtype=b_info.dtype,
        )
        if pair.rank is None:
            warnings.append(
                f"Rank mismatch for {module_name}: A {pair.a_shape}, B {pair.b_shape}"
            )
        pairs.append(pair)
    return pairs, warnings


def load_pair_tensors(handle: Any, pair: LoraPairInfo) -> tuple[Any, Any]:
    """Load one LoRA A/B tensor pair from an open safetensors handle."""
    return handle.get_tensor(pair.a_key), handle.get_tensor(pair.b_key)
