from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .resolver import resolve_config


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load YAML configs. "
            "Install it with `pip install pyyaml`."
        ) from e

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return {} if data is None else data


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a config file from JSON or YAML.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return load_json(path)

    if suffix in {".yaml", ".yml"}:
        return load_yaml(path)

    raise ValueError(
        f"Unsupported config file extension: {suffix!r}. "
        "Supported: .json, .yaml, .yml"
    )


def load_and_resolve_config(path: str | Path) -> dict[str, Any]:
    """
    Load config from disk and merge with package defaults.
    """
    loaded = load_config(path)
    return resolve_config(loaded)