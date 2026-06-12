from functools import lru_cache
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_EVAL_CONFIG_PATH = "configs/evaluation/eval_config.yaml"
DEFAULT_DATA_CONFIG_PATH = "configs/data/data_config.yaml"
DEFAULT_PLANNING_HYPERPARAMS_PATH = "configs/data/data_hyperparams_plan.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def load_yaml_config(path: str) -> dict:
    with open(resolve_repo_path(path), "r") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _strip_hydra_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key != "defaults" and not (isinstance(key, str) and key.startswith("#"))
    }


def _resolve_default_path(default: Any, current_path: Path) -> Path | None:
    if default == "_self_":
        return None

    if isinstance(default, str):
        if default.startswith("/"):
            rel = default[1:]
        else:
            rel = str(current_path.parent.relative_to(CONFIG_ROOT)) + "/" + default
        return CONFIG_ROOT / f"{rel}.yaml"

    if isinstance(default, dict) and len(default) == 1:
        group, option = next(iter(default.items()))
        if option is None:
            return None
        group = str(group)
        if group.startswith("/"):
            rel = f"{group[1:]}/{option}"
        else:
            current_group = current_path.parent.relative_to(CONFIG_ROOT)
            rel = f"{current_group}/{group}/{option}"
        return CONFIG_ROOT / f"{rel}.yaml"

    raise ValueError(f"Unsupported Hydra default entry in {current_path}: {default!r}")


def load_hydra_yaml_config(path: str | Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved_path = resolve_repo_path(path).resolve()
    seen = seen or set()
    if resolved_path in seen:
        raise ValueError(f"Recursive Hydra defaults include detected at {resolved_path}")
    seen.add(resolved_path)

    config = load_yaml_config(resolved_path)
    defaults = config.get("defaults", [])
    body = _strip_hydra_metadata(config)
    if not defaults:
        seen.remove(resolved_path)
        return body

    merged: dict[str, Any] = {}
    applied_self = False
    for default in defaults:
        if default == "_self_":
            deep_update(merged, body)
            applied_self = True
            continue
        default_path = _resolve_default_path(default, resolved_path)
        if default_path is not None:
            deep_update(merged, load_hydra_yaml_config(default_path, seen))

    if not applied_self:
        deep_update(merged, body)

    seen.remove(resolved_path)
    return merged


def load_experiment_config(config_path: str, default_config_path: str = DEFAULT_EVAL_CONFIG_PATH) -> dict:
    config = load_yaml_config(default_config_path)
    deep_update(config, load_hydra_yaml_config(config_path))
    return config


def compose_hydra_config(
    overrides: Iterable[str] | None = None,
    config_name: str = "config",
) -> dict[str, Any]:
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "Hydra config mode requires hydra-core. Install the environment from env.yaml "
            "or run the legacy --config/--exp command."
        ) from exc

    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))
    return OmegaConf.to_container(cfg, resolve=True)


def load_runtime_config(args: Any, config_attr: str = "config", default_config_path: str = DEFAULT_EVAL_CONFIG_PATH) -> dict:
    runtime_config = getattr(args, "runtime_config", None)
    if runtime_config is not None:
        return dict(runtime_config)
    return load_experiment_config(getattr(args, config_attr), default_config_path=default_config_path)


def save_yaml_config(config: dict, path: str | Path) -> None:
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def namespace_from_config(config: dict, section: str, **extra: Any) -> SimpleNamespace:
    base = config.get(section, {})
    values = dict(base) if isinstance(base, dict) else {}
    values.update(extra)
    values["runtime_config"] = config
    return SimpleNamespace(**values)


def update_runtime_section(config: dict, section: str, args: Any, keys: Iterable[str]) -> dict:
    base = config.get(section, {})
    section_config = dict(base) if isinstance(base, dict) else {}
    for key in keys:
        if hasattr(args, key):
            section_config[key] = getattr(args, key)
    config[section] = section_config
    return config


def int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item]
    return [int(item) for item in value]


def str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.split(",") if item]
    return [str(item) for item in value]


@lru_cache(maxsize=None)
def load_data_config(path: str = DEFAULT_DATA_CONFIG_PATH) -> dict:
    return load_yaml_config(path)


@lru_cache(maxsize=None)
def load_planning_hyperparams(path: str = DEFAULT_PLANNING_HYPERPARAMS_PATH) -> dict:
    return load_yaml_config(path)
