from pathlib import Path
from typing import Any


def _wandb_config(config: dict) -> dict:
    wandb_config = dict(config.get("wandb", {}))
    if config.get("use_wandb", False):
        wandb_config.setdefault("enabled", True)
    return wandb_config


def wandb_enabled(config: dict) -> bool:
    return bool(_wandb_config(config).get("enabled", False))


def init_wandb_run(config: dict, rank: int, job_type: str, logger=None):
    if rank != 0 or not wandb_enabled(config):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("WandB logging is enabled, but wandb is not installed.") from exc

    wandb_config = _wandb_config(config)
    init_kwargs: dict[str, Any] = {
        "project": wandb_config.get("project", "nwm"),
        "name": wandb_config.get("name") or config.get("run_name"),
        "config": config,
        "job_type": job_type,
        "resume": wandb_config.get("resume", "allow"),
    }
    for key in ("entity", "group", "mode", "tags", "dir"):
        value = wandb_config.get(key)
        if value not in (None, "", []):
            init_kwargs[key] = value

    run = wandb.init(**init_kwargs)
    if logger is not None:
        logger.info(f"WandB run initialized: {run.name}")
    return run


def log_metrics(run, metrics: dict[str, Any], step: int | None = None) -> None:
    if run is not None:
        run.log(metrics, step=step)


def log_images(run, image_paths, key: str, step: int | None = None, max_images: int = 10) -> None:
    if run is None:
        return

    try:
        import wandb
    except ImportError:
        return

    paths = [Path(path) for path in image_paths]
    images = [wandb.Image(str(path)) for path in paths[:max_images] if path.exists()]
    if images:
        run.log({key: images}, step=step)


def log_artifact(run, path: str, name: str, artifact_type: str = "checkpoint") -> None:
    if run is None:
        return

    try:
        import wandb
    except ImportError:
        return

    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(path)
    run.log_artifact(artifact)


def finish_wandb_run(run) -> None:
    if run is not None:
        run.finish()
