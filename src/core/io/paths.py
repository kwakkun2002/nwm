import os

DEFAULT_WEIGHTS_ROOT = os.environ.get("NWM_WEIGHTS_DIR", "weights")
DEFAULT_CHECKPOINT_ROOT = os.path.join(DEFAULT_WEIGHTS_ROOT, "checkpoints")
DEFAULT_ARTIFACT_ROOT = os.environ.get("NWM_ARTIFACT_DIR", "artifacts")
DEFAULT_BULK_ARTIFACT_ROOT = os.path.join(DEFAULT_ARTIFACT_ROOT, "bulk")
DEFAULT_SUMMARY_ARTIFACT_ROOT = os.path.join(DEFAULT_ARTIFACT_ROOT, "summaries")
DEFAULT_TRAIN_ARTIFACT_ROOT = os.path.join(DEFAULT_BULK_ARTIFACT_ROOT, "train")
DEFAULT_EVAL_ARTIFACT_ROOT = os.path.join(DEFAULT_BULK_ARTIFACT_ROOT, "eval")
DEFAULT_PLANNING_ARTIFACT_ROOT = os.path.join(DEFAULT_BULK_ARTIFACT_ROOT, "planning")
DEFAULT_SMOKE_ARTIFACT_ROOT = os.path.join(DEFAULT_ARTIFACT_ROOT, "smoke")


def get_run_log_dir(config: dict) -> str:
    return os.path.join(config.get("results_dir", "logs"), config["run_name"])


def get_run_artifact_dir(config: dict) -> str:
    return os.path.join(config.get("artifact_dir", DEFAULT_TRAIN_ARTIFACT_ROOT), config["run_name"])


def get_run_checkpoint_dir(config: dict) -> str:
    return os.path.join(config.get("weights_dir", DEFAULT_CHECKPOINT_ROOT), config["run_name"])


def get_checkpoint_path(config: dict, checkpoint_name: str) -> str:
    checkpoint_dir = get_run_checkpoint_dir(config)
    if checkpoint_name.endswith(".pth.tar"):
        filename = checkpoint_name
    else:
        filename = f"{checkpoint_name}.pth.tar"
    return os.path.join(checkpoint_dir, filename)


def get_eval_bulk_root(suite_name: str, artifact_root: str = DEFAULT_ARTIFACT_ROOT) -> str:
    return os.path.join(artifact_root, "bulk", "eval", suite_name)


def get_eval_summary_root(suite_name: str, artifact_root: str = DEFAULT_ARTIFACT_ROOT) -> str:
    return os.path.join(artifact_root, "summaries", "eval", suite_name)


def get_planning_bulk_root(suite_name: str, artifact_root: str = DEFAULT_ARTIFACT_ROOT) -> str:
    return os.path.join(artifact_root, "bulk", "planning", suite_name)


def get_preprocess_summary_root(name: str, artifact_root: str = DEFAULT_ARTIFACT_ROOT) -> str:
    return os.path.join(artifact_root, "summaries", "preprocess", name)


def get_smoke_artifact_dir(name: str, artifact_root: str = DEFAULT_ARTIFACT_ROOT) -> str:
    return os.path.join(artifact_root, "smoke", name)
