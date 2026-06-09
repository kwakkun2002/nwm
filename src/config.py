import yaml
from functools import lru_cache


DEFAULT_EVAL_CONFIG_PATH = "configs/evaluation/eval_config.yaml"
DEFAULT_DATA_CONFIG_PATH = "configs/data/data_config.yaml"
DEFAULT_PLANNING_HYPERPARAMS_PATH = "configs/data/data_hyperparams_plan.yaml"


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_experiment_config(config_path: str, default_config_path: str = DEFAULT_EVAL_CONFIG_PATH) -> dict:
    config = load_yaml_config(default_config_path)
    config.update(load_yaml_config(config_path))
    return config


@lru_cache(maxsize=None)
def load_data_config(path: str = DEFAULT_DATA_CONFIG_PATH) -> dict:
    return load_yaml_config(path)


@lru_cache(maxsize=None)
def load_planning_hyperparams(path: str = DEFAULT_PLANNING_HYPERPARAMS_PATH) -> dict:
    return load_yaml_config(path)
