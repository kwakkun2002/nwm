from src.data.datasets.eval_dataset import EvalDataset
from src.data.datasets.trajectory_eval_dataset import TrajectoryEvalDataset
from src.data.transforms.image import build_transform
from src.features.text.pipeline import get_text_conditioning_config


def _text_dataset_kwargs(config: dict) -> dict:
    text_config = get_text_conditioning_config(config)
    return {
        "text_embedding_root": text_config["embedding_root"] if text_config["enabled"] else None,
        "text_condition_source": text_config["condition_source"],
    }


def build_eval_dataset(config: dict, dataset_name: str, eval_type: str, predefined_index: bool = True) -> EvalDataset:
    data_config = config["eval_datasets"][dataset_name]
    predefined_index_path = f"data/splits/{dataset_name}/test/{eval_type}.pkl" if predefined_index else None

    return EvalDataset(
        data_folder=data_config["data_folder"],
        data_split_folder=data_config["test"],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
        max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
        len_traj_pred=config["eval_len_traj_pred"],
        traj_stride=config["traj_stride"],
        context_size=config["eval_context_size"],
        normalize=config["normalize"],
        transform=build_transform(config["image_size"]),
        goals_per_obs=data_config.get("goals_per_obs", 4),
        predefined_index=predefined_index_path,
        traj_names="traj_names.txt",
        **_text_dataset_kwargs(config),
    )


def build_trajectory_eval_dataset(config: dict, dataset_name: str, predefined_index: bool = True) -> TrajectoryEvalDataset:
    data_config = config["eval_datasets"][dataset_name]
    predefined_index_path = f"data/splits/{dataset_name}/test/navigation_eval.pkl" if predefined_index else None

    return TrajectoryEvalDataset(
        data_folder=data_config["data_folder"],
        data_split_folder=data_config["test"],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        min_dist_cat=config["trajectory_eval_distance"]["min_dist_cat"],
        max_dist_cat=config["trajectory_eval_distance"]["max_dist_cat"],
        len_traj_pred=config["trajectory_eval_len_traj_pred"],
        traj_stride=config["traj_stride"],
        context_size=config["trajectory_eval_context_size"],
        normalize=config["normalize"],
        transform=build_transform(config["image_size"]),
        predefined_index=predefined_index_path,
        traj_names="rollout_traj_names.txt",
        **_text_dataset_kwargs(config),
    )
