import numpy as np
import torch
from typing import Optional, Tuple

from src.data.transforms.action import normalize_data, get_delta_np
from src.data.io import load_traj_image
from src.data.datasets.base_dataset import BaseDataset


class EvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
        text_embedding_root: Optional[str] = None,
        text_condition_source: str = "current",
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs,
            text_embedding_root, text_condition_source)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, _, _ = self.index_to_data[i]
            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))

            context = [(f_curr, t) for t in context_times]
            pred = [(f_curr, t) for t in pred_times]

            obs_image = torch.stack([self.transform(load_traj_image(self.data_folder, f, t)) for f, t in context])
            pred_image = torch.stack([self.transform(load_traj_image(self.data_folder, f, t)) for f, t in pred])

            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            actions, _ = self._compute_actions(curr_traj_data, curr_time, np.array([curr_time+1])) # last argument is dummy goal
            actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
            delta = get_delta_np(actions)
            text_condition = self._build_text_condition(f_curr, curr_time, np.array([curr_time]), context_times)

            outputs = [
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(pred_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
            ]
            if text_condition is not None:
                outputs.append(torch.as_tensor(text_condition, dtype=torch.float32))
            return tuple(outputs)
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
