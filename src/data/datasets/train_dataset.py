import numpy as np
import torch
from typing import Optional, Tuple

from src.data.transforms.action import normalize_data
from src.data.io import load_traj_image
from src.data.datasets.base_dataset import BaseDataset


class TrainingDataset(BaseDataset):
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
        traj_names: str = 'traj_names.txt',
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
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1, size=(self.goals_per_obs))
            goal_time = (curr_time + goal_offset).astype('int')
            rel_time = (goal_offset).astype('float')/(128.) # TODO: refactor, currently a fixed const

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            context = [(f_curr, t) for t in context_times] + [(f_curr, t) for t in goal_time]

            obs_image = torch.stack([self.transform(load_traj_image(self.data_folder, f, t)) for f, t in context])

            # Load other trajectory data
            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
            goal_pos[:, :2] = normalize_data(goal_pos[:, :2], self.ACTION_STATS)
            text_condition = self._build_text_condition(f_curr, curr_time, goal_time, context_times)

            outputs = [
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(rel_time, dtype=torch.float32),
            ]
            if text_condition is not None:
                outputs.append(torch.as_tensor(text_condition, dtype=torch.float32))
            return tuple(outputs)
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
