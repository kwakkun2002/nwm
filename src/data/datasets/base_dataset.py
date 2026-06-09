# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import numpy as np
import os
from typing import Optional, Tuple
import pickle
import tqdm
from torch.utils.data import Dataset

from src.config import load_data_config
from src.data.transforms.action import angle_difference, to_local_coords
from src.data.io import load_traj_data
from src.features.text.pipeline import build_text_cache_path, infer_text_embedding_dim


class BaseDataset(Dataset):
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
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.goals_per_obs = goals_per_obs


        traj_names_file = os.path.join(data_split_folder, traj_names)
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.distance_categories = list(range(min_dist_cat, max_dist_cat + 1))
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.context_size = context_size
        self.normalize = normalize
        self.text_embedding_root = text_embedding_root
        self.text_condition_source = text_condition_source
        self._text_cache = {}
        self.text_embedding_dim = 0
        self._zero_text_embedding = None

        all_data_config = load_data_config()

        self.data_config = all_data_config[self.dataset_name]
        self.transform = transform
        self._load_index(predefined_index)
        self.ACTION_STATS = {}
        for key in all_data_config['action_stats']:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)
        if self.text_embedding_root:
            self.text_embedding_dim = infer_text_embedding_dim(self.text_embedding_root)
            self._zero_text_embedding = np.zeros((self.text_embedding_dim,), dtype=np.float32)

    def _load_index(self, predefined_index) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        if predefined_index:
            print(f"****** Using a predefined evaluation index... {predefined_index}******")
            with open(predefined_index, "rb") as f:
                self.index_to_data = pickle.load(f)
                return
        else:
            print("****** Evaluating from NON PREDEFINED index... ******")
            index_to_data_path = os.path.join(
                self.data_split_folder,
                f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_n{self.context_size}_len_traj_pred_{self.len_traj_pred}.pkl",
            )

            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                min_goal_distance = max(self.min_dist_cat, -curr_time)
                samples_index.append((traj_name, curr_time, min_goal_distance, max_goal_distance))

        return samples_index, goals_index

    def _get_trajectory(self, trajectory_name):
        return load_traj_data(self.data_folder, trajectory_name)

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _load_text_cache(self, trajectory_name: str):
        if trajectory_name in self._text_cache:
            return self._text_cache[trajectory_name]

        if not self.text_embedding_root:
            return None

        text_cache_path = build_text_cache_path(self.text_embedding_root, trajectory_name)
        if not os.path.isfile(text_cache_path):
            self._text_cache[trajectory_name] = None
            return None

        with np.load(text_cache_path, allow_pickle=False) as text_data:
            times = text_data["times"].astype(np.int64)
            embeddings = text_data["embeddings"].astype(np.float32)

        cache = {
            "times": times,
            "embeddings": embeddings,
            "time_to_index": {int(time): idx for idx, time in enumerate(times.tolist())},
        }
        self._text_cache[trajectory_name] = cache
        return cache

    def _get_text_embedding(self, trajectory_name: str, frame_time: int) -> np.ndarray:
        if not self.text_embedding_root:
            raise RuntimeError("Text embeddings requested but text_embedding_root is not configured")

        cache = self._load_text_cache(trajectory_name)
        if cache is None:
            return self._zero_text_embedding.copy()

        index = cache["time_to_index"].get(int(frame_time))
        if index is None:
            return self._zero_text_embedding.copy()
        return cache["embeddings"][index]

    def _build_text_condition(self, trajectory_name: str, curr_time: int, goal_time, context_times) -> Optional[np.ndarray]:
        if not self.text_embedding_root:
            return None

        goal_times = np.atleast_1d(goal_time).astype(int)
        num_goals = len(goal_times)

        if self.text_condition_source == "current":
            curr_embedding = self._get_text_embedding(trajectory_name, int(curr_time))
            return np.repeat(curr_embedding[None], num_goals, axis=0)

        if self.text_condition_source == "goal":
            return np.stack([self._get_text_embedding(trajectory_name, frame_time) for frame_time in goal_times], axis=0)

        if self.text_condition_source == "context_mean":
            context_embeddings = np.stack(
                [self._get_text_embedding(trajectory_name, frame_time) for frame_time in context_times],
                axis=0,
            )
            context_mean = context_embeddings.mean(axis=0)
            return np.repeat(context_mean[None], num_goals, axis=0)

        raise ValueError(f"Unsupported text_condition_source: {self.text_condition_source}")

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["position"][start_index:end_index]
        goal_pos = traj_data["position"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("is used?")

        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
        actions = actions[1:]

        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw = angle_difference(yaw[0], goal_yaw)

        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"]
            goal_pos[:, :2] /= self.data_config["metric_waypoint_spacing"]

        goal_pos = np.concatenate([goal_pos, goal_yaw.reshape(-1, 1)], axis=-1)
        return actions, goal_pos
