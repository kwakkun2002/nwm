import os
import pickle
import warnings
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image

from recon_datavis.utils import bytes2im, get_files_ending_with


def collect_hdf5_files(folder_or_folders):
    """Collect HDF5 files from one or more folders."""
    return get_files_ending_with(folder_or_folders, ".hdf5")


def collect_processed_trajectory_dirs(folder_or_folders):
    """Collect trajectory directories containing JPG frames and traj_data.pkl."""
    if isinstance(folder_or_folders, (str, os.PathLike)):
        folder = os.fspath(folder_or_folders)
        if not os.path.isdir(folder):
            return []

        trajectory_dirs = []
        for name in sorted(os.listdir(folder)):
            traj_dir = os.path.join(folder, name)
            if not os.path.isdir(traj_dir):
                continue
            if not os.path.isfile(os.path.join(traj_dir, "traj_data.pkl")):
                continue
            if not any(fname.endswith(".jpg") for fname in os.listdir(traj_dir)):
                continue
            trajectory_dirs.append(traj_dir)
        return trajectory_dirs

    trajectory_dirs = []
    for folder in folder_or_folders:
        trajectory_dirs.extend(collect_processed_trajectory_dirs(folder))
    return sorted(trajectory_dirs)


class JupyterHDF5Visualizer:
    """Render RECON HDF5 samples inside Jupyter without a GUI window."""

    def __init__(self, hdf5_fnames: Sequence[str]):
        if not hdf5_fnames:
            raise ValueError("No HDF5 files were provided.")
        self._hdf5_fnames = list(hdf5_fnames)
        self._invalid_hdf5_fnames = []
        self._frame_count_map = None

    @property
    def hdf5_fnames(self):
        return tuple(self._hdf5_fnames)

    @property
    def invalid_hdf5_fnames(self):
        return tuple(self._invalid_hdf5_fnames)

    @property
    def num_files(self):
        return len(self._hdf5_fnames)

    def get_file_length(self, file_idx: int):
        path = self._hdf5_fnames[file_idx]
        try:
            with h5py.File(path, "r") as hdf5_file:
                return len(hdf5_file["collision/any"])
        except OSError as exc:
            raise OSError(f"Failed to open HDF5 file: {path}") from exc

    def get_frame_count_map(self, skip_invalid: bool = True, warn_on_skip: bool = False):
        if self._frame_count_map is not None:
            return list(self._frame_count_map)

        valid_hdf5_fnames = []
        invalid_hdf5_fnames = []
        frame_count_map = []

        for path in self._hdf5_fnames:
            try:
                with h5py.File(path, "r") as hdf5_file:
                    frame_count_map.append(len(hdf5_file["collision/any"]))
                valid_hdf5_fnames.append(path)
            except OSError as exc:
                if not skip_invalid:
                    raise OSError(f"Failed to open HDF5 file: {path}") from exc
                invalid_hdf5_fnames.append(path)

        if not valid_hdf5_fnames:
            raise ValueError("No readable HDF5 files were found.")

        self._hdf5_fnames = valid_hdf5_fnames
        self._invalid_hdf5_fnames = invalid_hdf5_fnames
        self._frame_count_map = frame_count_map

        if invalid_hdf5_fnames and warn_on_skip:
            warnings.warn(
                f"Skipped {len(invalid_hdf5_fnames)} unreadable HDF5 file(s). "
                f"First skipped file: {invalid_hdf5_fnames[0]}",
                RuntimeWarning,
            )

        return list(self._frame_count_map)

    def render(self, file_idx: int = 0, timestep: int = 0, figsize=(20, 10)):
        file_idx = int(np.clip(file_idx, 0, self.num_files - 1))
        with h5py.File(self._hdf5_fnames[file_idx], "r") as hdf5_file:
            file_len = len(hdf5_file["collision/any"])
            timestep = int(np.clip(timestep, 0, file_len - 1))

            fig, axes = plt.subplots(2, 4, figsize=figsize)
            ((ax_rgb_left, ax_rgb_right, ax_thermal, ax_lidar),
             (ax_gpscompass, ax_coll, ax_imu, ax_speedsteer)) = axes

            self._plot_rgb_left(ax_rgb_left, hdf5_file, timestep)
            self._plot_rgb_right(ax_rgb_right, hdf5_file, timestep)
            self._plot_thermal(ax_thermal, hdf5_file, file_idx, timestep, file_len)
            self._plot_lidar(ax_lidar, hdf5_file, timestep)
            self._plot_gpscompass(ax_gpscompass, hdf5_file, timestep)
            self._plot_collision(ax_coll, hdf5_file, timestep)
            self._plot_imu(ax_imu, hdf5_file, timestep)
            self._plot_speedsteer(ax_speedsteer, hdf5_file, timestep)

            fig.tight_layout()
            return fig, axes

    def _get_topic(self, hdf5_file: h5py.File, topic: str, timestep: int):
        value = hdf5_file[topic][timestep]
        if isinstance(value, np.bytes_):
            value = bytes2im(value)
        return value

    def _plot_rgb_left(self, ax, hdf5_file, timestep):
        ax.imshow(self._get_topic(hdf5_file, "images/rgb_left", timestep))
        ax.set_title("RGB Left")
        ax.axis("off")

    def _plot_rgb_right(self, ax, hdf5_file, timestep):
        ax.imshow(self._get_topic(hdf5_file, "images/rgb_right", timestep))
        illuminance = self._get_topic(hdf5_file, "android/illuminance", timestep)
        ax.text(
            0.03,
            0.04,
            f"Illuminance: {illuminance:.0f}",
            transform=ax.transAxes,
            fontsize=11,
            color="w",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
        )
        ax.set_title("RGB Right")
        ax.axis("off")

    def _plot_thermal(self, ax, hdf5_file, file_idx, timestep, file_len):
        thermal = self._get_topic(hdf5_file, "images/thermal", timestep)
        ax.imshow(thermal, cmap="inferno")
        ax.text(
            0.03,
            0.04,
            (
                f"{os.path.basename(self._hdf5_fnames[file_idx])}\n"
                f"frame {timestep + 1}/{file_len}, file {file_idx + 1}/{self.num_files}"
            ),
            transform=ax.transAxes,
            fontsize=11,
            color="w",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
        )
        ax.set_title("Thermal")
        ax.axis("off")

    def _plot_lidar(self, ax, hdf5_file, timestep):
        lidar = np.array(self._get_topic(hdf5_file, "lidar", timestep), dtype=float)
        if len(lidar) != 360:
            raise ValueError(f"Expected 360 lidar beams, got {len(lidar)}.")

        measurement_indices = list(range(300, 360)) + list(range(0, 60))
        angles = np.deg2rad(np.linspace(30, 150.0, 120))
        lidar = lidar[measurement_indices]
        lidar_notfinite = np.logical_not(np.isfinite(lidar))
        max_range = 15.0
        lidar[lidar_notfinite] = max_range

        x = lidar * np.cos(angles)
        y = lidar * np.sin(angles)
        colors = [to_rgb("r") if invalid else to_rgb("k") for invalid in lidar_notfinite]

        ax.scatter(x, y, c=colors, s=6.0)
        ax.set_xlim((-16, 16))
        ax.set_ylim((0, 16))
        ax.set_aspect("equal")
        ax.set_title("Lidar")

    def _plot_speedsteer(self, ax, hdf5_file, timestep):
        commanded_linvel = self._get_topic(hdf5_file, "commands/linear_velocity", timestep)
        commanded_angvel = -self._get_topic(hdf5_file, "commands/angular_velocity", timestep)
        actual_linvel = self._get_topic(hdf5_file, "jackal/linear_velocity", timestep)
        actual_angvel = -self._get_topic(hdf5_file, "jackal/angular_velocity", timestep)

        ax.plot([0, 0], [0, commanded_linvel], linestyle="-", color="k", linewidth=10.0, label="Commanded")
        ax.plot([0, commanded_angvel], [0, 0], linestyle="-", color="k", linewidth=10.0)
        ax.plot([0, 0], [0, actual_linvel], linestyle="-", color="c", linewidth=4.0, label="Actual")
        ax.plot([0, actual_angvel], [0, 0], linestyle="-", color="c", linewidth=4.0)
        ax.legend(loc="lower left")
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5))
        ax.set_title("Speed / Steer")
        ax.axhline(0, color="0.8", linewidth=1.0)
        ax.axvline(0, color="0.8", linewidth=1.0)

    def _plot_collision(self, ax, hdf5_file, timestep):
        topics = [
            "collision/any",
            "collision/physical",
            "collision/close",
            "collision/flipped",
            "collision/stuck",
            "collision/outside_geofence",
        ]
        collisions = [self._get_topic(hdf5_file, topic, timestep) for topic in topics]
        labels = [topic.replace("collision/", "").replace("outside_", "") for topic in topics]
        ax.bar(np.arange(len(collisions)), collisions, tick_label=labels, color="r")
        ax.set_title("Collision")
        ax.set_xlim((-0.5, len(collisions) - 0.5))
        ax.set_ylim((0, 1))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    def _plot_imu(self, ax, hdf5_file, timestep):
        linacc = np.array(self._get_topic(hdf5_file, "imu/linear_acceleration", timestep), dtype=float)
        angvel = np.array(self._get_topic(hdf5_file, "imu/angular_velocity", timestep), dtype=float)
        jackal_linacc = np.array(
            self._get_topic(hdf5_file, "jackal/imu/linear_acceleration", timestep),
            dtype=float,
        )
        jackal_angvel = np.array(
            self._get_topic(hdf5_file, "jackal/imu/angular_velocity", timestep),
            dtype=float,
        )

        linacc[2] = -linacc[2] + 9.81
        jackal_linacc[2] -= 9.81

        x = np.arange(6)
        ax.bar(x, np.concatenate((linacc, angvel)), tick_label=["linacc", "", "", "angvel", "", ""], color="k", width=0.8, label="External")
        ax.bar(x, np.concatenate((jackal_linacc, jackal_angvel)), tick_label=["linacc", "", "", "angvel", "", ""], color="r", width=0.5, label="Jackal")
        ax.set_title("IMU")
        ax.set_xlim((-0.5, 6.5))
        ax.set_ylim((-3.0, 3.0))
        ax.legend(loc="upper right")
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    def _plot_gpscompass(self, ax, hdf5_file, timestep):
        latlong_all = np.array(hdf5_file["gps/latlong"][:], dtype=float)
        compass_bearing = self._get_topic(hdf5_file, "imu/compass_bearing", timestep)
        current_latlong = latlong_all[timestep]
        valid_mask = np.isfinite(latlong_all).all(axis=1)

        ax.set_title("GPS / Compass")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if valid_mask.any():
            valid_latlong = latlong_all[valid_mask]
            ax.plot(valid_latlong[:, 1], valid_latlong[:, 0], color="tab:blue", linewidth=2.0, label="Path")

            if np.isfinite(current_latlong).all():
                arrow_dx = 2e-5 * np.cos(compass_bearing)
                arrow_dy = 2e-5 * np.sin(compass_bearing)
                ax.scatter(current_latlong[1], current_latlong[0], color="r", s=30, label="Current")
                ax.arrow(
                    current_latlong[1],
                    current_latlong[0],
                    arrow_dx,
                    arrow_dy,
                    color="r",
                    width=5e-7,
                    head_width=3e-6,
                    length_includes_head=True,
                )
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No valid GPS fix", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])


class JupyterProcessedTrajectoryVisualizer:
    """Render processed RECON trajectories (JPG frames + traj_data.pkl) inside Jupyter."""

    def __init__(self, trajectory_dirs: Sequence[str]):
        if not trajectory_dirs:
            raise ValueError("No processed trajectory directories were provided.")
        self._trajectory_dirs = list(trajectory_dirs)
        self._frame_count_map = None
        self._frame_files = {}
        self._traj_data = {}

    @property
    def trajectory_dirs(self):
        return tuple(self._trajectory_dirs)

    @property
    def num_files(self):
        return len(self._trajectory_dirs)

    def _get_trajectory_dir(self, file_idx: int) -> str:
        return self._trajectory_dirs[file_idx]

    def _get_frame_files(self, file_idx: int):
        traj_dir = self._get_trajectory_dir(file_idx)
        if traj_dir not in self._frame_files:
            frame_files = []
            for fname in os.listdir(traj_dir):
                if fname.endswith(".jpg"):
                    stem, _ = os.path.splitext(fname)
                    try:
                        frame_idx = int(stem)
                    except ValueError:
                        continue
                    frame_files.append((frame_idx, os.path.join(traj_dir, fname)))
            frame_files.sort()
            self._frame_files[traj_dir] = [path for _, path in frame_files]
        return self._frame_files[traj_dir]

    def _get_traj_data(self, file_idx: int):
        traj_dir = self._get_trajectory_dir(file_idx)
        if traj_dir not in self._traj_data:
            with open(os.path.join(traj_dir, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            traj_data["position"] = np.asarray(traj_data["position"], dtype=float)
            traj_data["yaw"] = np.asarray(traj_data["yaw"], dtype=float)
            self._traj_data[traj_dir] = traj_data
        return self._traj_data[traj_dir]

    def get_file_length(self, file_idx: int):
        frame_files = self._get_frame_files(file_idx)
        traj_data = self._get_traj_data(file_idx)
        return min(len(frame_files), len(traj_data["position"]), len(traj_data["yaw"]))

    def get_frame_count_map(self):
        if self._frame_count_map is None:
            self._frame_count_map = [self.get_file_length(file_idx) for file_idx in range(self.num_files)]
        return list(self._frame_count_map)

    def render(self, file_idx: int = 0, timestep: int = 0, figsize=(14, 6)):
        file_idx = int(np.clip(file_idx, 0, self.num_files - 1))
        frame_files = self._get_frame_files(file_idx)
        traj_data = self._get_traj_data(file_idx)
        file_len = self.get_file_length(file_idx)
        timestep = int(np.clip(timestep, 0, file_len - 1))

        positions = traj_data["position"][:file_len]
        yaw = traj_data["yaw"][:file_len]
        image = np.asarray(Image.open(frame_files[timestep]).convert("RGB"))

        fig, (ax_image, ax_traj) = plt.subplots(1, 2, figsize=figsize)
        self._plot_processed_image(ax_image, image, file_idx, timestep, file_len)
        self._plot_processed_trajectory(ax_traj, positions, yaw, timestep)
        fig.tight_layout()
        return fig, (ax_image, ax_traj)

    def _plot_processed_image(self, ax, image, file_idx, timestep, file_len):
        traj_name = os.path.basename(self._trajectory_dirs[file_idx])
        ax.imshow(image)
        ax.text(
            0.03,
            0.04,
            (
                f"{traj_name}\n"
                f"frame {timestep + 1}/{file_len}, trajectory {file_idx + 1}/{self.num_files}"
            ),
            transform=ax.transAxes,
            fontsize=11,
            color="w",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
        )
        ax.set_title("RGB")
        ax.axis("off")

    def _plot_processed_trajectory(self, ax, positions, yaw, timestep):
        current_position = positions[timestep]
        current_yaw = yaw[timestep]

        ax.plot(positions[:, 0], positions[:, 1], color="tab:blue", linewidth=2.0, label="Path")
        ax.scatter(current_position[0], current_position[1], color="r", s=40, label="Current")

        extent = np.ptp(positions, axis=0)
        arrow_scale = max(0.5, float(np.max(extent)) * 0.08)
        ax.arrow(
            current_position[0],
            current_position[1],
            arrow_scale * np.cos(current_yaw),
            arrow_scale * np.sin(current_yaw),
            color="r",
            width=arrow_scale * 0.03,
            head_width=arrow_scale * 0.15,
            length_includes_head=True,
        )

        ax.set_title("Trajectory")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.legend(loc="best")
