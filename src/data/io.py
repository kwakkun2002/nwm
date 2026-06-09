import io
import os
import pickle

import numpy as np
from PIL import Image

try:
    import h5py
except ImportError:
    h5py = None


def get_data_path(data_folder: str, trajectory_name: str, time: int, data_type: str = "image"):
    data_ext = {
        "image": ".jpg",
    }
    return os.path.join(data_folder, trajectory_name, f"{str(time)}{data_ext[data_type]}")


def get_raw_recon_hdf5_path(data_folder: str, trajectory_name: str) -> str:
    return os.path.join(data_folder, f"{trajectory_name}.hdf5")


def load_traj_data(data_folder: str, trajectory_name: str):
    processed_traj_path = os.path.join(data_folder, trajectory_name, "traj_data.pkl")
    if os.path.isfile(processed_traj_path):
        with open(processed_traj_path, "rb") as f:
            traj_data = pickle.load(f)
        for key, value in traj_data.items():
            traj_data[key] = value.astype("float")
        return traj_data

    raw_recon_path = get_raw_recon_hdf5_path(data_folder, trajectory_name)
    if os.path.isfile(raw_recon_path):
        if h5py is None:
            raise ImportError(
                "h5py is required to read raw RECON .hdf5 files. "
                "Install it in the active environment first."
            )
        with h5py.File(raw_recon_path, "r") as h5_f:
            return {
                "position": np.asarray(h5_f["jackal"]["position"][:, :2], dtype="float"),
                "yaw": np.asarray(h5_f["jackal"]["yaw"][()], dtype="float"),
            }

    raise FileNotFoundError(
        f"Could not find processed trajectory or raw RECON file for {trajectory_name} under {data_folder}"
    )


def load_traj_image(data_folder: str, trajectory_name: str, time: int) -> Image.Image:
    processed_image_path = get_data_path(data_folder, trajectory_name, time)
    if os.path.isfile(processed_image_path):
        return Image.open(processed_image_path).convert("RGB")

    raw_recon_path = get_raw_recon_hdf5_path(data_folder, trajectory_name)
    if os.path.isfile(raw_recon_path):
        if h5py is None:
            raise ImportError(
                "h5py is required to read raw RECON .hdf5 files. "
                "Install it in the active environment first."
            )
        with h5py.File(raw_recon_path, "r") as h5_f:
            image_bytes = h5_f["images"]["rgb_left"][time]
            if hasattr(image_bytes, "tobytes"):
                image_bytes = image_bytes.tobytes()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    raise FileNotFoundError(
        f"Could not find processed image or raw RECON frame for {trajectory_name} at t={time} under {data_folder}"
    )
