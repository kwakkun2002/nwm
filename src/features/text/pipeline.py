import os
from typing import Dict, List, Optional

import numpy as np


def get_text_conditioning_config(config):
    text_config = config.get("text_conditioning", {})
    enabled = bool(text_config.get("enabled", False))
    embedding_root = text_config.get("embedding_root")
    text_dim = int(text_config.get("text_dim", 0))
    if enabled and text_dim <= 0 and embedding_root:
        text_dim = infer_text_embedding_dim(embedding_root)
    return {
        "enabled": enabled,
        "embedding_root": embedding_root,
        "condition_source": text_config.get("condition_source", "current"),
        "text_dim": text_dim,
    }


def infer_trajectory_name(image_path: str, input_root: Optional[str] = None) -> str:
    normalized_path = os.path.normpath(image_path)
    parent_dir = os.path.dirname(normalized_path)
    if input_root:
        try:
            rel_parent = os.path.relpath(parent_dir, input_root)
            if rel_parent != ".":
                return rel_parent
        except ValueError:
            pass
    return os.path.basename(parent_dir)


def infer_frame_time(image_path: str) -> Optional[int]:
    basename = os.path.splitext(os.path.basename(image_path))[0]
    if basename.isdigit():
        return int(basename)
    return None


def build_text_cache_path(output_root: str, trajectory_name: str) -> str:
    return os.path.join(output_root, f"{trajectory_name}.npz")


def infer_text_embedding_dim(output_root: str) -> int:
    for root, _, files in os.walk(output_root):
        for filename in sorted(files):
            if not filename.endswith(".npz"):
                continue
            path = os.path.join(root, filename)
            with np.load(path, allow_pickle=False) as text_data:
                embeddings = text_data["embeddings"]
                if embeddings.ndim != 2:
                    raise ValueError(f"Expected 2D embeddings in {path}, got shape {embeddings.shape}")
                return int(embeddings.shape[-1])
    raise FileNotFoundError(f"Could not find text embedding cache under {output_root}")


def discover_image_records(
    input_root: str,
    image_extensions: Optional[List[str]] = None,
) -> List[Dict]:
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    allowed_exts = {ext.lower() for ext in image_extensions}
    records = []
    for root, _, files in os.walk(input_root):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in allowed_exts:
                continue
            image_path = os.path.join(root, filename)
            records.append(
                {
                    "image_path": image_path,
                    "trajectory_name": infer_trajectory_name(image_path, input_root=input_root),
                    "frame_time": infer_frame_time(image_path),
                }
            )
    records.sort(key=lambda record: (record["trajectory_name"], record.get("frame_time") or -1, record["image_path"]))
    return records
