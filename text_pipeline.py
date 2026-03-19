import json
import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np


PROMPT_TEMPLATES = {
    "scene_only": (
        "Describe only the visible scene in one short sentence. "
        "Focus on layout, terrain, obstacles, and navigable space. "
        "Do not mention the image itself, do not speculate, and do not include actions."
    ),
    "scene_goal": (
        "Describe the visible scene and the likely near-term navigation affordance in one short sentence. "
        "Focus on layout, terrain, obstacles, free space, and immediate path cues. "
        "Do not mention the image itself and do not speculate beyond what is visible."
    ),
    "scene_tags": (
        "Return only 6 to 12 short tags describing the visible scene, separated by commas. "
        "Focus on terrain, obstacles, structures, and free space. "
        "Do not mention the image itself."
    ),
}

TEXT_BOILERPLATE_PATTERNS = [
    r"^\s*this image (shows|depicts|contains)\s+",
    r"^\s*the image (shows|depicts|contains)\s+",
    r"^\s*the scene (shows|depicts|features|contains)\s+",
    r"^\s*in this image[, ]+",
    r"^\s*we can see\s+",
    r"^\s*you can see\s+",
    r"^\s*there is\s+",
    r"^\s*there are\s+",
]

STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "that", "the", "this", "to", "with",
}


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


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


def normalize_caption_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    text = text.strip(" \t\r\n.,;:")

    lowered = text.lower()
    for pattern in TEXT_BOILERPLATE_PATTERNS:
        lowered = re.sub(pattern, "", lowered, flags=re.IGNORECASE)
    lowered = re.sub(r"\s+", " ", lowered).strip(" \t\r\n.,;:")
    return lowered


def caption_to_tags(text: str, max_tags: int = 12) -> str:
    text = normalize_caption_text(text)
    if not text:
        return ""

    tokens = re.split(r"[^a-z0-9]+", text.lower())
    deduped = []
    seen = set()
    for token in tokens:
        if not token or token in STOPWORDS or token in seen:
            continue
        deduped.append(token)
        seen.add(token)
        if len(deduped) >= max_tags:
            break
    return ", ".join(deduped)
