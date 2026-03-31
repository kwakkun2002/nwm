import json
import os
import re
from typing import Dict, Iterable


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
