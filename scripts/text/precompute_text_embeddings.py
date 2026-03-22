#!/usr/bin/env python3
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from text_pipeline import build_text_cache_path, infer_frame_time, infer_trajectory_name, iter_jsonl


def load_text_encoder(model_name_or_path: str, dtype: str):
    try:
        from transformers import AutoModel, AutoTokenizer, CLIPTextModelWithProjection
    except ImportError as exc:
        raise ImportError(
            "Failed to import transformers for text embedding precompute. "
            "Check that transformers and huggingface-hub versions are compatible in the active environment."
        ) from exc

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    try:
        model = CLIPTextModelWithProjection.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
    except Exception:
        model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


@torch.inference_mode()
def encode_batch(tokenizer, model, device: str, texts, max_length: int):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded)

    if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        embeddings = outputs.text_embeds
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embeddings = outputs.pooler_output
    else:
        embeddings = outputs.last_hidden_state[:, 0]

    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings.detach().cpu().to(torch.float16).numpy()


def canonicalize_record(record, input_root: str = None):
    trajectory_name = record.get("trajectory_name")
    frame_time = record.get("frame_time")
    image_path = record.get("image_path")

    if trajectory_name is None and image_path is not None:
        trajectory_name = infer_trajectory_name(image_path, input_root=input_root)
    if frame_time is None and image_path is not None:
        frame_time = infer_frame_time(image_path)
    if trajectory_name is None or frame_time is None:
        raise ValueError(f"Record is missing trajectory alignment fields: {record}")

    clean_text = record.get("clean_text")
    if clean_text is None:
        clean_text = record.get("raw_caption", "")

    return trajectory_name, int(frame_time), clean_text


def save_grouped_embeddings(grouped_records, grouped_embeddings, output_root: str):
    os.makedirs(output_root, exist_ok=True)
    for trajectory_name, items in grouped_records.items():
        times = np.asarray([item["frame_time"] for item in items], dtype=np.int32)
        embeddings = grouped_embeddings[trajectory_name]

        sort_idx = np.argsort(times)
        times = times[sort_idx]
        embeddings = embeddings[sort_idx]

        output_path = build_text_cache_path(output_root, trajectory_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, times=times, embeddings=embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--input-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    records = list(iter_jsonl(args.input))
    if not records:
        raise ValueError(f"No caption records found in {args.input}")

    tokenizer, model, device = load_text_encoder(args.model_name_or_path, dtype=args.dtype)

    grouped_records = defaultdict(list)
    grouped_embeddings = defaultdict(list)

    for start in range(0, len(records), args.batch_size):
        batch_records = records[start:start + args.batch_size]
        batch_texts = []
        batch_keys = []
        for record in batch_records:
            trajectory_name, frame_time, clean_text = canonicalize_record(record, input_root=args.input_root)
            batch_texts.append(clean_text)
            batch_keys.append((trajectory_name, frame_time))

        batch_embeddings = encode_batch(
            tokenizer=tokenizer,
            model=model,
            device=device,
            texts=batch_texts,
            max_length=args.max_length,
        )

        for (trajectory_name, frame_time), embedding in zip(batch_keys, batch_embeddings):
            grouped_records[trajectory_name].append({"frame_time": frame_time})
            grouped_embeddings[trajectory_name].append(embedding)

        print(f"Encoded {min(start + args.batch_size, len(records))}/{len(records)}")

    grouped_embeddings = {
        trajectory_name: np.asarray(embeddings, dtype=np.float16)
        for trajectory_name, embeddings in grouped_embeddings.items()
    }
    save_grouped_embeddings(grouped_records, grouped_embeddings, args.output_root)


if __name__ == "__main__":
    main()
