#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from misc import load_traj_image
from text_pipeline import PROMPT_TEMPLATES, discover_image_records, iter_jsonl, write_jsonl


def load_records(input_root: str = None, manifest_path: str = None) -> List[dict]:
    if manifest_path:
        return list(iter_jsonl(manifest_path))
    if input_root:
        return discover_image_records(input_root)
    raise ValueError("Either --input-root or --manifest must be provided")


def shard_records(records: List[dict], num_shards: int, shard_index: int) -> List[dict]:
    if num_shards <= 1:
        return records
    return [record for index, record in enumerate(records) if index % num_shards == shard_index]


def load_qwen_model(model_name_or_path: str, dtype: str, attn_implementation: str = None):
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Failed to import transformers for Qwen2-VL loading. "
            "Check that transformers and huggingface-hub versions are compatible in the active environment."
        ) from exc

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()
    return processor, model


def build_prompt(prompt_mode: str, prompt_text: str = None) -> str:
    if prompt_text:
        return prompt_text
    if prompt_mode not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    return PROMPT_TEMPLATES[prompt_mode]


def load_record_image(record: dict, data_root: str = None):
    image_path = record.get("image_path")
    if image_path:
        from PIL import Image

        return Image.open(image_path).convert("RGB")

    trajectory_name = record.get("trajectory_name")
    frame_time = record.get("frame_time")
    if data_root and trajectory_name is not None and frame_time is not None:
        return load_traj_image(data_root, trajectory_name, int(frame_time))

    raise ValueError(
        "Record must include either image_path or trajectory_name/frame_time with --data-root. "
        f"Got: {record}"
    )


def build_messages(batch_records: List[dict], prompt: str, data_root: str = None):
    messages = []
    images = []
    for record in batch_records:
        image = load_record_image(record, data_root=data_root)
        images.append(image)
        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )
    return messages, images


@torch.inference_mode()
def generate_batch(processor, model, batch_records: List[dict], prompt: str, max_new_tokens: int, data_root: str = None):
    messages, images = build_messages(batch_records, prompt, data_root=data_root)
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    outputs = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return [output.strip() for output in outputs]


def main():
    parser = argparse.ArgumentParser()
    default_weights_root = os.environ.get("NWM_WEIGHTS_DIR", "weights")
    default_qwen_path = os.path.join(default_weights_root, "pretrained", "Qwen2-VL-7B-Instruct")
    parser.add_argument("--model-name-or-path", type=str, default=default_qwen_path)
    parser.add_argument("--input-root", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prompt-mode", type=str, default="scene_only", choices=sorted(PROMPT_TEMPLATES.keys()))
    parser.add_argument("--prompt-text", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    records = load_records(input_root=args.input_root, manifest_path=args.manifest)
    records = shard_records(records, num_shards=args.num_shards, shard_index=args.shard_index)
    if not records:
        raise ValueError("No records matched the requested shard")

    prompt = build_prompt(args.prompt_mode, prompt_text=args.prompt_text)
    processor, model = load_qwen_model(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    output_records = []
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start:start + args.batch_size]
        captions = generate_batch(
            processor=processor,
            model=model,
            batch_records=batch_records,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            data_root=args.data_root,
        )
        for record, caption in zip(batch_records, captions):
            output_records.append(
                {
                    **record,
                    "prompt_mode": args.prompt_mode,
                    "prompt_text": prompt,
                    "raw_caption": caption,
                }
            )
        print(f"Processed {min(start + args.batch_size, len(records))}/{len(records)}")

    write_jsonl(args.output, output_records)


if __name__ == "__main__":
    main()
