#!/usr/bin/env python3
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from text_pipeline import caption_to_tags, iter_jsonl, normalize_caption_text, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--format", type=str, default="sentence", choices=["sentence", "tags"])
    args = parser.parse_args()

    output_records = []
    for record in iter_jsonl(args.input):
        raw_caption = record.get("raw_caption", "")
        if args.format == "sentence":
            clean_text = normalize_caption_text(raw_caption)
        else:
            clean_text = caption_to_tags(raw_caption)

        output_records.append(
            {
                **record,
                "clean_format": args.format,
                "clean_text": clean_text,
            }
        )

    write_jsonl(args.output, output_records)


if __name__ == "__main__":
    main()
