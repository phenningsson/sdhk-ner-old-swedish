#!/usr/bin/env python3
"""
Reproduce the 80/10/10 train/val/test split used by train_ner_v2.py and
copy the 43 test files out of data/1380_1382_dataset/ into their own
directory, so the same files can be fed to evaluate_ner_v2.py and
evaluate_pipeline.py.

The split is file-level, shuffled with random.Random(seed=42).shuffle
inside src.training.data_utils.split_files — deterministic, so this
script re-derives the exact same 43 files that train_ner_v2.py held
out during training. A JSON manifest is also written for audit.

Output:
    data/internal_test_set/
        sdhk_<id>_silver.conll         × 43
        split_manifest.json            (seed, full lists for all splits)
"""

import argparse
import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.training.data_utils import split_files


def main():
    parser = argparse.ArgumentParser(
        description="Copy the 43 internal test files to a standalone dir."
    )
    parser.add_argument("--data-dir",   default=config.TRAINING_DATA_DIR)
    parser.add_argument("--output-dir", default=os.path.join(
        PROJECT_ROOT, "data", "internal_test_set"))
    parser.add_argument("--seed",       type=int, default=config.TRAIN_SEED)
    args = parser.parse_args()

    train_files, val_files, test_files = split_files(args.data_dir, seed=args.seed)

    print(f"Split (seed={args.seed}, 80/10/10) of {args.data_dir}:")
    print(f"  train: {len(train_files)}")
    print(f"  val:   {len(val_files)}")
    print(f"  test:  {len(test_files)}")

    os.makedirs(args.output_dir, exist_ok=True)

    for src in test_files:
        dst = os.path.join(args.output_dir, os.path.basename(src))
        shutil.copyfile(src, dst)

    manifest = {
        "seed":       args.seed,
        "source_dir": args.data_dir,
        "output_dir": args.output_dir,
        "train":      [os.path.basename(f) for f in train_files],
        "val":        [os.path.basename(f) for f in val_files],
        "test":       [os.path.basename(f) for f in test_files],
    }
    manifest_path = os.path.join(args.output_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nCopied {len(test_files)} test files to:")
    print(f"  {args.output_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
