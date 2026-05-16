#!/usr/bin/env python3
"""
Evaluate the fine-tuned Old Swedish NER model from HuggingFace Hub on a
local directory of CoNLL test files. CPU-only; no GPU required.

Mirrors EXACTLY the preprocessing protocol used by scripts/train_ner_v2.py
for train/val/test:
  · read CoNLL via src.training.data_utils.merge_files (drops middle-dot
    tokens U+2027 — editorial separators);
  · tokenizer loaded with add_prefix_space=True (required for the
    xlm-roberta-large SentencePiece tokenizer used by the mlm-adapted
    base model);
  · O-boundary chunking at max_length=512 subwords, imported from
    train_ner_v2.py so the two scripts cannot drift;
  · NERDataset with is_split_into_words=True and first-subword label
    alignment (identical to the training Dataset class).

Differences vs. GPU training:
  · CUDA disabled (CUDA_VISIBLE_DEVICES=''); use_cpu=True;
  · fp16/bf16 disabled (fp32 inference — tiny numerical difference from
    any GPU run that used autocast, but deterministic on CPU);
  · lower default batch size.

Usage:
    export HF_TOKEN=hf_xxx…                       # private repo access
    .venv/bin/python3 scripts/evaluate_ner_v2.py  # all defaults
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)

# Import chunker directly from the training script so train and eval
# cannot drift apart — identical function, identical behaviour.
from train_ner_v2 import chunk_at_o_boundaries

import config
from src.evaluation.report import compute_and_report
from src.training.data_utils import merge_files

DEFAULT_MODEL_ID = "phenningsson/sdhk-ner-old-swedish-v2"
DEFAULT_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "expert_gold_test")
DEFAULT_MAX_LENGTH = 512  # matches ner_results.json: chunking.max_length


# ================================================================
# Dataset (identical to train_ner_v2.NERDataset)
# ================================================================


class NERDataset(Dataset):
    def __init__(self, tokens, labels, tokenizer, label2id, max_length=512):
        self.tokens = tokens
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(labels):
                    aligned_labels.append(self.label2id.get(labels[word_idx], 0))
                else:
                    aligned_labels.append(-100)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


# ================================================================
# Main
# ================================================================


def resolve_hf_token(cli_token):
    if cli_token:
        return cli_token
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the HF-hosted Old Swedish NER model on a "
        "CoNLL test directory, using the EXACT protocol of "
        "train_ner_v2.py (CPU-only)."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token for private repos. Falls back to "
        "$HF_TOKEN / $HUGGING_FACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Subword budget per chunk; must match training "
        f"(default {DEFAULT_MAX_LENGTH}).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to write the JSON results. Defaults to "
        "eval_results_<safe-model-id>.json in the project "
        "root — set this explicitly to keep expert-gold and "
        "internal-test-set runs from overwriting each other.",
    )
    args = parser.parse_args()

    # ── Force CPU ──
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    hf_token = resolve_hf_token(args.hf_token)

    label_list = config.LABEL_LIST
    label2id = {l: i for i, l in enumerate(label_list)}

    print("=" * 60)
    print("  NER Evaluation (HF Hub, CPU, train-v2 protocol)")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Test dir:   {args.test_dir}")
    print(f"  Max length: {args.max_length}  (matches training)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  HF token:   {'set' if hf_token else 'not set (public only)'}")

    # Version fingerprint
    import platform
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    import transformers as _tf

    def _v(pkg):
        try:
            return _pkg_version(pkg)
        except PackageNotFoundError:
            return "unknown"

    seqeval_ver = _v("seqeval")
    print()
    print("  Reproducibility fingerprint:")
    print(f"    python        {platform.python_version()}")
    print(f"    torch         {torch.__version__}")
    print(f"    transformers  {_tf.__version__}")
    print(f"    seqeval       {seqeval_ver}")
    print(f"    numpy         {np.__version__}")
    print(f"    platform      {platform.platform()}")
    print()

    # ── Tokenizer + model ──
    # add_prefix_space=True matches train_ner_v2.py for xlm-roberta /
    # mlm-adapted variants. Passing it is safe even if already set in the
    # saved tokenizer_config.json (same value → no-op).
    tokenizer_kwargs = {"add_prefix_space": True}
    model_kwargs = {}
    if hf_token:
        tokenizer_kwargs["token"] = hf_token
        model_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    model = AutoModelForTokenClassification.from_pretrained(args.model, **model_kwargs)

    if getattr(model.config, "id2label", None):
        hub_labels = [model.config.id2label[i] for i in sorted(model.config.id2label)]
        if hub_labels != label_list:
            print("  ⚠ model id2label differs from config.LABEL_LIST:")
            print(f"    model : {hub_labels}")
            print(f"    config: {label_list}")
            print("    Using model's own id2label ordering.")
            label_list = hub_labels
            label2id = {l: i for i, l in enumerate(label_list)}

    # ── Test data ──
    all_files = sorted(
        os.path.join(args.test_dir, f)
        for f in os.listdir(args.test_dir)
        if f.endswith(".conll")
    )
    if not all_files:
        print(f"  No .conll files found in {args.test_dir}")
        return

    print(f"  Files:      {len(all_files)}")
    tokens, labels = merge_files(all_files)
    pre_sent = len(tokens)
    pre_ent = sum(1 for s in labels for l in s if l.startswith("B-"))

    # ── O-boundary chunking (SAME function as training) ──
    tokens, labels, chunk_stats = chunk_at_o_boundaries(
        tokens, labels, tokenizer, args.max_length
    )

    dataset = NERDataset(tokens, labels, tokenizer, label2id, args.max_length)

    post_ent = sum(1 for s in labels for l in s if l.startswith("B-"))
    total_tokens = sum(len(s) for s in tokens)
    entity_tokens = sum(1 for s in labels for l in s if l != "O")

    print(f"\n  Chunking (max_length={args.max_length}, O-boundary):")
    print(f"    Input sentences:   {pre_sent}")
    print(f"    Kept as-is:        {chunk_stats['kept_as_is']}")
    print(
        f"    Chunked:           {chunk_stats['chunked']}  "
        f"→ {chunk_stats['chunks_produced']} chunks"
    )
    print(
        f"    Unchunkable:       {chunk_stats['unchunkable']}  "
        f"(0 = zero truncation of entity spans)"
    )
    print(f"    Final sequences:   {len(tokens)}")
    print()
    print(f"  Tokens:         {total_tokens}")
    print(
        f"  Entity tokens:  {entity_tokens} ({100 * entity_tokens / total_tokens:.1f}%)"
    )
    print(f"  Gold entities:  {post_ent}  (pre-chunk: {pre_ent} — should be equal)")
    if post_ent != pre_ent:
        print(f"    ⚠ entity count changed during chunking — investigate.")
    print()

    # ── Inference ──
    training_args = TrainingArguments(
        output_dir="/tmp/eval_ner_v2",
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        report_to="none",
        fp16=False,
        bf16=False,
        use_cpu=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    pred_output = trainer.predict(dataset)
    preds = np.argmax(pred_output.predictions, axis=2)

    true_labels, true_predictions = [], []
    for pred_seq, label_seq in zip(preds, pred_output.label_ids):
        pl, gl = [], []
        for p, g in zip(pred_seq, label_seq):
            if g != -100:
                pl.append(label_list[p])
                gl.append(label_list[g])
        true_labels.append(gl)
        true_predictions.append(pl)

    # --- Compute all metrics, print, and dump JSON — shared with
    # evaluate_pipeline.py via src/evaluation/report.py so the two
    # eval scripts cannot drift apart.
    import re as _re

    safe_model = _re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    out_path = args.output_json or os.path.join(
        PROJECT_ROOT, f"eval_results_{safe_model}.json"
    )

    extra_fields = {
        "model": args.model,
        "test_dir": args.test_dir,
        "num_files": len(all_files),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": "cpu",
        "protocol": "train_ner_v2 (O-boundary chunking, add_prefix_space=True)",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _tf.__version__,
            "seqeval": seqeval_ver,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "data": {
            "input_sentences": pre_sent,
            "chunks_produced": len(tokens),
            "chunked_sentences": chunk_stats["chunked"],
            "unchunkable": chunk_stats["unchunkable"],
            "total_tokens": total_tokens,
            "entity_tokens": entity_tokens,
            "gold_entities": post_ent,
        },
    }
    compute_and_report(
        true_labels,
        true_predictions,
        dataset.tokens,
        source="model",
        output_path=out_path,
        extra_fields=extra_fields,
    )


if __name__ == "__main__":
    main()
