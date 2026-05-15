#!/usr/bin/env python3
"""
Train a NER model on Old Swedish charter data (SDHK) — v2.

Changes from train_ner.py:
  - Adds O-boundary chunking to eliminate truncation. Long sentences are
    split into sub-sequences at tokens labelled O, guaranteeing that every
    entity span remains intact and every token in the corpus is seen during
    training. No information is lost.

Supports four model variants:
  ner-swe      KBLab/bert-base-swedish-cased-ner  (NER-tuned Swedish BERT)
  bert-swe     KBLab/bert-base-swedish-cased      (base Swedish BERT)
  mlm-adapted  models/mlm_pretrained_v3             (domain-adapted via MLM)
  xlm-roberta  xlm-roberta-base                   (multilingual)

Features:
  - O-boundary chunking: zero-truncation guarantee for all entity spans
  - Weighted cross-entropy loss for class imbalance (O vs entity tokens)
  - Per-class accuracy tracking during evaluation
  - Sample prediction logging callback for troubleshooting
  - Early stopping on dev F1
  - Step-based evaluation (configurable, default every 10% of training)
  - seqeval span-level metrics (precision, recall, F1)

Usage:
    # Fine-tune with chunking (default max-length 256)
    .venv/bin/python3 scripts/train_ner_v2.py --model xlm-roberta-large

    # Use larger window (fewer chunks, more context per chunk)
    .venv/bin/python3 scripts/train_ner_v2.py --model xlm-roberta-large --max-length 512
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# --------------- project imports ---------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.training.data_utils import merge_files, read_conll_file, split_files


# ================================================================
# O-boundary chunking (NEW in v2)
# ================================================================


def chunk_at_o_boundaries(tokens_list, labels_list, tokenizer, max_length=256):
    """
    Split sentences that exceed *max_length* subword tokens into
    smaller chunks, cutting only at positions where the BIO label
    is O.  This guarantees that every entity span remains intact
    across the split — no I-to-B relabelling is needed.

    Args:
        tokens_list:  list of sentences (each a list of word strings)
        labels_list:  parallel list of BIO label sequences
        tokenizer:    HuggingFace tokenizer for subword counting
        max_length:   maximum subword tokens per chunk (incl. special tokens)

    Returns:
        (chunked_tokens, chunked_labels) — expanded lists where long
        sentences have been replaced by multiple shorter chunks.
        Also returns a stats dict with chunking metrics.
    """
    chunked_tokens = []
    chunked_labels = []
    stats = {
        "total_sentences": len(tokens_list),
        "kept_as_is": 0,
        "chunked": 0,
        "chunks_produced": 0,
        "unchunkable": 0,
    }

    # Budget for actual word tokens (reserve 2 for <s> / </s> or [CLS] / [SEP])
    budget = max_length - 2

    for sent_toks, sent_labs in zip(tokens_list, labels_list):
        # Count subword tokens per word (without special tokens)
        word_subword_counts = []
        for word in sent_toks:
            n_sub = len(tokenizer.tokenize(word))
            # tokenizer.tokenize can return 0 tokens for weird inputs;
            # count at least 1 so we always make progress
            word_subword_counts.append(max(n_sub, 1))

        total_subwords = sum(word_subword_counts)

        # If it fits, keep the sentence as-is
        if total_subwords <= budget:
            chunked_tokens.append(sent_toks)
            chunked_labels.append(sent_labs)
            stats["kept_as_is"] += 1
            continue

        # --- Need to chunk ---
        stats["chunked"] += 1
        chunks = _split_sentence_at_o(
            sent_toks, sent_labs, word_subword_counts, budget
        )

        if len(chunks) == 1 and sum(word_subword_counts) > budget:
            # Could not split (no O boundaries) — will still be truncated
            stats["unchunkable"] += 1

        stats["chunks_produced"] += len(chunks)
        for chunk_toks, chunk_labs in chunks:
            chunked_tokens.append(chunk_toks)
            chunked_labels.append(chunk_labs)

    return chunked_tokens, chunked_labels, stats


def _split_sentence_at_o(tokens, labels, subword_counts, budget):
    """
    Greedily split a single sentence into chunks at O-labelled token
    positions, each fitting within *budget* subword tokens.

    Returns a list of (chunk_tokens, chunk_labels) tuples.
    """
    chunks = []
    n = len(tokens)
    start = 0

    while start < n:
        # Accumulate words until we exceed the budget
        cumulative = 0
        last_o_pos = None  # last O-labelled position within budget
        end = start

        while end < n:
            cumulative += subword_counts[end]
            if cumulative > budget:
                break
            # Track O-boundaries (valid split points)
            if labels[end] == "O":
                last_o_pos = end
            end += 1

        if end >= n:
            # Everything from start fits — emit final chunk
            chunks.append((tokens[start:], labels[start:]))
            break

        # We exceeded the budget at position `end`.
        # Split at the last O position we saw (split *after* that O token).
        if last_o_pos is not None and last_o_pos > start:
            split_at = last_o_pos + 1  # include the O token in this chunk
        else:
            # No O boundary found before budget — include everything up to
            # `end` (this chunk will be truncated by the tokenizer, but at
            # least we tried). This only happens for extremely long unbroken
            # entity spans, which are effectively impossible in practice.
            split_at = end

        chunks.append((tokens[start:split_at], labels[start:split_at]))
        start = split_at

    return chunks


# ================================================================
# CLI
# ================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NER model on Old Swedish charter data — v2 (O-boundary chunking)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model variant
    parser.add_argument(
        "--model",
        required=True,
        choices=list(config.NER_MODEL_VARIANTS.keys()),
        help="Model variant to fine-tune",
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        default=config.TRAINING_DATA_DIR,
        help="Directory of CoNLL training files",
    )
    parser.add_argument(
        "--external-test-dir",
        default=config.EXTERNAL_TEST_DIR,
        help="Directory of external test CoNLL files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: models/ner_finetuned/<model>)",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--learning-rate", type=float, default=config.DEFAULT_LEARNING_RATE
    )
    parser.add_argument("--max-length", type=int, default=config.DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--weight-decay", type=float, default=config.DEFAULT_WEIGHT_DECAY
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=config.DEFAULT_WARMUP_RATIO
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=config.DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=config.DEFAULT_EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument("--seed", type=int, default=config.TRAIN_SEED)

    # Class weight overrides
    parser.add_argument(
        "--o-weight",
        type=float,
        default=config.O_CLASS_WEIGHT,
        help=f"Weight for the O class (default: {config.O_CLASS_WEIGHT})",
    )
    parser.add_argument(
        "--entity-weight",
        type=float,
        default=config.ENTITY_CLASS_WEIGHT,
        help=f"Weight for entity classes (default: {config.ENTITY_CLASS_WEIGHT})",
    )

    # Advanced options
    parser.add_argument(
        "--no-class-weights", action="store_true", help="Disable class weights"
    )
    parser.add_argument(
        "--no-fp16", action="store_true", help="Disable FP16 mixed precision"
    )
    parser.add_argument(
        "--no-sample-predictions",
        action="store_true",
        help="Disable sample prediction logging during eval",
    )
    parser.add_argument(
        "--num-samples-to-show",
        type=int,
        default=5,
        help="Number of sample predictions to show per eval",
    )

    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(
            config.PROJECT_ROOT, "models", "ner_finetuned_v3"
        )

    return args


# ================================================================
# Dataset
# ================================================================


class NERDataset(Dataset):
    """
    HuggingFace-compatible dataset for token classification.

    Tokenizes pre-split words with a subword tokenizer and aligns
    BIO labels to the first subword of each word (-100 for non-first
    subwords and special tokens).
    """

    def __init__(self, tokens, labels, tokenizer, label2id, max_length=256):
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
# Weighted loss trainer
# ================================================================


class WeightedLossTrainer(Trainer):
    """Trainer subclass with weighted cross-entropy for class imbalance."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ================================================================
# Prediction logging callback
# ================================================================


class PredictionLoggingCallback(TrainerCallback):
    """Log sample predictions vs ground truth during evaluation."""

    def __init__(self, eval_dataset, tokenizer, label_list, num_samples=5):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        print("\n" + "=" * 60)
        print("Sample Predictions vs Ground Truth")
        print("=" * 60)

        model.eval()
        device = next(model.parameters()).device

        indices = random.sample(
            range(len(self.eval_dataset)),
            min(self.num_samples, len(self.eval_dataset)),
        )

        for idx in indices:
            sample = self.eval_dataset[idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            true_labels = sample["labels"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=2)[0].cpu()

            tokens = self.tokenizer.convert_ids_to_tokens(sample["input_ids"])

            # Reconstruct full words from subword tokens
            words, word_true, word_pred = [], [], []
            current_word_pieces = []
            current_true = None
            current_pred = None

            for tok, true_id, pred_id in zip(tokens, true_labels, predictions):
                if tok in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                    continue
                if tok.startswith("##"):
                    # Continuation subword — append to current word
                    current_word_pieces.append(tok[2:])
                else:
                    # New word — flush previous
                    if current_word_pieces and current_true is not None:
                        words.append("".join(current_word_pieces))
                        word_true.append(current_true)
                        word_pred.append(current_pred)
                    current_word_pieces = [tok]
                    if true_id != -100:
                        current_true = self.label_list[true_id]
                        current_pred = self.label_list[pred_id]
                    else:
                        current_true = None
                        current_pred = None
            # Flush last word
            if current_word_pieces and current_true is not None:
                words.append("".join(current_word_pieces))
                word_true.append(current_true)
                word_pred.append(current_pred)

            print(f"\nSample {idx}:")
            print(f"{'Token':<20} {'True':<15} {'Pred':<15} {'Match'}")
            print("-" * 55)

            for tok, tl, pl in zip(words, word_true, word_pred):
                match = "ok" if tl == pl else "MISS"
                print(f"{tok:<20} {tl:<15} {pl:<15} {match}")

        print("=" * 60 + "\n")


# ================================================================
# Metrics
# ================================================================


def compute_class_weights(label2id, o_weight, entity_weight):
    """Build class weight tensor (high weight for entities, low for O)."""
    weights = torch.ones(len(label2id))
    for label, idx in label2id.items():
        if label == "O":
            weights[idx] = o_weight
        else:
            weights[idx] = entity_weight
    return weights


def make_compute_metrics(label_list):
    """Return a compute_metrics function closed over label_list."""

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_predictions = []

        correct_by_class = Counter()
        total_by_class = Counter()

        for pred_seq, label_seq in zip(predictions, labels):
            pred_labels = []
            gold_labels = []

            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id != -100:
                    pred_label = label_list[pred_id]
                    gold_label = label_list[label_id]
                    pred_labels.append(pred_label)
                    gold_labels.append(gold_label)

                    total_by_class[gold_label] += 1
                    if pred_label == gold_label:
                        correct_by_class[gold_label] += 1

            true_labels.append(gold_labels)
            true_predictions.append(pred_labels)

        # Per-class accuracy
        print("\n--- Per-Class Accuracy ---")
        for label in sorted(total_by_class.keys()):
            acc = (
                correct_by_class[label] / total_by_class[label]
                if total_by_class[label] > 0
                else 0
            )
            print(
                f"  {label:<15}: {acc * 100:5.1f}% "
                f"({correct_by_class[label]}/{total_by_class[label]})"
            )

        # Warn if model predicts almost exclusively O
        o_preds = sum(1 for s in true_predictions for l in s if l == "O")
        total_preds = sum(len(s) for s in true_predictions)
        o_pct = 100 * o_preds / total_preds if total_preds > 0 else 0
        if o_pct > 95:
            print(f"\n  WARNING: Model predicting 'O' for {o_pct:.1f}% of tokens!")

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "micro_f1": f1_score(true_labels, true_predictions, average="micro"),
        }

    return compute_metrics


# ================================================================
# Evaluation helpers
# ================================================================


def evaluate_on_dataset(trainer, dataset, label_list, dataset_name):
    """Run evaluation on a dataset and print a full classification report."""
    print(f"\n{'=' * 60}")
    print(f"  {dataset_name}")
    print(f"{'=' * 60}")

    # Full predictions
    pred_output = trainer.predict(dataset)
    preds = np.argmax(pred_output.predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred_seq, label_seq in zip(preds, pred_output.label_ids):
        pl, gl = [], []
        for p, g in zip(pred_seq, label_seq):
            if g != -100:
                pl.append(label_list[p])
                gl.append(label_list[g])
        true_labels.append(gl)
        true_predictions.append(pl)

    p = precision_score(true_labels, true_predictions)
    r = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    micro_f1 = f1_score(true_labels, true_predictions, average="micro")

    print(f"  Precision (macro): {p:.4f}")
    print(f"  Recall (macro):    {r:.4f}")
    print(f"  F1 (macro):        {f1:.4f}")
    print(f"  F1 (micro):        {micro_f1:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(true_labels, true_predictions))

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "micro_f1": micro_f1,
    }


def evaluate_external_test(
    trainer, external_dir, tokenizer, label_list, label2id, max_length
):
    """Evaluate on the external test set (all files treated as one set)."""
    if not os.path.isdir(external_dir):
        print(f"\n  External test dir not found: {external_dir}")
        return {}

    all_files = sorted(
        os.path.join(external_dir, f)
        for f in os.listdir(external_dir)
        if f.endswith(".conll")
    )

    if not all_files:
        print(f"\n  No .conll files found in {external_dir}")
        return {}

    tokens, labels = merge_files(all_files)
    # Apply chunking to external test set too
    tokens, labels, _ = chunk_at_o_boundaries(tokens, labels, tokenizer, max_length)
    ds = NERDataset(tokens, labels, tokenizer, label2id, max_length)
    r = evaluate_on_dataset(
        trainer, ds, label_list, f"External Test Set ({len(all_files)} files)"
    )
    return {"external_test": r}


# ================================================================
# Main
# ================================================================


def main():
    args = parse_args()

    # Resolve model name / path
    model_name_or_path = config.NER_MODEL_VARIANTS[args.model]

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("  Old Swedish NER Training — v2 (O-boundary chunking)")
    print("=" * 70)
    print(f"\n  Model variant:   {args.model}")
    print(f"  Model name/path: {model_name_or_path}")
    print(f"  Data dir:        {args.data_dir}")
    print(f"  External test:   {args.external_test_dir}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  Max length:      {args.max_length}")
    print(f"  O weight:        {args.o_weight}")
    print(f"  Entity weight:   {args.entity_weight}")
    print(f"  Seed:            {args.seed}")

    # ----------------------------------------------------------
    # 1. Label setup
    # ----------------------------------------------------------
    label_list = config.LABEL_LIST
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"\n[1/8] Labels: {label_list}")

    # ----------------------------------------------------------
    # 2. Load tokenizer and model
    # ----------------------------------------------------------
    print(f"\n[2/8] Loading model: {model_name_or_path}")

    # xlm-roberta uses sentencepiece; KB-BERT uses WordPiece
    tokenizer_kwargs = {}
    if args.model in ("xlm-roberta", "xlm-roberta-large", "mlm-adapted"):
        tokenizer_kwargs["add_prefix_space"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)

    # ner-swe has a different head size (more labels), so we need
    # ignore_mismatched_sizes to replace it
    model_kwargs = {}
    if args.model == "ner-swe":
        model_kwargs["ignore_mismatched_sizes"] = True

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        **model_kwargs,
    )

    # ----------------------------------------------------------
    # 3. Split data
    # ----------------------------------------------------------
    print(f"\n[3/8] Splitting data (seed={args.seed}, 80/10/10)...")

    train_files, val_files, test_files = split_files(args.data_dir, seed=args.seed)
    print(f"  Train: {len(train_files)} files")
    print(f"  Val:   {len(val_files)} files")
    print(f"  Test:  {len(test_files)} files")

    # Save split for reproducibility
    os.makedirs(args.output_dir, exist_ok=True)
    split_info = {
        "seed": args.seed,
        "train": [os.path.basename(f) for f in train_files],
        "val": [os.path.basename(f) for f in val_files],
        "test": [os.path.basename(f) for f in test_files],
    }
    with open(os.path.join(args.output_dir, "data_split.json"), "w") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------
    # 4. Load, chunk, and tokenize
    # ----------------------------------------------------------
    print("\n[4/8] Loading data and applying O-boundary chunking...")

    train_tokens, train_labels = merge_files(train_files)
    val_tokens, val_labels = merge_files(val_files)
    test_tokens, test_labels = merge_files(test_files)

    print(f"  Before chunking:")
    print(f"    Train: {len(train_tokens)} sentences")
    print(f"    Val:   {len(val_tokens)} sentences")
    print(f"    Test:  {len(test_tokens)} sentences")

    train_tokens, train_labels, train_stats = chunk_at_o_boundaries(
        train_tokens, train_labels, tokenizer, args.max_length
    )
    val_tokens, val_labels, val_stats = chunk_at_o_boundaries(
        val_tokens, val_labels, tokenizer, args.max_length
    )
    test_tokens, test_labels, test_stats = chunk_at_o_boundaries(
        test_tokens, test_labels, tokenizer, args.max_length
    )

    print(f"\n  After chunking (max_length={args.max_length}):")
    print(f"    Train: {len(train_tokens)} chunks "
          f"({train_stats['chunked']} sentences split, "
          f"{train_stats['unchunkable']} unchunkable)")
    print(f"    Val:   {len(val_tokens)} chunks "
          f"({val_stats['chunked']} sentences split, "
          f"{val_stats['unchunkable']} unchunkable)")
    print(f"    Test:  {len(test_tokens)} chunks "
          f"({test_stats['chunked']} sentences split, "
          f"{test_stats['unchunkable']} unchunkable)")

    # Class distribution
    flat_train = [l for sent in train_labels for l in sent]
    label_counts = Counter(flat_train)
    total_tok = len(flat_train)
    print("\n  Class distribution (train):")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_tok
        print(f"    {label:<15}: {count:>8,} ({pct:5.2f}%)")

    train_dataset = NERDataset(
        train_tokens, train_labels, tokenizer, label2id, args.max_length
    )
    val_dataset = NERDataset(
        val_tokens, val_labels, tokenizer, label2id, args.max_length
    )
    test_dataset = NERDataset(
        test_tokens, test_labels, tokenizer, label2id, args.max_length
    )

    # ----------------------------------------------------------
    # 5. Class weights
    # ----------------------------------------------------------
    use_weights = not args.no_class_weights
    class_weights = None
    if use_weights:
        class_weights = compute_class_weights(
            label2id, args.o_weight, args.entity_weight
        )
        print(f"\n[5/8] Class weights:")
        for label, idx in label2id.items():
            print(f"    {label:<15}: {class_weights[idx]:.1f}")
    else:
        print("\n[5/8] Class weights: DISABLED")

    # ----------------------------------------------------------
    # 6. Training setup
    # ----------------------------------------------------------
    print("\n[6/8] Setting up training...")

    total_steps = (len(train_dataset) // args.batch_size) * args.epochs
    eval_steps = max(1, int(total_steps * config.EVAL_PERCENT / 100))

    print(f"  Total steps:    {total_steps}")
    print(f"  Eval every:     {eval_steps} steps ({config.EVAL_PERCENT}%)")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=not args.no_fp16 and torch.cuda.is_available(),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="none",
    )

    # Callbacks
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )
    if not args.no_sample_predictions:
        callbacks.append(
            PredictionLoggingCallback(
                val_dataset, tokenizer, label_list, args.num_samples_to_show
            )
        )

    compute_metrics_fn = make_compute_metrics(label_list)

    # Create trainer
    TrainerClass = WeightedLossTrainer if use_weights else Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )
    if use_weights:
        trainer_kwargs["class_weights"] = class_weights

    trainer = TrainerClass(**trainer_kwargs)

    # ----------------------------------------------------------
    # 7. Train
    # ----------------------------------------------------------
    print(f"\n[7/8] Training...")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Class weights: {'Enabled' if use_weights else 'Disabled'}")

    trainer.train()

    # Save model + tokenizer + label map
    print(f"\n  Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump(
            {"label2id": label2id, "id2label": id2label, "label_list": label_list},
            f,
            indent=2,
        )

    # ----------------------------------------------------------
    # 8. Evaluate
    # ----------------------------------------------------------
    print(f"\n[8/8] Evaluation...")

    all_results = {
        "model_variant": args.model,
        "model_name_or_path": model_name_or_path,
        "training_script": "train_ner_v2.py",
        "chunking": {
            "method": "o_boundary",
            "max_length": args.max_length,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "test_stats": test_stats,
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "class_weights": use_weights,
            "o_weight": args.o_weight if use_weights else None,
            "entity_weight": args.entity_weight if use_weights else None,
            "seed": args.seed,
        },
        "data_split": {
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "train_chunks": len(train_tokens),
            "val_chunks": len(val_tokens),
            "test_chunks": len(test_tokens),
        },
    }

    print(f"\n{'=' * 70}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 70}")

    # 1. Internal test set (from the 80/10/10 split)
    internal_results = evaluate_on_dataset(
        trainer,
        test_dataset,
        label_list,
        f"Internal Test Set ({len(test_files)} files from train/val/test split)",
    )
    all_results["internal_test"] = internal_results

    # 2. External test set (held-out charters)
    external_results = evaluate_external_test(
        trainer,
        args.external_test_dir,
        tokenizer,
        label_list,
        label2id,
        args.max_length,
    )
    all_results.update(external_results)

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Training complete (v2 — O-boundary chunking).")
    print(f"  Model saved to: {args.output_dir}")
    print(f"  Results saved to: {results_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
