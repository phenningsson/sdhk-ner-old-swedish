#!/usr/bin/env python3
"""
Domain-adapt a BERT model via Masked Language Modelling (MLM) on
Old Swedish edition texts from SDHK charters.

Extracts Edition fields from the raw scraped JSON files and runs
continued pre-training with standard MLM (15% mask probability).
The resulting model can then be fine-tuned for NER via train_ner.py
using --model mlm-adapted.

Default hyperparameters are aligned with the Old Icelandic MLM
pipeline (batch 8 x grad_accum 4 = effective batch 32, lr 3e-5,
cosine schedule, warmup 6%, 8 epochs).

Usage:
    # Default settings (all three JSON files, 8 epochs)
    .venv/bin/python3 scripts/pretrain_mlm.py

    # Custom settings
    .venv/bin/python3 scripts/pretrain_mlm.py \\
        --epochs 20 --batch-size 8 --base-model KBLab/bert-base-swedish-cased
"""

import argparse
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --------------- project imports ---------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.training.data_utils import extract_edition_texts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Domain-adapt a BERT model via MLM on Old Swedish texts",
    )
    parser.add_argument(
        "--base-model",
        default="xlm-roberta-large",
        help="Base model to continue pre-training (default: xlm-roberta-large)",
    )
    parser.add_argument(
        "--strip-middle-dots",
        action="store_true",
        help="Strip editorial middle-dot separators (U+2027) from edition texts",
    )
    parser.add_argument(
        "--output-dir",
        default=config.MLM_PRETRAINED_DIR,
        help="Output directory for the adapted model",
    )
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_MLM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_MLM_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=config.DEFAULT_MLM_GRADIENT_ACCUMULATION,
        help=f"Gradient accumulation steps (default: {config.DEFAULT_MLM_GRADIENT_ACCUMULATION}, "
             f"effective batch = batch-size * this)",
    )
    parser.add_argument(
        "--mlm-probability",
        type=float,
        default=config.DEFAULT_MLM_PROBABILITY,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=config.DEFAULT_MLM_MAX_LENGTH,
    )
    parser.add_argument("--seed", type=int, default=config.TRAIN_SEED)
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable FP16 mixed precision")
    parser.add_argument(
        "--learning-rate", type=float, default=config.DEFAULT_MLM_LEARNING_RATE,
        help=f"Learning rate (default: {config.DEFAULT_MLM_LEARNING_RATE})",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=config.DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=config.DEFAULT_MLM_WARMUP_RATIO,
        help=f"Warmup ratio (default: {config.DEFAULT_MLM_WARMUP_RATIO})",
    )
    parser.add_argument(
        "--eval-steps", type=int, default=config.DEFAULT_MLM_EVAL_STEPS,
        help=f"Evaluate and save every N steps (default: {config.DEFAULT_MLM_EVAL_STEPS})",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="Early stopping patience in eval rounds (0 = disabled, default: 0)",
    )
    return parser.parse_args()


# ================================================================
# GPU detection
# ================================================================

def detect_device():
    """Detect available device and print hardware info."""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU detected: {name}")
        print(f"  GPU memory:   {mem:.1f} GB")
        return "cuda", True
    else:
        print("  WARNING: No GPU detected — training will be slow!")
        return "cpu", False


# ================================================================
# MLM logging callback
# ================================================================


class MLMLoggingCallback:
    """Log training progress with ETA, perplexity, and CSV output."""

    def __init__(self, output_dir, total_steps, total_epochs):
        self.output_dir = output_dir
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.start_time = None
        self.last_train_loss = None
        self.csv_path = os.path.join(output_dir, "training_log.csv")

    def _write_csv_header(self):
        with open(self.csv_path, "w") as f:
            f.write("step,epoch,train_loss,eval_loss,perplexity,elapsed_seconds\n")

    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.start_time = time.time()
        os.makedirs(self.output_dir, exist_ok=True)
        self._write_csv_header()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import math
        import time

        if metrics is None or self.start_time is None:
            return

        eval_loss = metrics.get("eval_loss", 0.0)
        perplexity = math.exp(eval_loss) if eval_loss < 100 else float("inf")
        elapsed = time.time() - self.start_time
        elapsed_min = elapsed / 60

        progress = state.global_step / self.total_steps if self.total_steps > 0 else 1
        eta_min = (elapsed / progress - elapsed) / 60 if progress > 0 else 0
        epoch = state.epoch or 0

        train_loss_str = f"{self.last_train_loss:.4f}" if self.last_train_loss is not None else "n/a"

        print(
            f"\n  [Step {state.global_step}/{self.total_steps} | "
            f"Epoch {epoch:.1f}/{self.total_epochs} | "
            f"TrainLoss {train_loss_str} | "
            f"EvalLoss {eval_loss:.4f} | "
            f"PPL {perplexity:.2f} | "
            f"Elapsed {elapsed_min:.0f}m | "
            f"ETA ~{eta_min:.0f}m]\n"
        )

        with open(self.csv_path, "a") as f:
            f.write(
                f"{state.global_step},{epoch:.2f},"
                f"{self.last_train_loss if self.last_train_loss is not None else ''},"
                f"{eval_loss:.6f},{perplexity:.4f},{elapsed:.1f}\n"
            )

    def on_train_end(self, args, state, control, **kwargs):
        import time
        if self.start_time:
            elapsed = (time.time() - self.start_time) / 60
            print(f"\n  Total training time: {elapsed:.1f} minutes")


# Wrap as a TrainerCallback-compatible object
def _make_callback_class():
    from transformers import TrainerCallback

    class _MLMLoggingTrainerCallback(TrainerCallback, MLMLoggingCallback):
        def __init__(self, output_dir, total_steps, total_epochs):
            TrainerCallback.__init__(self)
            MLMLoggingCallback.__init__(self, output_dir, total_steps, total_epochs)

    return _MLMLoggingTrainerCallback


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()

    # Lazy imports — heavy libraries
    import json
    import math
    import random
    import time

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    # Seed everything
    set_seed(args.seed)

    print("=" * 70)
    print("  MLM Domain Adaptation — Old Swedish")
    print("=" * 70)

    # Detect device
    device, use_gpu = detect_device()

    if not use_gpu:
        response = input("\n  Continue on CPU anyway? (y/n): ")
        if response.lower() != "y":
            return
        # Adjust for CPU
        args.batch_size = 2
        args.gradient_accumulation_steps = 8

    eff_batch = args.batch_size * args.gradient_accumulation_steps
    print(f"\n  Base model:        {args.base_model}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Grad accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch:   {eff_batch}")
    print(f"  MLM probability:   {args.mlm_probability}")
    print(f"  Max length:        {args.max_length}")
    print(f"  Learning rate:     {args.learning_rate}")
    print(f"  Warmup ratio:      {args.warmup_ratio}")
    print(f"  LR scheduler:      cosine")
    print(f"  Eval every:        {args.eval_steps} steps")
    print(f"  Seed:              {args.seed}")

    # ----------------------------------------------------------
    # 1. Extract edition texts
    # ----------------------------------------------------------
    print(f"\n[1/4] Extracting edition texts...")
    json_paths = config.MLM_RAW_JSON_FILES
    texts = extract_edition_texts(json_paths, strip_middle_dots=args.strip_middle_dots)
    if args.strip_middle_dots:
        print("  Middle-dot separators: STRIPPED")
    print(f"  Loaded {len(texts)} edition texts from {len(json_paths)} JSON files")

    total_chars = sum(len(t) for t in texts)
    print(f"  Total characters: {total_chars:,}")

    # ----------------------------------------------------------
    # 2. Tokenize
    # ----------------------------------------------------------
    print(f"\n[2/4] Tokenizing with {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            return_overflowing_tokens=True,
            stride=128,
            padding=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # Remove overflow_to_sample_mapping if present (added by
    # return_overflowing_tokens but not needed for training)
    if "overflow_to_sample_mapping" in tokenized.column_names:
        tokenized = tokenized.remove_columns(["overflow_to_sample_mapping"])

    # Filter short sequences (prevents NaN loss)
    original_count = len(tokenized)
    min_len = config.DEFAULT_MLM_MIN_SEQ_LENGTH

    def filter_short(example):
        return len(example["input_ids"]) >= min_len

    tokenized = tokenized.filter(filter_short)
    removed = original_count - len(tokenized)
    if removed > 0:
        print(f"  Removed {removed:,} short sequences < {min_len} tokens "
              f"({100 * removed / original_count:.1f}%)")

    # 90/10 train/eval split for monitoring MLM loss
    split = tokenized.train_test_split(test_size=0.1, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"  Train chunks: {len(train_ds)}")
    print(f"  Eval chunks:  {len(eval_ds)}")

    # ----------------------------------------------------------
    # 3. Model + data collator
    # ----------------------------------------------------------
    print(f"\n[3/4] Loading model for MLM...")
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)
    model.to(device)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # ----------------------------------------------------------
    # 4. Train
    # ----------------------------------------------------------
    steps_per_epoch = len(train_ds) // (args.batch_size * args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"\n[4/4] Training...")
    print(f"  Device:          {'CUDA' if use_gpu else 'CPU'}")
    print(f"  Steps/epoch:     {steps_per_epoch}")
    print(f"  Total steps:     {total_steps}")
    print(f"  Warmup steps:    {warmup_steps}")

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",

        # Evaluation
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_steps,

        # Saving
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # GPU optimisation
        fp16=not args.no_fp16 and use_gpu,
        fp16_full_eval=not args.no_fp16 and use_gpu,
        dataloader_num_workers=4 if use_gpu else 0,
        dataloader_pin_memory=use_gpu,

        # Misc
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        disable_tqdm=False,
    )

    MLMCallback = _make_callback_class()
    callbacks = [MLMCallback(args.output_dir, total_steps, args.epochs)]
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.001,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Capture hardware info before training starts
    hw_info = {}
    if use_gpu:
        hw_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        }

    print(f"\n{'=' * 70}")
    print("  Starting MLM training...")
    print(f"{'=' * 70}\n")

    train_start = time.time()

    try:
        trainer.train()

        training_time_min = (time.time() - train_start) / 60

        # Save final model
        print(f"\n  Saving domain-adapted model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Final evaluation
        print("\n  Final validation metrics:")
        metrics = trainer.evaluate()
        final_eval_loss = metrics["eval_loss"]
        final_perplexity = math.exp(final_eval_loss) if final_eval_loss < 100 else float("inf")
        print(f"  Validation loss: {final_eval_loss:.4f}")
        print(f"  Perplexity:      {final_perplexity:.2f}")

        # Get last train loss from callback
        mlm_cb = callbacks[0]
        final_train_loss = mlm_cb.last_train_loss

        # Best checkpoint
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        if best_ckpt:
            best_ckpt = os.path.basename(best_ckpt)

        # Save training config for reference and reproducibility
        import datetime

        mlm_config = {
            "base_model": args.base_model,
            "training_date": datetime.date.today().isoformat(),
            "corpus_files": [os.path.basename(p) for p in json_paths],
            "num_texts": len(texts),
            "total_characters": total_chars,
            "train_chunks": len(train_ds),
            "eval_chunks": len(eval_ds),
            "short_seqs_removed": removed,
            "strip_middle_dots": args.strip_middle_dots,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": eff_batch,
            "mlm_probability": args.mlm_probability,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "lr_scheduler": "cosine",
            "max_grad_norm": 1.0,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "seed": args.seed,
            "hardware": hw_info,
            "final_eval_loss": final_eval_loss,
            "final_perplexity": round(final_perplexity, 4),
            "final_train_loss": final_train_loss,
            "best_checkpoint": best_ckpt,
            "training_time_minutes": round(training_time_min, 1),
        }
        with open(os.path.join(args.output_dir, "mlm_config.json"), "w") as f:
            json.dump(mlm_config, f, indent=2)

        print(f"\n{'=' * 70}")
        print(f"  MLM pre-training complete.")
        print(f"  Training time:   {training_time_min:.1f} minutes")
        print(f"  Best checkpoint: {best_ckpt or 'final'}")
        print(f"  Model saved to:  {args.output_dir}")
        print(f"  Training log:    {os.path.join(args.output_dir, 'training_log.csv')}")
        print(f"  Use with: python scripts/train_ner.py --model mlm-adapted")
        print(f"{'=' * 70}")

    except KeyboardInterrupt:
        print("\n\n  Training interrupted — saving checkpoint...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"  Checkpoint saved to {args.output_dir}")

    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"\n  GPU out of memory! Try reducing batch size:")
            print(f"    --batch-size 4 --gradient-accumulation-steps 8")
        raise

    finally:
        if use_gpu:
            mem_used = torch.cuda.memory_allocated(0) / 1e9
            print(f"\n  GPU memory used: {mem_used:.2f} GB")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
