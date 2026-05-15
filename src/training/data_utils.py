"""
Data utilities for NER training and MLM pre-training.

Handles:
- Reading CoNLL files (space-separated BIO format)
- Splitting a directory of CoNLL files into train/val/test (80/10/10)
- Extracting edition texts from raw JSON for MLM pre-training
"""

import json
import os
import random
from typing import Dict, List, Tuple

# Editorial separator in SDHK editions — not a real word token
_MIDDLE_DOT = "\u2027"


# ------------------------------------------------------------------
# CoNLL reading
# ------------------------------------------------------------------

def read_conll_file(filepath: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read a CoNLL file and return parallel lists of tokens and labels.

    Handles both space-separated (SDHK pipeline output) and
    tab-separated (Icelandic convention) formats.

    Returns:
        (all_tokens, all_labels) where each element is a list of
        sentences, and each sentence is a list of strings.
    """
    all_tokens: List[List[str]] = []
    all_labels: List[List[str]] = []
    current_tokens: List[str] = []
    current_labels: List[str] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if current_tokens:
                    all_tokens.append(current_tokens)
                    all_labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                # Try tab first (Icelandic format), then last whitespace
                # split (SDHK format where tokens may contain spaces is
                # unlikely, but rsplit is safest).
                if "\t" in line:
                    parts = line.split("\t", maxsplit=1)
                else:
                    parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    token, label = parts[0], parts[1]
                    # Skip standalone middle-dot tokens (editorial markers,
                    # not in BERT vocabulary → [UNK])
                    if token == _MIDDLE_DOT:
                        continue
                    current_tokens.append(token)
                    current_labels.append(label)

    # Flush last sentence if file does not end with blank line
    if current_tokens:
        all_tokens.append(current_tokens)
        all_labels.append(current_labels)

    return all_tokens, all_labels


def read_conll_directory(
    directory: str,
) -> Dict[str, Tuple[List[List[str]], List[List[str]]]]:
    """
    Read all .conll files in *directory*.

    Returns:
        dict mapping filename (without extension) to (tokens, labels).
    """
    result = {}
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".conll"):
            path = os.path.join(directory, fname)
            tokens, labels = read_conll_file(path)
            key = fname.rsplit(".", 1)[0]
            result[key] = (tokens, labels)
    return result


# ------------------------------------------------------------------
# Train / val / test splitting
# ------------------------------------------------------------------

def split_files(
    directory: str,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Shuffle and split .conll filenames from *directory* into
    train / val / test lists at the given ratios.

    The split is file-level (each charter goes entirely into one
    partition) to avoid data leakage.

    Returns:
        (train_files, val_files, test_files) — lists of full paths.
    """
    all_files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".conll")
    )
    rng = random.Random(seed)
    rng.shuffle(all_files)

    n = len(all_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = all_files[:n_train]
    val_files = all_files[n_train : n_train + n_val]
    test_files = all_files[n_train + n_val :]

    return train_files, val_files, test_files


def merge_files(
    file_list: List[str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read and concatenate sentences from multiple CoNLL files.

    Returns:
        (all_tokens, all_labels) — merged across all files.
    """
    all_tokens: List[List[str]] = []
    all_labels: List[List[str]] = []

    for path in file_list:
        tokens, labels = read_conll_file(path)
        all_tokens.extend(tokens)
        all_labels.extend(labels)

    return all_tokens, all_labels


# ------------------------------------------------------------------
# Edition text extraction for MLM
# ------------------------------------------------------------------

def extract_edition_texts(
    json_paths: List[str], strip_middle_dots: bool = False,
) -> List[str]:
    """
    Load charter JSON files and return all non-empty Edition texts.

    Each charter's Edition field is one string; returned as-is
    (no tokenization — the MLM tokenizer handles that).

    If *strip_middle_dots* is True, editorial middle-dot separators
    (U+2027) are removed from the text before returning.
    """
    texts: List[str] = []

    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            charters = json.load(f)
        for charter in charters:
            edition = charter.get("Edition", "").strip()
            if edition:
                if strip_middle_dots:
                    edition = edition.replace(_MIDDLE_DOT, "")
                texts.append(edition)

    return texts
