"""
Convert the expert-annotated test set to CoNLL 2002 BIO format.

Input : expert_annot/test_set/annotator_{1,5,6,7}_test.txt
Output: expert_annot/test_set_conll/sdhk_<id>_gold.conll  (one file per charter)

Formatting mirrors data/1380_1382_dataset/*_silver.conll (the training set
format, produced by src.projection.voting.labels_to_conll):
  · one "token LABEL" line per token (single space delimiter);
  · BIO tags: B-Person / I-Person / B-Location / I-Location / O;
  · all tokens emitted, including punctuation tagged O;
  · a blank line is inserted after every sentence-ending token (. ! ?)
    to mark sentence boundaries — matches the training data exactly;
  · file is written as "\n".join(lines) + "\n", so if the charter ends
    on a sentence boundary the file ends with "\n\n" (also matches).

Pipeline per charter:
  1. split SIDBRYTNING sections, extract body after the "Datum:" line;
  2. strip <persname>/<placename> tags → parallel char-level IO labels;
  3. apply the project's clean_edition_text transforms (º/° → space, etc.);
  4. tokenise with src.preprocessing.clean_text.tokenize_edition;
  5. label each token with the IO tag of its first char;
  6. convert IO → BIO (B- on entity start, I- on continuation of same type);
  7. write one file per charter.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.compute_iaa import (
    FILES,
    iter_charters,
    extract_body,
    strip_and_label,
    clean_parallel,
    tokens_with_labels,
)

SRC_DIR = os.path.join(PROJECT_ROOT, "data", "expert_annotations", "test_set")
DST_DIR = os.path.join(PROJECT_ROOT, "data", "expert_gold_test")

TEST_FILES = {
    "A1": "annotator_1_test.txt",
    "A5": "annotator_5_test.txt",
    "A6": "annotator_6_test.txt",
    "A7": "annotator_7_test.txt",
}


SENTENCE_END = {".", "!", "?"}


def labeled_to_conll_lines(labeled):
    """Convert IO-labeled tokens (list of {'text','label'}) into the list
    of CoNLL lines matching src.projection.voting.labels_to_conll:
      · emit every token (punctuation included) as "token BIO_LABEL";
      · insert a blank line after each sentence-ending punct (. ! ?)."""
    lines = []
    prev = "O"
    for tok in labeled:
        lab = tok["label"]
        if lab == "O":
            bio = "O"
        elif lab != prev:
            bio = f"B-{lab}"
        else:
            bio = f"I-{lab}"
        lines.append(f"{tok['text']} {bio}")
        prev = lab
        if tok["text"] in SENTENCE_END:
            lines.append("")
    return lines


def write_conll(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    written = []
    for a, fname in TEST_FILES.items():
        src_path = os.path.join(SRC_DIR, fname)
        for cid, sec in iter_charters(src_path):
            body = extract_body(sec)
            stripped, char_labels = strip_and_label(body)
            cleaned, cleaned_labels = clean_parallel(stripped, char_labels)
            labeled = tokens_with_labels(cleaned, cleaned_labels)
            lines = labeled_to_conll_lines(labeled)

            out_path = os.path.join(DST_DIR, f"sdhk_{cid}_gold.conll")
            write_conll(out_path, lines)

            n_tok = sum(1 for l in lines if l)
            n_ent = sum(1 for l in lines if l.endswith(" B-Person")
                                        or l.endswith(" B-Location"))
            written.append((cid, a, n_tok, n_ent))

    written.sort(key=lambda r: int(r[0]))
    print(f"Wrote {len(written)} CoNLL files to: {DST_DIR}")
    print()
    print(f"  {'charter':<12} {'src':<4} {'tokens':>7} {'entities':>9}")
    print(f"  {'-'*12} {'-'*4} {'-'*7} {'-'*9}")
    for cid, a, n_tok, n_ent in written:
        print(f"  SDHK {cid:<7} {a:<4} {n_tok:>7} {n_ent:>9}")
    print()
    print(f"  TOTAL: {len(written)} charters, "
          f"{sum(r[2] for r in written)} tokens, "
          f"{sum(r[3] for r in written)} entities")


if __name__ == "__main__":
    main()
