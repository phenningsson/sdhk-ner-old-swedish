"""
Inter-annotator agreement on the 10 shared charters (v2).

Two metrics:
  1. Token-level Krippendorff's α (nominal) across all 4 coders. Labels are
     IO (O / Person / Location), which captures type agreement. α does not
     measure boundary disagreements; that question is handled by (2).
  2. Pairwise entity-level F1 over all 6 coder pairs. An entity is a
     (charter, token-span, type) triple; a match requires EXACT span + type.

Tokenisation mirrors src.preprocessing.clean_text so that the IAA token
scheme lines up with the later CoNLL 2002 BIO conversion.

v2: α is now computed via NLTK's AnnotationTask (Artstein & Poesio 2008
test cases) rather than the custom implementation in compute_iaa.py.
This script is to be used for the IAA calculation.
"""

import os
import re
import sys
from collections import Counter
from itertools import combinations

from nltk.metrics.agreement import AnnotationTask

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.clean_text import tokenize_edition

DIR = os.path.join(PROJECT_ROOT, "data", "expert_annotations", "pre_adjudication")
FILES = {
    "A1": "annotator_1_annotations.txt",
    "A5": "annotator_5_annotations.txt",
    "A6": "annotator_6_annotations.txt",
    "A7": "annotator_7_annotations.txt",
}
SIDBRYTNING   = "--- SIDBRYTNING ---"
HEADER_RE     = re.compile(r"SDHK\s+(\d+)", re.IGNORECASE)
OPEN_TAG_RE   = re.compile(r"<(persname|placename)>")
CLOSE_TAG_RE  = re.compile(r"</(persname|placename)>")
TYPE_MAP      = {"persname": "Person", "placename": "Location"}


# ------------------------------------------------------------------ parsing
def iter_charters(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for sec in text.split(SIDBRYTNING):
        m = HEADER_RE.search(sec)
        if m:
            yield m.group(1), sec


def extract_body(sec):
    """Body = everything after the 'Datum:' line."""
    lines = sec.split("\n")
    start = 0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("datum:"):
            start = i + 1
            break
    return "\n".join(lines[start:])


def strip_and_label(body):
    """Walk the tagged body, stripping <persname>/<placename> tags and
    emitting a parallel char-level label array (O / Person / Location)."""
    out_chars, out_labels = [], []
    current = "O"
    i = 0
    while i < len(body):
        m = OPEN_TAG_RE.match(body, i)
        if m:
            current = TYPE_MAP[m.group(1)]
            i = m.end()
            continue
        m = CLOSE_TAG_RE.match(body, i)
        if m:
            current = "O"
            i = m.end()
            continue
        out_chars.append(body[i])
        out_labels.append(current)
        i += 1
    return "".join(out_chars), out_labels


def clean_parallel(text, labels):
    """Mirror src.preprocessing.clean_text.clean_edition_text on (text, labels):
    º/° → space (label reset to O), collapse multi-space runs, then strip."""
    t1, l1 = [], []
    for c, lab in zip(text, labels):
        if c in ("\u00ba", "\u00b0"):
            t1.append(" "); l1.append("O")
        else:
            t1.append(c); l1.append(lab)
    t2, l2 = [], []
    prev_space = False
    for c, lab in zip(t1, l1):
        if c == " ":
            if prev_space:
                continue
            prev_space = True
        else:
            prev_space = False
        t2.append(c); l2.append(lab)
    while t2 and t2[0].isspace():
        t2.pop(0); l2.pop(0)
    while t2 and t2[-1].isspace():
        t2.pop(); l2.pop()
    return "".join(t2), l2


def tokens_with_labels(cleaned, char_labels):
    """Tokenise with the project's tokeniser; assign each token the label
    of its first character. Tokens never straddle a label boundary
    because the source contains no mid-word tags."""
    out = []
    for t in tokenize_edition(cleaned):
        lab = char_labels[t["start"]] if t["start"] < len(char_labels) else "O"
        out.append({"text": t["text"], "label": lab,
                    "start": t["start"], "end": t["end"]})
    return out


def extract_entities(labeled):
    """Convert IO token stream → list of (token_start, token_end_excl, type)
    using maximal runs of the same non-O label."""
    ents, i, n = [], 0, len(labeled)
    while i < n:
        lab = labeled[i]["label"]
        if lab == "O":
            i += 1
            continue
        j = i
        while j < n and labeled[j]["label"] == lab:
            j += 1
        ents.append((i, j, lab))
        i = j
    return ents


# ------------------------------------------------------------------ main
def main():
    per_annot = {a: {} for a in FILES}
    for a, fname in FILES.items():
        for cid, sec in iter_charters(os.path.join(DIR, fname)):
            body = extract_body(sec)
            stripped, char_labels = strip_and_label(body)
            cleaned, cleaned_labels = clean_parallel(stripped, char_labels)
            per_annot[a][cid] = tokens_with_labels(cleaned, cleaned_labels)

    shared = sorted(
        set.intersection(*[set(d.keys()) for d in per_annot.values()]),
        key=int,
    )
    print("=" * 72)
    print(f"Shared charters: {len(shared)}  →  {shared}")
    print("=" * 72)

    # Sanity: token sequences must match across annotators (raw text is
    # byte-identical, so tokens will be too — check anyway).
    for cid in shared:
        tok_seqs = {a: tuple(t["text"] for t in per_annot[a][cid]) for a in FILES}
        if len(set(tok_seqs.values())) != 1:
            print(f"  ⚠ token sequences differ for SDHK {cid} — aborting.")
            for a, seq in tok_seqs.items():
                print(f"    {a}: len={len(seq)}")
            return

    # ---- 1. Token-level Krippendorff's α ----
    # Build (coder_id, item_id, label) triples for NLTK.
    triples = []
    for cid in shared:
        n_tokens = len(per_annot["A1"][cid])
        for i in range(n_tokens):
            item_id = f"{cid}:{i}"
            for a in FILES:
                triples.append((a, item_id, per_annot[a][cid][i]["label"]))

    alpha = AnnotationTask(data=triples).alpha()
    n_items = len({t[1] for t in triples})

    print("\nToken-level Krippendorff's α (nominal, IO labels)")
    print("-" * 72)
    print(f"  α = {alpha:.4f}")
    print(f"  units: {n_items} tokens × {len(FILES)} coders "
          f"over {len(shared)} charters")

    # Per-annotator label distribution
    print("\n  Per-annotator label counts on shared charters:")
    for a in FILES:
        c = Counter(t["label"] for cid in shared for t in per_annot[a][cid])
        total = sum(c.values())
        print(f"    {a}: O={c['O']:>4}  Person={c['Person']:>3}  "
              f"Location={c['Location']:>3}  (total={total})")

    # ---- 2. Pairwise entity-level F1 ----
    ents_per_a = {
        a: {(cid, s, e, t)
            for cid in shared
            for (s, e, t) in extract_entities(per_annot[a][cid])}
        for a in FILES
    }

    print("\nEntity counts per annotator on shared charters:")
    for a in FILES:
        c = Counter(t for (_, _, _, t) in ents_per_a[a])
        print(f"  {a}: {len(ents_per_a[a]):>3} total  "
              f"(Person={c['Person']}, Location={c['Location']})")

    print("\nPairwise entity-level F1 (exact span + type match)")
    print("-" * 72)
    hdr = f"  {'Pair':<8} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} " \
          f"{'|A|':>5} {'|B|':>5}"
    print(hdr)
    f1s = []
    for a, b in combinations(FILES, 2):
        A, B = ents_per_a[a], ents_per_a[b]
        tp = len(A & B)
        p = tp / len(B) if B else 0.0
        r = tp / len(A) if A else 0.0
        f1 = 2 * tp / (len(A) + len(B)) if (A or B) else 1.0
        f1s.append(f1)
        print(f"  {a}↔{b:<4} {p:>7.3f} {r:>7.3f} {f1:>7.3f} "
              f"{tp:>5} {len(A):>5} {len(B):>5}")
    print(f"\n  Mean pairwise F1: {sum(f1s) / len(f1s):.4f}")

    # Per-type pairwise F1 (break out Person vs Location)
    print("\nPairwise F1 broken out by type:")
    for etype in ("Person", "Location"):
        sub = {a: {e for e in ents_per_a[a] if e[3] == etype} for a in FILES}
        print(f"  [{etype}]")
        row_f1 = []
        for a, b in combinations(FILES, 2):
            A, B = sub[a], sub[b]
            tp = len(A & B)
            f1 = 2 * tp / (len(A) + len(B)) if (A or B) else 1.0
            row_f1.append(f1)
            print(f"    {a}↔{b:<4} F1={f1:.3f}  TP={tp}  |A|={len(A)}  |B|={len(B)}")
        print(f"    mean F1: {sum(row_f1) / len(row_f1):.4f}")


if __name__ == "__main__":
    main()
