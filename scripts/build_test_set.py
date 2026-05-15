"""
Assemble the final expert-annotated test set from normalisedv2.

For each annotator, keep:
  · every charter UNIQUE to that annotator (i.e. not among the 10 shared);
  · every SHARED charter whose adjudication assigns it to that annotator.

All other annotators' copies of a shared charter are discarded for the
test set, so each SDHK ID appears in exactly one output file.

Output layout:
    expert_annot/test_set/
        annotator_1_test.txt
        annotator_5_test.txt
        annotator_6_test.txt
        annotator_7_test.txt
        manifest.txt

The per-annotator files are byte-for-byte passthroughs of the chosen
charters from normalisedv2, including headers (SDHK <id>, annot. N /
annoterare N) and --- SIDBRYTNING --- separators. The manifest lists
every charter in the test set with its source annotator.
"""

import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_DIR  = os.path.join(PROJECT_ROOT, "data", "expert_annotations", "normalised")
DST_DIR  = os.path.join(PROJECT_ROOT, "data", "expert_annotations", "test_set")

FILES = {
    "A1": "annotator_1_annotations.txt",
    "A5": "annotator_5_annotations.txt",
    "A6": "annotator_6_annotations.txt",
    "A7": "annotator_7_annotations.txt",
}
OUT_NAMES = {
    "A1": "annotator_1_test.txt",
    "A5": "annotator_5_test.txt",
    "A6": "annotator_6_test.txt",
    "A7": "annotator_7_test.txt",
}

# Adjudication: {cid: annotator key}
ADJUDICATION = {
    "10643": "A6",
    "10710": "A5",
    "10789": "A5",
    "11180": "A1",
    "11459": "A5",
    "11998": "A5",
    "12106": "A5",
    "12113": "A6",
    "12157": "A6",
    "41463": "A5",
}

SIDBRYTNING = "--- SIDBRYTNING ---"
HEADER_RE   = re.compile(r"SDHK\s+(\d+)", re.IGNORECASE)


def parse_file(path):
    """Yield (charter_id, raw_section) for every charter in `path`.
    `raw_section` preserves the text between SIDBRYTNING markers exactly
    (leading/trailing whitespace included) so we can re-emit it verbatim."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for sec in text.split(SIDBRYTNING):
        m = HEADER_RE.search(sec)
        if m:
            yield m.group(1), sec


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    # {annotator: {cid: raw_section}}
    src = {a: {} for a in FILES}
    for a, fname in FILES.items():
        for cid, sec in parse_file(os.path.join(SRC_DIR, fname)):
            src[a][cid] = sec

    # Identify the shared set (should match ADJUDICATION keys).
    all_cids = {a: set(src[a]) for a in FILES}
    shared = set.intersection(*all_cids.values())
    assert shared == set(ADJUDICATION), (
        f"Adjudication keys don't match the shared set.\n"
        f"  shared:       {sorted(shared, key=int)}\n"
        f"  adjudication: {sorted(ADJUDICATION, key=int)}"
    )

    # Decide which charters go into which output file.
    # For each annotator: unique charters + charters adjudicated to them.
    keep = {a: [] for a in FILES}
    for a in FILES:
        unique = all_cids[a] - shared
        assigned_shared = {c for c, who in ADJUDICATION.items() if who == a}
        keep[a] = sorted(unique | assigned_shared, key=int)

    # Write per-annotator files, preserving the SIDBRYTNING separator
    # between sections. We rejoin with "--- SIDBRYTNING ---" so the
    # output format is identical to the input.
    totals = {}
    for a in FILES:
        sections = [src[a][cid] for cid in keep[a]]
        out_text = SIDBRYTNING.join(sections)
        out_path = os.path.join(DST_DIR, OUT_NAMES[a])
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_text)
        totals[a] = len(keep[a])

    # Sanity: every charter appears in exactly one output.
    all_written = []
    for a in FILES:
        all_written.extend(keep[a])
    assert len(all_written) == len(set(all_written)), \
        "duplicate charter(s) across output files!"

    # --- manifest ----------------------------------------------------------
    lines = []
    lines.append("EXPERT-ANNOTATED TEST SET — MANIFEST")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Source: expert_annot/normalisedv2/ (normalised + verified)")
    lines.append("Assembly rule:")
    lines.append("  · Unique charters go to their sole annotator's file.")
    lines.append("  · Shared charters go to the adjudicated annotator's file.")
    lines.append("")
    lines.append("Per-annotator totals:")
    for a in FILES:
        unique = len(all_cids[a] - shared)
        adjud  = sum(1 for c in ADJUDICATION.values() if c == a)
        total  = totals[a]
        lines.append(f"  {OUT_NAMES[a]:<26}  "
                     f"unique={unique:>3}  "
                     f"adjudicated_shared={adjud:>2}  "
                     f"total={total:>3}")
    grand = sum(totals.values())
    lines.append(f"  {'TOTAL':<26}  "
                 f"{'':>10}{' ':>3}  "
                 f"{'':>18}{' ':>2}  total={grand:>3}")
    lines.append("")

    lines.append("Adjudication (10 shared charters):")
    for cid in sorted(ADJUDICATION, key=int):
        lines.append(f"  SDHK {cid:<8}  → {ADJUDICATION[cid]}  "
                     f"({OUT_NAMES[ADJUDICATION[cid]]})")
    lines.append("")

    lines.append("Full charter inventory (sorted by SDHK id):")
    lines.append("-" * 72)
    # Build reverse index: cid → annotator
    cid_to_a = {}
    for a in FILES:
        for cid in keep[a]:
            cid_to_a[cid] = a
    for cid in sorted(cid_to_a, key=int):
        a = cid_to_a[cid]
        status = "ADJUDICATED" if cid in shared else "unique"
        lines.append(f"  SDHK {cid:<8}  {a}  ({OUT_NAMES[a]})  [{status}]")
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"Total charters in test set: {grand}")
    lines.append("")

    with open(os.path.join(DST_DIR, "manifest.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --- console summary --------------------------------------------------
    print(f"Wrote test set to: {DST_DIR}")
    for a in FILES:
        unique = len(all_cids[a] - shared)
        adjud  = sum(1 for c in ADJUDICATION.values() if c == a)
        print(f"  {OUT_NAMES[a]:<26}  "
              f"unique={unique:>3}  "
              f"adjud={adjud:>2}  "
              f"total={totals[a]:>3}")
    print(f"  {'TOTAL':<26}  {grand} charters")
    print(f"  manifest.txt also written.")


if __name__ == "__main__":
    main()
