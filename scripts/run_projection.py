#!/usr/bin/env python3
"""
Run the three-signal entity projection pipeline on charter data.

Usage:
    .venv/bin/python3 scripts/run_projection.py
    .venv/bin/python3 scripts/run_projection.py --data data/raw/sdhk_1380_1382.json
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PATH, PILOT_DATA_PATH, TORA_PATH, DF_PATH, SMP_PATH, SILVER_DIR, SILVER_V2_DIR, VERIFIED_SILVER_DIR
from src.preprocessing.clean_text import load_charter_data
from src.preprocessing.prepare_gazetteers import load_tora, load_df, load_smp
from src.projection.signal_1_ner import run_ner_on_summaries
from src.pipeline import run_pipeline


def _load_verified_ids(verified_dir):
    """
    Read charter IDs already present in a verified directory.

    Expects filenames of the form sdhk_<ID>_silver.conll.
    Returns a set of ID strings.
    """
    ids = set()
    if not os.path.isdir(verified_dir):
        return ids
    for fname in os.listdir(verified_dir):
        if fname.startswith("sdhk_") and fname.endswith("_silver.conll"):
            parts = fname.split("_")
            if len(parts) >= 2:
                ids.add(parts[1])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Run entity projection pipeline")
    parser.add_argument("--data", default=DATA_PATH, help="Input JSON path")
    parser.add_argument("--output", default=SILVER_V2_DIR, help="Output directory for CoNLL")
    parser.add_argument(
        "--skip-verified",
        default=VERIFIED_SILVER_DIR,
        metavar="DIR",
        help="Skip charters whose CoNLL files already exist in this directory "
             "(default: data/verified_silver/)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Three-Signal Entity Projection Pipeline")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    charters = load_charter_data(args.data)
    print(f"  Loaded {len(charters)} charters from {args.data}")

    # Filter out already-verified charters
    if args.skip_verified:
        verified_ids = _load_verified_ids(args.skip_verified)
        if verified_ids:
            before = len(charters)
            charters = [c for c in charters if str(c["Id"]) not in verified_ids]
            skipped = before - len(charters)
            print(f"  Skipping {skipped} already-verified charters "
                  f"(from {args.skip_verified})")
            print(f"  Processing {len(charters)} remaining charters")

    # Load gazetteers
    print("\n[2/4] Loading gazetteers...")
    tora_exact, tora_fuzzy, tora_sources, tora_stats = load_tora(TORA_PATH)
    df_exact, df_sources, df_stats = load_df(DF_PATH)
    smp_exact, smp_fuzzy, smp_sources, smp_stats = load_smp(SMP_PATH)

    # Merge DF (Diplomatarium Fennicum) into TORA sets — both are place names
    df_new = df_exact - tora_exact
    tora_exact |= df_exact
    tora_sources.update({k: v for k, v in df_sources.items() if k not in tora_sources})
    tora_fuzzy = sorted(tora_exact)

    print(f"  TORA: {tora_stats['unique_lookup_names']} lookup names")
    print(f"  DF:   {df_stats['unique_names']} lookup names ({len(df_new)} new)")
    print(f"  TORA+DF combined: {len(tora_exact)} lookup names")
    print(f"  SMP:  {smp_stats['unique_names']} lookup names")

    # Run NER
    print("\n[3/4] Running NER on modern summaries...")
    ner_results = run_ner_on_summaries(charters)

    # Run pipeline
    print("\n[4/4] Processing charters through pipeline...")
    conll_dir = os.path.join(args.output, "conll")
    results = run_pipeline(
        charters,
        tora_data=(tora_exact, tora_fuzzy, tora_sources),
        smp_data=(smp_exact, smp_fuzzy, smp_sources),
        ner_results=ner_results,
        output_dir=conll_dir,
    )

    # Save results JSON
    results_path = os.path.join(args.output, "pipeline_results.json")
    os.makedirs(args.output, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate review report (cap-only unknowns per charter)
    report_path = os.path.join(args.output, "review_report.txt")
    _write_review_report(results, report_path)

    print(f"\n  CoNLL files:    {conll_dir}/")
    print(f"  Results JSON:   {results_path}")
    print(f"  Review report:  {report_path}")
    print(f"\n{'=' * 70}")
    print("  Pipeline complete.")


def _write_review_report(results, path):
    """
    Write a human-readable report of cap-only unknown tokens.

    These are capitalised tokens that only Signal 3 flagged, with no type
    determined — labelled O but likely entities the pipeline missed.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  Review Report: Capitalisation-Only Unknown Tokens")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Tokens below were capitalised (Signal 3) but had no type from")
    lines.append("any signal, so they were labelled O. Many are likely entities")
    lines.append("that need manual correction.")
    lines.append("")

    total = 0
    for cid in sorted(results.keys(), key=int):
        r = results[cid]
        vote_log = r.get("voting_log", {})
        details = vote_log.get("cap_only_unknown_details", [])
        if not details:
            continue

        total += len(details)
        lines.append(f"--- SDHK {cid} ({len(details)} tokens) ---")

        # Deduplicate while preserving order, show count if repeated
        seen = {}
        for d in details:
            tok = d["token_text"]
            seen[tok] = seen.get(tok, 0) + 1

        for tok, count in seen.items():
            suffix = f"  (x{count})" if count > 1 else ""
            lines.append(f"  {tok}{suffix}")

        lines.append("")

    # Summary
    n_charters_affected = sum(
        1 for r in results.values()
        if r.get("voting_log", {}).get("cap_only_unknown_details")
    )
    lines.append("-" * 70)
    lines.append(f"Total: {total} cap-only unknown tokens across "
                 f"{n_charters_affected}/{len(results)} charters")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Review report: {total} cap-only unknowns across "
          f"{n_charters_affected}/{len(results)} charters")


if __name__ == "__main__":
    main()
