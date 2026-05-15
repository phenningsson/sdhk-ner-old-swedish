#!/usr/bin/env python3
"""
Evaluate the three-signal entity projection pipeline on a directory of
BIO-labelled CoNLL test files, using the same metrics protocol as
scripts/evaluate_ner_v2.py (entity-level seqeval + token-level sklearn
+ FP/FN error diagnostic) so pipeline and NER numbers are directly
comparable in the thesis.

Supports two test sets, selected via --edition-source:

  annotator  (default) — expert gold, 75 charters.
      Charter Edition text is reconstructed from
      expert_annot/test_set/annotator_*_test.txt via the same
      extract_body/strip_and_label/clean_parallel steps that built
      expert_annot/final_test_set_conll/. Required because the raw
      JSON Edition text diverges from the annotator source for 13 of
      the 75 charters (PUA chars, hand-corrected transcriptions, a
      stray SDHK 10955 header), and per-charter token alignment is
      enforced strictly. Modern Summary comes from the raw JSON.

  silver-conll — internal test set, 43 charters.
      The 43 files held out by the 80/10/10 split in train_ner_v2.py
      (reproducible with scripts/extract_internal_test_set.py). The
      silver CoNLL in data/1380_1382_dataset/ was hand-verified after
      the pipeline generated it, and the fixes preserve characters the
      current pipeline strips (e.g. º/° Roman-numeral markers, merged
      multi-punctuation like '...'). Reading Edition straight from the
      raw JSON therefore drifts from silver tokens on a handful of
      charters. Instead this mode reconstructs Edition text by
      space-joining the silver tokens themselves, then applies the
      same normalisation (strip º/°, split multi-punct) to *both*
      sides so alignment is exact. Modern Summary still comes from the
      raw JSON (Signal 1 input).

Both modes share the same downstream metrics / error diagnostic / JSON
dump via src.evaluation.report.compute_and_report.

Usage:
    # Expert gold (75 charters)
    .venv/bin/python3 scripts/evaluate_pipeline.py

    # Internal test set (43 charters from the 80/10/10 split)
    .venv/bin/python3 scripts/evaluate_pipeline.py \\
        --test-dir data/internal_test_set \\
        --edition-source silver-conll \\
        --output-json eval_results_pipeline_internal.json
"""

import argparse
import os
import platform
import re
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR  = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)

import config
from src.evaluation.report import compute_and_report
from src.preprocessing.clean_text import load_charter_data
from src.preprocessing.prepare_gazetteers import load_tora, load_df, load_smp
from src.projection.signal_1_ner import run_ner_on_summaries
from src.pipeline import run_pipeline
from src.training.data_utils import read_conll_file
# Reuse the exact gold-text extraction used to build the gold CoNLL files.
from scripts.compute_iaa import (
    iter_charters,
    extract_body,
    strip_and_label,
    clean_parallel,
)


DIR = os.path.join(PROJECT_ROOT, "data", "expert_annotations", "normalised")

DEFAULT_TEST_DIR = os.path.join(
    PROJECT_ROOT, "data", "expert_gold_test"
)
ANNOTATOR_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "expert_annotations", "test_set")
ANNOTATOR_FILES = [
    "annotator_1_test.txt",
    "annotator_5_test.txt",
    "annotator_6_test.txt",
    "annotator_7_test.txt",
]


# ================================================================
# Helpers
# ================================================================


def load_gold_edition_texts():
    """
    Return {charter_id: edition_text} by walking every annotator test
    file and applying the same body-extraction + tag-stripping the gold
    CoNLL builder used. Extra whitespace/º/° are cleaned here so the
    text is ready for tokenize_edition (which is what the pipeline
    calls internally).
    """
    out = {}
    for fname in ANNOTATOR_FILES:
        path = os.path.join(ANNOTATOR_TEST_DIR, fname)
        for cid, sec in iter_charters(path):
            body = extract_body(sec)
            stripped, char_labels = strip_and_label(body)
            cleaned, _ = clean_parallel(stripped, char_labels)
            out[cid] = cleaned
    return out


def reconstruct_edition_from_silver(silver_path):
    """
    Rebuild an Edition text from a silver CoNLL by space-joining its
    tokens. Feeding the result through the pipeline's own cleaner +
    tokeniser should (after the shared normalisation below) yield
    exactly the silver tokens — see normalise_silver_for_pipeline.
    """
    toks_per_sent, _ = read_conll_file(silver_path)
    return " ".join(t for s in toks_per_sent for t in s)


# tokenize_edition's inner split: word chars + combining diacritics, or
# any single non-word non-space character. Applied to a silver token, it
# produces the same sub-token sequence the pipeline would.
_TOKENIZE_SUB_RE = re.compile(r"[\w\u0300-\u036f]+|[^\w\s]")


def normalise_silver_for_pipeline(tokens, labels):
    """
    Apply the same transformations to silver that the pipeline applies
    to its input, so per-token alignment is exact:
      · replace º (U+00BA) and ° (U+00B0) with spaces inside each
          token — clean_edition_text uses spaces specifically so that
          embedded markers (e.g. "Mºcccºlxxxo") act as word boundaries,
          splitting the silver token into pieces (M / ccc / lxxxo);
      · split any silver token that contains multiple non-word chars
          (e.g. "...") into single-char tokens — tokenize_edition
          produces only single-punct tokens.
    Labels for split pieces are copied from the source token (safe: any
    punctuation-only silver token is labelled 'O' in practice; Roman-
    numeral spans are O too).
    """
    out_toks, out_labs = [], []
    for tok, lab in zip(tokens, labels):
        cleaned = tok.replace("\u00ba", " ").replace("\u00b0", " ")
        subs = [m.group() for m in _TOKENIZE_SUB_RE.finditer(cleaned)]
        if not subs:
            continue
        for sub in subs:
            out_toks.append(sub)
            out_labs.append(lab)
    return out_toks, out_labs


def build_charter_inputs(gold_ids, raw_json_path, edition_source, cid_to_path):
    """
    Construct the list of charter dicts the pipeline expects:
        {Id, Summary, Edition}

    Summary always comes from raw JSON (Signal 1 needs it).
    Edition depends on *edition_source*:
        "annotator"    — normalised annotator-source text
                         (expert_annot/test_set/annotator_*_test.txt).
        "silver-conll" — reconstructed by space-joining the silver
                         CoNLL tokens for that charter.
    """
    if edition_source not in ("annotator", "silver-conll"):
        raise ValueError(f"Unknown edition_source: {edition_source!r}")

    raw = load_charter_data(raw_json_path)
    raw_by_id = {str(c["Id"]): c for c in raw}

    gold_texts = load_gold_edition_texts() if edition_source == "annotator" else None

    missing_raw, missing_edition = [], []
    charters = []
    for cid in gold_ids:
        if cid not in raw_by_id:
            missing_raw.append(cid);  continue
        base = raw_by_id[cid]
        if edition_source == "annotator":
            if cid not in gold_texts:
                missing_edition.append(cid); continue
            edition = gold_texts[cid]
        else:
            edition = reconstruct_edition_from_silver(cid_to_path[cid])
            if not edition.strip():
                missing_edition.append(cid); continue
        charters.append({
            "Id":      cid,
            "Summary": base.get("Summary", ""),
            "Edition": edition,
        })
    return charters, missing_raw, missing_edition


def flatten_conll(path):
    """Read a CoNLL file, flatten all sentences into one (tokens, labels) pair."""
    toks_per_sent, lbls_per_sent = read_conll_file(path)
    tokens = [t for s in toks_per_sent for t in s]
    labels = [l for s in lbls_per_sent for l in s]
    return tokens, labels


def discover_gold_files(test_dir):
    """
    Return (gold_files, cid_to_path). Accepts any ``sdhk_<id>_*.conll``
    filename — charter id is the second underscore-separated segment.
    """
    gold_files = sorted(
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.startswith("sdhk_") and f.endswith(".conll")
    )
    cid_to_path = {}
    for path in gold_files:
        parts = os.path.basename(path).split("_")
        if len(parts) < 2:
            continue
        cid_to_path[parts[1]] = path
    return gold_files, cid_to_path


# ================================================================
# Main
# ================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the entity projection pipeline on the "
                    "75-charter expert gold test set using the same "
                    "metrics protocol as evaluate_ner_v2.py."
    )
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR,
                        help="Directory of gold CoNLL files (one per charter).")
    parser.add_argument("--edition-source",
                        choices=["annotator", "silver-conll"], default="annotator",
                        help="Where to pull the Old Swedish Edition text from. "
                             "'annotator' (default) for the expert gold set; "
                             "'silver-conll' for the internal test set — "
                             "reconstructs Edition by space-joining the silver "
                             "tokens, then normalises both sides for exact "
                             "per-charter alignment.")
    parser.add_argument("--raw-json", default=os.path.join(
        PROJECT_ROOT, "data", "raw", "sdhk_all_swedish_scraped.json"),
        help="Charter JSON providing modern Summary text (Signal 1 input).")
    parser.add_argument("--tora-path", default=config.TORA_PATH)
    parser.add_argument("--df-path",   default=config.DF_PATH)
    parser.add_argument("--smp-path",  default=config.SMP_PATH)
    parser.add_argument("--output-json", default=os.path.join(
        PROJECT_ROOT, "eval_results_pipeline.json"))
    args = parser.parse_args()

    print("=" * 72)
    print("  Pipeline Evaluation (three-signal projection)")
    print("=" * 72)
    print(f"  Test dir:        {args.test_dir}")
    print(f"  Edition source:  {args.edition_source}")
    print(f"  Raw JSON:        {args.raw_json}")
    print(f"  TORA:            {args.tora_path}")
    print(f"  DF:              {args.df_path}")
    print(f"  SMP:             {args.smp_path}")

    # ── Reproducibility fingerprint ──
    import seqeval, sklearn
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    def _v(pkg):
        try:
            return _pkg_version(pkg)
        except PackageNotFoundError:
            return "unknown"
    versions = {
        "python":       platform.python_version(),
        "numpy":        np.__version__,
        "seqeval":      _v("seqeval"),
        "scikit-learn": _v("scikit-learn"),
        "rapidfuzz":    _v("rapidfuzz"),
        "transformers": _v("transformers"),
        "platform":     platform.platform(),
    }
    print()
    print("  Reproducibility fingerprint:")
    for k, val in versions.items():
        print(f"    {k:<13} {val}")
    print()

    # ── Discover test charters ──
    gold_files, cid_to_path = discover_gold_files(args.test_dir)
    if not gold_files:
        print(f"  No sdhk_*.conll files found in {args.test_dir}")
        return
    gold_ids = sorted(cid_to_path.keys(), key=lambda c: int(c) if c.isdigit() else c)
    print(f"  Gold charters: {len(gold_ids)}")

    # ── Build pipeline inputs ──
    charters, missing_raw, missing_edition = build_charter_inputs(
        gold_ids, args.raw_json, args.edition_source, cid_to_path
    )
    if missing_raw:
        raise RuntimeError(
            f"{len(missing_raw)} test charters not in raw JSON: "
            f"{missing_raw[:5]}...")
    if missing_edition:
        what = ("annotator-source text"
                if args.edition_source == "annotator"
                else "non-empty tokens in silver CoNLL")
        raise RuntimeError(
            f"{len(missing_edition)} test charters have no {what}: "
            f"{missing_edition[:5]}...")
    edition_note = ("Edition from annotator file, Summary from raw JSON"
                    if args.edition_source == "annotator"
                    else "Edition reconstructed from silver CoNLL, "
                         "Summary from raw JSON")
    print(f"  Constructed {len(charters)} pipeline inputs "
          f"({edition_note}).")

    # ── Load gazetteers (same paths used upstream of NER training) ──
    print("\n[1/3] Loading gazetteers...")
    tora_exact, tora_fuzzy, tora_sources, tora_stats = load_tora(args.tora_path)
    df_exact,   df_sources,   df_stats   = load_df(args.df_path)
    smp_exact,  smp_fuzzy,   smp_sources, smp_stats = load_smp(args.smp_path)
    # Merge DF into TORA (identical logic to scripts/run_projection.py)
    tora_exact |= df_exact
    tora_sources.update(
        {k: v for k, v in df_sources.items() if k not in tora_sources}
    )
    tora_fuzzy = sorted(tora_exact)
    print(f"  TORA+DF:  {len(tora_exact)} lookup names")
    print(f"  SMP:      {smp_stats['unique_names']} lookup names")

    # ── Run NER on summaries (Signal 1 input) ──
    print("\n[2/3] Running NER on modern summaries...")
    ner_results = run_ner_on_summaries(charters)

    # ── Run the projection pipeline, writing CoNLL output to a tmp dir ──
    print("\n[3/3] Running pipeline...")
    with tempfile.TemporaryDirectory(prefix="pipeline_eval_") as tmpdir:
        pipeline_results = run_pipeline(
            charters,
            tora_data=(tora_exact, tora_fuzzy, tora_sources),
            smp_data=(smp_exact, smp_fuzzy, smp_sources),
            ner_results=ner_results,
            output_dir=tmpdir,
        )

        # ── Align pipeline output against gold, per charter ──
        #
        # Both sides go through read_conll_file (same middle-dot filter,
        # same tokeniser upstream). Since Edition text is identical on
        # both sides (option A), token streams must be identical too.
        # We assert token identity per charter and fail loudly if not.
        true_labels, true_predictions, tokens_per_seq = [], [], []
        gold_entity_total = 0
        total_tokens      = 0
        entity_tokens     = 0
        per_charter_stats = []

        for cid in gold_ids:
            gold_path = cid_to_path[cid]
            pred_path = os.path.join(tmpdir, f"sdhk_{cid}_silver.conll")
            if not os.path.exists(pred_path):
                raise RuntimeError(
                    f"Pipeline did not produce output for charter {cid} "
                    f"(expected {pred_path}). "
                    f"Silent charter drop would bias micro averages; aborting."
                )
            gold_tok, gold_lab = flatten_conll(gold_path)
            pred_tok, pred_lab = flatten_conll(pred_path)

            # In silver-conll mode, normalise gold tokens the same way the
            # pipeline normalises Edition input (strip º/°, split multi-
            # punctuation tokens). Without this, 2/43 charters have token-
            # stream drift — not gold mislabelling, just that the silver
            # was hand-verified and retains chars the pipeline strips.
            if args.edition_source == "silver-conll":
                gold_tok, gold_lab = normalise_silver_for_pipeline(
                    gold_tok, gold_lab
                )

            assert len(gold_tok) == len(pred_tok), (
                f"Charter {cid}: token-count mismatch "
                f"(gold={len(gold_tok)}, pred={len(pred_tok)}). "
                f"Tokenisation drift invalidates the comparison — aborting."
            )
            if gold_tok != pred_tok:
                for i, (g, p) in enumerate(zip(gold_tok, pred_tok)):
                    if g != p:
                        raise AssertionError(
                            f"Charter {cid}: token mismatch at position {i}: "
                            f"gold={g!r} pred={p!r}. "
                            f"Context gold={gold_tok[max(0,i-2):i+3]} "
                            f"pred={pred_tok[max(0,i-2):i+3]}."
                        )

            true_labels.append(gold_lab)
            true_predictions.append(pred_lab)
            tokens_per_seq.append(gold_tok)

            n_ent = sum(1 for l in gold_lab if l.startswith("B-"))
            gold_entity_total += n_ent
            total_tokens      += len(gold_tok)
            entity_tokens     += sum(1 for l in gold_lab if l != "O")
            per_charter_stats.append({
                "charter_id":  cid,
                "tokens":      len(gold_tok),
                "gold_entities": n_ent,
                "pred_entities": sum(1 for l in pred_lab if l.startswith("B-")),
            })

    # ── Sanity checks ──
    assert len(true_labels) == len(gold_ids), (
        f"Expected {len(gold_ids)} charters, got {len(true_labels)}. "
        f"Something in the file iteration failed silently.")
    # The 1739-entity total is specific to the 75-charter expert gold set.
    if args.edition_source == "annotator" and len(gold_ids) == 75:
        if gold_entity_total != 1739:
            print(f"  ⚠ Total gold entity count = {gold_entity_total}, "
                  f"NER eval reported 1739. Investigate before trusting numbers.")
        else:
            print(f"  ✓ Gold entity total matches NER eval: 1739.")
    print(f"  ✓ All {len(gold_ids)} charters passed per-charter "
          f"token-identity assertion.")
    print(f"  Tokens (post middle-dot filter): {total_tokens}")
    print(f"  Entity tokens: {entity_tokens} "
          f"({100 * entity_tokens / total_tokens:.1f}%)")
    print(f"  Gold B- entities: {gold_entity_total}")

    # ── Compute + print + dump (shared with evaluate_ner_v2.py) ──
    edition_text_source = (
        "expert_annot/test_set/annotator_*_test.txt — same body "
        "extraction used by the gold CoNLL builder, to guarantee "
        "per-charter token alignment"
        if args.edition_source == "annotator"
        else
        f"raw JSON Edition field, unchanged ({args.raw_json})"
    )
    extra_fields = {
        "test_dir":        args.test_dir,
        "raw_json":        args.raw_json,
        "num_charters":    len(true_labels),
        "protocol":        ("three-signal projection "
                            "(Signal 1 NER + Signal 2 gazetteer + "
                            "Signal 3 capitalisation, voting)"),
        "edition_source":      args.edition_source,
        "edition_text_source": edition_text_source,
        "pipeline_config": {
            "ner_model":      config.NER_MODEL_NAME,
            "ner_aggregation": config.NER_AGGREGATION,
            "tora_path":      args.tora_path,
            "df_path":        args.df_path,
            "smp_path":       args.smp_path,
            "s1_fuzzy_threshold":   config.S1_FUZZY_THRESHOLD,
            "s1_consonant_threshold": config.S1_CONSONANT_THRESHOLD,
            "s2_fuzzy_threshold":   config.S2_FUZZY_THRESHOLD,
        },
        "versions": versions,
        "data": {
            "num_charters":    len(true_labels),
            "total_tokens":    total_tokens,
            "entity_tokens":   entity_tokens,
            "gold_entities":   gold_entity_total,
        },
        "per_charter": per_charter_stats,
    }

    compute_and_report(
        true_labels,
        true_predictions,
        tokens_per_seq,
        source="pipeline",
        output_path=args.output_json,
        extra_fields=extra_fields,
    )


if __name__ == "__main__":
    main()
