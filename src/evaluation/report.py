"""
Shared reporting logic for NER evaluation: entity-level (seqeval) +
token-level (sklearn, O excluded from aggregates) + FP/FN error
diagnostic + JSON dump. Imported by both scripts/evaluate_ner_v2.py
(model evaluation) and scripts/evaluate_pipeline.py (projection-pipeline
evaluation) so the two cannot drift apart — the thesis compares their
numbers directly.

Single entry point:
    compute_and_report(true_labels, true_predictions, tokens_per_seq,
                       *, source, output_path, extra_fields=None)
"""

import datetime
import json

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.metrics.sequence_labeling import get_entities
from sklearn.metrics import (
    accuracy_score,
    classification_report as sk_classification_report,
    precision_recall_fscore_support,
)


# ================================================================
# Error diagnostic
# ================================================================


def classify_errors(true_labels, true_predictions, tokens_per_seq):
    """
    For every FN (missed gold entity) and FP (spurious prediction),
    assign a category:

      · type_confusion      — gold and pred have identical span but
                              differ only in type (Person ↔ Location).
                              Contributes 1 FN + 1 FP.
      · boundary_same_type  — overlapping spans, same type, different
                              boundaries (off-by-a-token on either side).
                              Contributes 1 FN + 1 FP.
      · boundary_and_type   — overlapping spans, different types AND
                              different boundaries. 1 FN + 1 FP.
      · pure_miss           — gold entity with no overlapping prediction
                              in that sequence. Contributes 1 FN only.
      · pure_hallucination  — predicted span with no overlapping gold
                              entity. Contributes 1 FP only.

    Each FN is paired greedily with a single FP in the same sequence;
    any FP left unpaired becomes a pure_hallucination, any FN left
    unpaired becomes a pure_miss.
    """
    cats = {
        "type_confusion":     [],
        "boundary_same_type": [],
        "boundary_and_type":  [],
        "pure_miss":          [],
        "pure_hallucination": [],
    }

    def _overlap(a, b):  # a, b = (type, start, end_incl)
        return a[1] <= b[2] and b[1] <= a[2]

    for seq_idx, (gl, pl, toks) in enumerate(
        zip(true_labels, true_predictions, tokens_per_seq)
    ):
        gold = set(get_entities(gl))  # seqeval: (type, start, end_incl)
        pred = set(get_entities(pl))
        fn_set = sorted(gold - pred)
        fp_set = sorted(pred - gold)
        used_fp_idx = set()

        def _pair(predicate, fn_set):
            matched, left = [], []
            for g in fn_set:
                best = None
                for i, f in enumerate(fp_set):
                    if i in used_fp_idx:
                        continue
                    if predicate(g, f):
                        best = (i, f)
                        break
                if best is not None:
                    used_fp_idx.add(best[0])
                    matched.append((g, best[1]))
                else:
                    left.append(g)
            return matched, left

        # 1. same span, different type
        matched, fn_set = _pair(
            lambda g, f: g[1] == f[1] and g[2] == f[2] and g[0] != f[0],
            fn_set,
        )
        for g, f in matched:
            cats["type_confusion"].append({
                "seq": seq_idx, "gold_type": g[0], "pred_type": f[0],
                "span_tokens": toks[g[1]: g[2] + 1],
            })

        # 2. overlap, same type, different span
        matched, fn_set = _pair(
            lambda g, f: g[0] == f[0] and _overlap(g, f),
            fn_set,
        )
        for g, f in matched:
            cats["boundary_same_type"].append({
                "seq": seq_idx, "type": g[0],
                "gold_tokens": toks[g[1]: g[2] + 1],
                "pred_tokens": toks[f[1]: f[2] + 1],
            })

        # 3. overlap, different type AND different span
        matched, fn_set = _pair(
            lambda g, f: g[0] != f[0] and _overlap(g, f),
            fn_set,
        )
        for g, f in matched:
            cats["boundary_and_type"].append({
                "seq": seq_idx,
                "gold_type": g[0], "pred_type": f[0],
                "gold_tokens": toks[g[1]: g[2] + 1],
                "pred_tokens": toks[f[1]: f[2] + 1],
            })

        # 4. remaining FN → pure miss
        for g in fn_set:
            cats["pure_miss"].append({
                "seq": seq_idx, "type": g[0],
                "tokens": toks[g[1]: g[2] + 1],
            })

        # 5. remaining FP → pure hallucination
        for i, f in enumerate(fp_set):
            if i in used_fp_idx:
                continue
            cats["pure_hallucination"].append({
                "seq": seq_idx, "type": f[0],
                "tokens": toks[f[1]: f[2] + 1],
            })

    return cats


def print_error_diagnostic(cats, fp_total, fn_total, max_examples=5):
    """Print a human-readable diagnostic summary."""
    n_type  = len(cats["type_confusion"])
    n_bst   = len(cats["boundary_same_type"])
    n_bat   = len(cats["boundary_and_type"])
    n_miss  = len(cats["pure_miss"])
    n_hal   = len(cats["pure_hallucination"])
    n_total = n_type + n_bst + n_bat + n_miss + n_hal

    fp_acc = n_type + n_bst + n_bat + n_hal
    fn_acc = n_type + n_bst + n_bat + n_miss

    print()
    print("=" * 72)
    print(f"  Error diagnostic — how the {fp_total} FP and {fn_total} FN decompose")
    print("=" * 72)
    print(f"    {'category':<22} {'events':>7} {'FN':>5} {'FP':>5}")
    print(f"    {'type_confusion':<22} {n_type:>7} {n_type:>5} {n_type:>5}  "
          f"(same span, wrong type)")
    print(f"    {'boundary_same_type':<22} {n_bst:>7} {n_bst:>5} {n_bst:>5}  "
          f"(overlap, correct type, wrong boundary)")
    print(f"    {'boundary_and_type':<22} {n_bat:>7} {n_bat:>5} {n_bat:>5}  "
          f"(overlap, wrong type and boundary)")
    print(f"    {'pure_miss':<22} {n_miss:>7} {n_miss:>5}     0  "
          f"(gold entity with no overlapping pred)")
    print(f"    {'pure_hallucination':<22} {n_hal:>7}     0 {n_hal:>5}  "
          f"(pred span with no overlapping gold)")
    print(f"    {'TOTAL':<22} {n_total:>7} {fn_acc:>5} {fp_acc:>5}")
    print()
    print(f"    Accounting check: "
          f"reconstructed FN={fn_acc} (seqeval reports {fn_total}); "
          f"reconstructed FP={fp_acc} (seqeval reports {fp_total}).")

    def _show(title, items, fmt):
        if not items:
            return
        print()
        print(f"  Examples — {title} (showing up to {max_examples} of {len(items)}):")
        for it in items[:max_examples]:
            print(f"    {fmt(it)}")

    _show("type_confusion", cats["type_confusion"],
          lambda it: f"{it['gold_type']:<9} →  pred {it['pred_type']:<9}  "
                     f"span={' '.join(it['span_tokens'])!r}")
    _show("boundary_same_type", cats["boundary_same_type"],
          lambda it: f"{it['type']:<9}  gold={' '.join(it['gold_tokens'])!r}  "
                     f"pred={' '.join(it['pred_tokens'])!r}")
    _show("boundary_and_type", cats["boundary_and_type"],
          lambda it: f"{it['gold_type']}→{it['pred_type']}  "
                     f"gold={' '.join(it['gold_tokens'])!r}  "
                     f"pred={' '.join(it['pred_tokens'])!r}")
    _show("pure_miss", cats["pure_miss"],
          lambda it: f"{it['type']:<9}  gold={' '.join(it['tokens'])!r}  "
                     f"(no overlapping prediction)")
    _show("pure_hallucination", cats["pure_hallucination"],
          lambda it: f"{it['type']:<9}  pred={' '.join(it['tokens'])!r}  "
                     f"(no overlapping gold entity)")


# ================================================================
# Metrics + report + JSON dump — the shared entry point
# ================================================================


def compute_and_report(
    true_labels,
    true_predictions,
    tokens_per_seq,
    *,
    source,
    output_path,
    extra_fields=None,
):
    """
    Run the full thesis-reporting pipeline on already-aligned
    gold/prediction sequences:

      · entity-level (strict) metrics via seqeval
      · per-class TP/FP/FN reconstruction via get_entities
      · token-level (IO, O excluded from micro) metrics via sklearn
      · error-category diagnostic
      · printed results block + JSON file

    Args:
        true_labels, true_predictions:
            list-of-lists of BIO strings, aligned one-to-one.
        tokens_per_seq:
            list-of-lists of token strings, same shape as true_labels;
            only used for the error diagnostic examples.
        source:
            "model" or "pipeline" — stored in the top-level JSON under
            "source" so downstream consumers can tell the two runs apart.
        output_path:
            where to write the JSON results file.
        extra_fields:
            optional dict merged into the JSON payload (after the metric
            keys), for run-specific metadata (model id, gazetteer paths,
            version fingerprint, data stats, …).

    Returns:
        the payload dict that was written.
    """
    # --- entity-level aggregates ---
    p_micro = precision_score(true_labels, true_predictions, average="micro")
    r_micro = recall_score(true_labels, true_predictions, average="micro")
    f1_micro = f1_score(true_labels, true_predictions, average="micro")
    p_macro = precision_score(true_labels, true_predictions, average="macro")
    r_macro = recall_score(true_labels, true_predictions, average="macro")
    f1_macro = f1_score(true_labels, true_predictions, average="macro")
    p_weighted  = precision_score(true_labels, true_predictions, average="weighted")
    r_weighted  = recall_score(true_labels, true_predictions, average="weighted")
    f1_weighted = f1_score(true_labels, true_predictions, average="weighted")

    # --- per-class TP/FP/FN (raw counts, so CIs can be computed later) ---
    gold_ents = {(i,) + e for i, s in enumerate(true_labels) for e in get_entities(s)}
    pred_ents = {(i,) + e for i, s in enumerate(true_predictions) for e in get_entities(s)}
    by_type = {}
    for t in sorted({e[1] for e in gold_ents | pred_ents}):
        g = {e for e in gold_ents if e[1] == t}
        pr = {e for e in pred_ents if e[1] == t}
        tp = len(g & pr)
        fp = len(pr - g)
        fn = len(g - pr)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1t  = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        by_type[t] = dict(tp=tp, fp=fp, fn=fn,
                          support=tp + fn, pred=tp + fp,
                          precision=prec, recall=rec, f1=f1t)

    tp_all = sum(v["tp"] for v in by_type.values())
    fp_all = sum(v["fp"] for v in by_type.values())
    fn_all = sum(v["fn"] for v in by_type.values())

    # --- token-level (IO, O excluded from micro/macro/weighted) ---
    def _strip_bio(lab):
        return lab.split("-", 1)[1] if lab != "O" else "O"

    flat_gold = [_strip_bio(l) for seq in true_labels      for l in seq]
    flat_pred = [_strip_bio(l) for seq in true_predictions for l in seq]

    tok_classes = ["Person", "Location"]
    tok_p_micro, tok_r_micro, tok_f1_micro, _ = precision_recall_fscore_support(
        flat_gold, flat_pred, labels=tok_classes, average="micro", zero_division=0
    )
    tok_p_macro, tok_r_macro, tok_f1_macro, _ = precision_recall_fscore_support(
        flat_gold, flat_pred, labels=tok_classes, average="macro", zero_division=0
    )
    tok_p_weighted, tok_r_weighted, tok_f1_weighted, _ = precision_recall_fscore_support(
        flat_gold, flat_pred, labels=tok_classes, average="weighted", zero_division=0
    )
    tok_p_per, tok_r_per, tok_f1_per, tok_sup_per = precision_recall_fscore_support(
        flat_gold, flat_pred, labels=tok_classes, average=None, zero_division=0
    )
    tok_by_class = {
        tok_classes[i]: dict(
            precision=float(tok_p_per[i]),
            recall=float(tok_r_per[i]),
            f1=float(tok_f1_per[i]),
            support=int(tok_sup_per[i]),
        )
        for i in range(len(tok_classes))
    }
    tok_accuracy_incl_O = accuracy_score(flat_gold, flat_pred)

    # --- print ---
    print("=" * 72)
    print(f"  Results (full precision — thesis-reporting format, source={source})")
    print("=" * 72)
    print()
    print("  Aggregate metrics:")
    print(f"    {'':<10} {'precision':>12} {'recall':>12} {'f1':>12}")
    print(f"    {'micro':<10} {p_micro:>12.6f} {r_micro:>12.6f} {f1_micro:>12.6f}")
    print(f"    {'macro':<10} {p_macro:>12.6f} {r_macro:>12.6f} {f1_macro:>12.6f}")
    print(f"    {'weighted':<10} "
          f"{p_weighted:>12.6f} {r_weighted:>12.6f} {f1_weighted:>12.6f}")
    print()
    print("  Per-class counts and metrics (entity-level, exact-match):")
    print(f"    {'class':<10} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'support':>8} {'predicted':>10} "
          f"{'precision':>12} {'recall':>12} {'f1':>12}")
    for t, v in by_type.items():
        print(f"    {t:<10} {v['tp']:>5} {v['fp']:>5} {v['fn']:>5} "
              f"{v['support']:>8} {v['pred']:>10} "
              f"{v['precision']:>12.6f} {v['recall']:>12.6f} {v['f1']:>12.6f}")
    print(f"    {'TOTAL':<10} {tp_all:>5} {fp_all:>5} {fn_all:>5} "
          f"{tp_all + fn_all:>8} {tp_all + fp_all:>10} "
          f"{p_micro:>12.6f} {r_micro:>12.6f} {f1_micro:>12.6f}")
    print()
    print("  seqeval classification_report (4-decimal, for cross-check):")
    print(classification_report(true_labels, true_predictions, digits=4))

    print("=" * 72)
    print("  Token-level results (IO, O excluded from micro)")
    print("=" * 72)
    print()
    print("  Aggregate metrics (entity classes only):")
    print(f"    {'':<10} {'precision':>12} {'recall':>12} {'f1':>12}")
    print(f"    {'micro':<10} {tok_p_micro:>12.6f} {tok_r_micro:>12.6f} "
          f"{tok_f1_micro:>12.6f}")
    print(f"    {'macro':<10} {tok_p_macro:>12.6f} {tok_r_macro:>12.6f} "
          f"{tok_f1_macro:>12.6f}")
    print(f"    {'weighted':<10} {tok_p_weighted:>12.6f} {tok_r_weighted:>12.6f} "
          f"{tok_f1_weighted:>12.6f}")
    print()
    print("  Per-class token metrics:")
    print(f"    {'class':<10} {'support':>8} "
          f"{'precision':>12} {'recall':>12} {'f1':>12}")
    for c in tok_classes:
        v = tok_by_class[c]
        print(f"    {c:<10} {v['support']:>8} "
              f"{v['precision']:>12.6f} {v['recall']:>12.6f} {v['f1']:>12.6f}")
    print()
    print(f"  Token accuracy (incl. O): {tok_accuracy_incl_O:.6f}  "
          f"(descriptive — dominated by the O class)")
    print()
    print("  sklearn classification_report (4-decimal, entity classes, "
          "for cross-check):")
    print(sk_classification_report(
        flat_gold, flat_pred, labels=tok_classes, digits=4, zero_division=0
    ))

    # --- error diagnostic ---
    cats = classify_errors(true_labels, true_predictions, tokens_per_seq)
    print_error_diagnostic(cats, fp_total=fp_all, fn_total=fn_all)

    # --- JSON payload ---
    payload = {
        "source":          source,
        "timestamp_utc":   datetime.datetime.utcnow().isoformat() + "Z",
        "aggregate_micro":    dict(precision=p_micro,    recall=r_micro,    f1=f1_micro),
        "aggregate_macro":    dict(precision=p_macro,    recall=r_macro,    f1=f1_macro),
        "aggregate_weighted": dict(precision=p_weighted, recall=r_weighted, f1=f1_weighted),
        "per_class":          by_type,
        "confusion_counts":   dict(tp=tp_all, fp=fp_all, fn=fn_all),
        "token_level": {
            "note": "IO labels (BIO prefix stripped). Micro/macro/weighted "
                    "computed over entity classes only (O excluded). "
                    "accuracy_incl_O is descriptive and counts every token.",
            "aggregate_micro": dict(
                precision=float(tok_p_micro),
                recall=float(tok_r_micro),
                f1=float(tok_f1_micro),
            ),
            "aggregate_macro": dict(
                precision=float(tok_p_macro),
                recall=float(tok_r_macro),
                f1=float(tok_f1_macro),
            ),
            "aggregate_weighted": dict(
                precision=float(tok_p_weighted),
                recall=float(tok_r_weighted),
                f1=float(tok_f1_weighted),
            ),
            "per_class":        tok_by_class,
            "accuracy_incl_O":  float(tok_accuracy_incl_O),
        },
        "error_categories": {
            k: {"count": len(v), "items": v} for k, v in cats.items()
        },
    }
    if extra_fields:
        payload.update(extra_fields)

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Full results (with 15+ digits of floating-point precision) "
          f"written to:\n    {output_path}")

    return payload
