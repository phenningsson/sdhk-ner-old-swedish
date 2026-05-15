"""
Voting: Combine three signal annotations into final BIO labels.

Implements signal combination with:
- Type priority: Signal 1 (NER) > Signal 2 (Gazetteer) > Signal 3 (Capitalisation)
- Patronymic continuation rule
- Title stripping from Person span starts
- Preposition stripping from Location span starts
- Capitalisation-only with Unknown type → O
"""

import re

from config import PATRONYMIC_RE, TITLE_WORDS, LOCATION_PREFIX_STOP


# ============================================================
# VOTING
# ============================================================

def combine_signals(edition_tokens, s1_annotations, s2_annotations, s3_annotations):
    """
    Combine three signal annotations into final BIO labels.

    Args:
        edition_tokens:  list of token dicts from tokenize_edition
        s1_annotations:  dict tok_idx -> annotation from signal_1_ner
        s2_annotations:  dict tok_idx -> annotation from signal_2_gazetteer
        s3_annotations:  dict tok_idx -> annotation from signal_3_capitalisation

    Returns:
        final_labels:  list of BIO label strings, one per token
        log:           dict with detailed voting info per token and summary stats
    """
    final_labels = []
    token_logs = []

    prev_label = None
    prev_was_entity = False

    for token in edition_tokens:
        idx = token["idx"]
        tok_text = token["text"]
        is_word = token["is_word"]

        s1 = s1_annotations.get(idx)
        s2 = s2_annotations.get(idx)
        s3 = s3_annotations.get(idx)

        signals_present = []
        if s1:
            signals_present.append("signal_1")
        if s2:
            signals_present.append("signal_2")
        if s3:
            signals_present.append("signal_3")

        n_signals = len(signals_present)

        # --- Non-word tokens (punctuation) are always O ---
        if not is_word:
            final_labels.append("O")
            prev_label = None
            prev_was_entity = False
            continue

        # --- No signal flagged this token ---
        if n_signals == 0:
            # Patronymic continuation
            if (
                prev_was_entity
                and prev_label == "Person"
                and PATRONYMIC_RE.search(tok_text)
            ):
                final_bio = "I-Person"
                final_labels.append(final_bio)
                token_logs.append({
                    "token_idx": idx,
                    "token_text": tok_text,
                    "signals": [],
                    "n_signals": 0,
                    "decision": final_bio,
                    "reason": "patronymic_continuation",
                })
                continue

            final_labels.append("O")
            token_logs.append({
                "token_idx": idx,
                "token_text": tok_text,
                "signals": [],
                "n_signals": 0,
                "decision": "O",
                "reason": "no_signal",
            })
            prev_label = None
            prev_was_entity = False
            continue

        # --- At least one signal flagged this token ---

        # Collect type votes
        type_votes = {}
        if s1:
            type_votes["signal_1"] = s1["label"]
        if s2:
            type_votes["signal_2"] = s2["label"]
        if s3:
            type_votes["signal_3"] = s3["label"]

        # Determine entity type by priority: S1 > S2 > S3
        resolved_type = None
        type_source = None

        if s1 and s1["label"] in ("Person", "Location"):
            resolved_type = s1["label"]
            type_source = "signal_1"
        elif s2 and s2["label"] in ("Person", "Location"):
            resolved_type = s2["label"]
            type_source = "signal_2"
        elif s3 and s3["label"] in ("Person", "Location"):
            resolved_type = s3["label"]
            type_source = "signal_3"

        # --- Only Signal 3 with Unknown type → O ---
        if resolved_type is None:
            final_labels.append("O")
            token_logs.append({
                "token_idx": idx,
                "token_text": tok_text,
                "signals": signals_present,
                "n_signals": n_signals,
                "type_votes": type_votes,
                "decision": "O",
                "reason": "only_capitalisation_unknown_type",
            })
            prev_label = None
            prev_was_entity = False
            continue

        # --- Strip titles from Person span starts ---
        if (
            tok_text.lower() in TITLE_WORDS
            and resolved_type == "Person"
            and not (prev_was_entity and prev_label == "Person")
        ):
            final_labels.append("O")
            token_logs.append({
                "token_idx": idx,
                "token_text": tok_text,
                "signals": signals_present,
                "n_signals": n_signals,
                "type_votes": type_votes,
                "decision": "O",
                "reason": "title_stripped",
            })
            prev_label = None
            prev_was_entity = False
            continue

        # --- Strip prepositions from Location span starts ---
        if (
            tok_text.lower() in LOCATION_PREFIX_STOP
            and resolved_type == "Location"
            and not (prev_was_entity and prev_label == "Location")
        ):
            final_labels.append("O")
            token_logs.append({
                "token_idx": idx,
                "token_text": tok_text,
                "signals": signals_present,
                "n_signals": n_signals,
                "type_votes": type_votes,
                "decision": "O",
                "reason": "location_prefix_stripped",
            })
            prev_label = None
            prev_was_entity = False
            continue

        # --- Determine B vs I prefix ---
        bio_prefix = "B"

        if s1 and s1.get("bio_position") == "I":
            if prev_was_entity and prev_label == resolved_type:
                bio_prefix = "I"
        elif prev_was_entity and prev_label == resolved_type:
            bio_prefix = "I"

        final_bio = f"{bio_prefix}-{resolved_type}"
        final_labels.append(final_bio)

        type_values = [v for v in type_votes.values() if v in ("Person", "Location")]
        type_disagreement = len(set(type_values)) > 1

        token_logs.append({
            "token_idx": idx,
            "token_text": tok_text,
            "signals": signals_present,
            "n_signals": n_signals,
            "type_votes": type_votes,
            "resolved_type": resolved_type,
            "type_source": type_source,
            "type_disagreement": type_disagreement,
            "bio_prefix": bio_prefix,
            "decision": final_bio,
            "reason": "voted_entity",
            "signal_details": {
                "signal_1": _summarize_signal(s1, "s1") if s1 else None,
                "signal_2": _summarize_signal(s2, "s2") if s2 else None,
                "signal_3": _summarize_signal(s3, "s3") if s3 else None,
            },
        })

        prev_label = resolved_type
        prev_was_entity = True

    # --- Build summary statistics ---
    entity_tokens = [l for l in final_labels if l != "O"]
    person_tokens = [l for l in entity_tokens if "Person" in l]
    location_tokens = [l for l in entity_tokens if "Location" in l]

    disagreements = [t for t in token_logs if t.get("type_disagreement")]
    cap_only_unknown = [
        t for t in token_logs
        if t.get("reason") == "only_capitalisation_unknown_type"
    ]

    signal_combos = {}
    for t in token_logs:
        if t.get("reason") == "voted_entity":
            combo = "+".join(sorted(t["signals"]))
            signal_combos[combo] = signal_combos.get(combo, 0) + 1

    log = {
        "token_logs": token_logs,
        "total_tokens": len(edition_tokens),
        "word_tokens": sum(1 for t in edition_tokens if t["is_word"]),
        "entity_tokens": len(entity_tokens),
        "person_tokens": len(person_tokens),
        "location_tokens": len(location_tokens),
        "o_tokens": len(final_labels) - len(entity_tokens),
        "type_disagreements": len(disagreements),
        "disagreement_details": disagreements,
        "cap_only_unknown": len(cap_only_unknown),
        "cap_only_unknown_details": cap_only_unknown,
        "signal_combinations": signal_combos,
    }

    return final_labels, log


def _summarize_signal(annotation, prefix):
    """Create a compact summary of one signal's annotation for logging."""
    if prefix == "s1":
        return {
            "label": annotation.get("label"),
            "bio": annotation.get("bio_position"),
            "match_score": annotation.get("match_score"),
            "match_level": annotation.get("match_level"),
            "summary_entity": annotation.get("summary_entity"),
            "ner_confidence": annotation.get("ner_confidence"),
        }
    elif prefix == "s2":
        return {
            "label": annotation.get("label"),
            "gazetteer_name": annotation.get("gazetteer_name"),
            "gazetteer_score": annotation.get("gazetteer_score"),
            "match_level": annotation.get("match_level"),
        }
    elif prefix == "s3":
        return {
            "label": annotation.get("label"),
            "heuristic": annotation.get("heuristic"),
        }
    return annotation


# ============================================================
# FORMAT OUTPUT
# ============================================================

def labels_to_conll(edition_tokens, final_labels):
    """
    Format tokens and labels as CoNLL 2002 lines.

    Includes both word tokens (with their BIO labels) and punctuation
    tokens (tagged O).  Sentence-ending punctuation (. ! ?) is followed
    by a blank line to mark sentence boundaries for INCEpTION.

    Returns:
        list of strings (token lines and blank separator lines)
    """
    lines = []
    for token, label in zip(edition_tokens, final_labels):
        if token["is_word"]:
            lines.append(f"{token['text']} {label}")
        else:
            # Include punctuation tokens tagged as O
            lines.append(f"{token['text']} O")
            if token["text"] in ".!?":
                # Blank line = sentence boundary
                lines.append("")
    return lines


def extract_entity_spans(edition_tokens, final_labels):
    """
    Extract entity spans from the BIO-labelled token list.

    Returns:
        list of entity dicts: {text, label, tokens, token_indices}
    """
    entities = []
    current_entity = None

    for token, label in zip(edition_tokens, final_labels):
        if not token["is_word"]:
            continue

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]
            current_entity = {
                "text": token["text"],
                "label": entity_type,
                "tokens": [token["text"]],
                "token_indices": [token["idx"]],
            }
        elif label.startswith("I-") and current_entity:
            current_entity["text"] += " " + token["text"]
            current_entity["tokens"].append(token["text"])
            current_entity["token_indices"].append(token["idx"])
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities
