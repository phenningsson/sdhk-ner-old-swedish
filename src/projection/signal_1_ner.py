"""
Signal 1: NER Projection.

1. Run KBLab/bert-base-swedish-cased-ner on modern Swedish summaries
2. Extract Person and Location entities (log ORG/MISC separately)
3. Project each entity onto the Old Swedish edition text via progressive
   fuzzy matching: exact -> consonant skeleton -> Levenshtein
"""

import re

from rapidfuzz import fuzz
from transformers import logging as hf_logging
from transformers import pipeline as hf_pipeline

from config import (
    NER_MODEL_NAME,
    NER_AGGREGATION,
    S1_FUZZY_THRESHOLD,
    S1_CONSONANT_THRESHOLD,
    S1_CONSONANT_MIN_LEN,
)
from src.preprocessing.clean_text import consonant_skeleton

hf_logging.set_verbosity_error()

# BERT max input length (tokens); we use a conservative character estimate
_MAX_CHARS = 1500  # ~400 BERT tokens — safe margin below 512 limit


def _run_ner_safe(ner, text):
    """
    Run NER on text, chunking if it exceeds BERT's 512-token limit.

    Splits long texts on sentence boundaries (". ") with overlap so that
    entities spanning chunk borders are not lost.  Deduplicates results by
    (start, end) character offsets in the original text.
    """
    if len(text) <= _MAX_CHARS:
        return ner(text)

    # Split into sentence-boundary chunks with ~200-char overlap
    overlap = 200
    chunks = []
    pos = 0
    while pos < len(text):
        end = pos + _MAX_CHARS
        if end >= len(text):
            chunks.append((pos, len(text)))
            break
        # Try to split on ". " near the end of the chunk
        split_at = text.rfind(". ", pos + _MAX_CHARS - overlap, end)
        if split_at == -1:
            # Fall back to space
            split_at = text.rfind(" ", pos + _MAX_CHARS - overlap, end)
        if split_at == -1:
            split_at = end
        else:
            split_at += 2 if text[split_at:split_at + 2] == ". " else 1
        chunks.append((pos, split_at))
        pos = max(pos + 1, split_at - overlap)

    # Run NER on each chunk and map offsets back to original text
    seen = set()
    merged = []
    for chunk_start, chunk_end in chunks:
        chunk_text = text[chunk_start:chunk_end]
        for ent in ner(chunk_text):
            abs_start = ent["start"] + chunk_start
            abs_end = ent["end"] + chunk_start
            key = (abs_start, abs_end)
            if key not in seen:
                seen.add(key)
                ent["start"] = abs_start
                ent["end"] = abs_end
                merged.append(ent)

    merged.sort(key=lambda e: e["start"])
    return merged


# ============================================================
# NER ON SUMMARIES
# ============================================================


def run_ner_on_summaries(charters, model_name=None):
    """
    Run Swedish NER on the modern summary of each charter.

    Args:
        charters: list of charter dicts (must have 'Id' and 'Summary')
        model_name: HuggingFace model identifier (defaults to config)

    Returns:
        dict mapping charter_id (str) -> {
            'entities':      list of PER/LOC entities (mapped to Person/Location),
            'org_entities':  list of ORG entities (logged, not projected),
            'misc_entities': list of MISC/other entities (logged),
        }
    """
    if model_name is None:
        model_name = NER_MODEL_NAME

    print(f"  Loading NER model: {model_name}")
    ner = hf_pipeline("ner", model=model_name, aggregation_strategy=NER_AGGREGATION)
    print("  Model loaded.")

    results = {}

    for charter in charters:
        charter_id = str(charter["Id"])
        summary = charter["Summary"]

        # Strip editorial asterisks (*) used in SDHK summaries to mark
        # identified historical locations (e.g. "*Lerby" → "Lerby").
        # These hurt NER tokenization and downstream fuzzy matching.
        summary = summary.replace("*", "")

        ner_output = _run_ner_safe(ner, summary)

        entities = []
        org_entities = []
        misc_entities = []

        for ent in ner_output:
            info = {
                "text": ent["word"].strip(),
                "ner_label": ent["entity_group"],
                "score": round(float(ent["score"]), 4),
                "start": ent["start"],
                "end": ent["end"],
            }

            if ent["entity_group"] == "PER":
                info["mapped_label"] = "Person"
                entities.append(info)
            elif ent["entity_group"] == "LOC":
                info["mapped_label"] = "Location"
                entities.append(info)
            elif ent["entity_group"] == "ORG":
                info["mapped_label"] = "Organization"
                org_entities.append(info)
            else:
                info["mapped_label"] = ent["entity_group"]
                misc_entities.append(info)

        results[charter_id] = {
            "entities": entities,
            "org_entities": org_entities,
            "misc_entities": misc_entities,
        }

        per_count = sum(1 for e in entities if e["mapped_label"] == "Person")
        loc_count = sum(1 for e in entities if e["mapped_label"] == "Location")
        print(
            f"  Charter {charter_id}: "
            f"{per_count} PER, {loc_count} LOC, "
            f"{len(org_entities)} ORG, {len(misc_entities)} MISC"
        )

    return results


# ============================================================
# PROGRESSIVE FUZZY MATCHING
# ============================================================


def progressive_fuzzy_match(entity_text, word_tokens, threshold=None):
    """
    Match a modern entity string against edition word tokens using
    progressive matching:
        Level 1: Exact match (case-insensitive)
        Level 2: Consonant skeleton similarity (>= S1_CONSONANT_THRESHOLD)
        Level 3: Levenshtein ratio via rapidfuzz (>= threshold)

    Args:
        entity_text: entity string from the modern summary
        word_tokens: list of word-token dicts from the edition (is_word=True)
        threshold: minimum Levenshtein ratio for level 3 (defaults to config)

    Returns:
        list of match dicts sorted by quality (best first)
    """
    if threshold is None:
        threshold = S1_FUZZY_THRESHOLD

    entity_lower = entity_text.lower().strip()
    entity_word_count = len(entity_text.split())
    entity_cons = consonant_skeleton(entity_text)

    matches = []

    max_span = min(entity_word_count + 2, len(word_tokens) + 1)

    for span_len in range(1, max_span):
        for i in range(len(word_tokens) - span_len + 1):
            span = word_tokens[i : i + span_len]
            span_text = " ".join(t["text"] for t in span)
            span_lower = span_text.lower()
            wc_match = span_len == entity_word_count

            base = {
                "matched_text": span_text,
                "token_indices": [t["idx"] for t in span],
                "start": span[0]["start"],
                "end": span[-1]["end"],
                "word_count_match": wc_match,
            }

            # Level 1: Exact
            if span_lower == entity_lower:
                matches.append({**base, "score": 100, "match_level": "exact"})
                continue

            # Level 2: Consonant skeleton
            if entity_cons and len(entity_cons) >= S1_CONSONANT_MIN_LEN:
                span_cons = consonant_skeleton(span_text)
                if span_cons:
                    cons_score = fuzz.ratio(entity_cons, span_cons)
                    if cons_score >= S1_CONSONANT_THRESHOLD:
                        matches.append(
                            {
                                **base,
                                "score": cons_score,
                                "match_level": "consonant_skeleton",
                                "entity_skeleton": entity_cons,
                                "span_skeleton": span_cons,
                            }
                        )
                        continue

            # Level 3: Levenshtein
            lev_score = fuzz.ratio(entity_lower, span_lower)
            if lev_score >= threshold:
                matches.append(
                    {**base, "score": lev_score, "match_level": "levenshtein"}
                )

    # Sort: match_level priority, then word_count_match, then score
    level_priority = {"exact": 3, "consonant_skeleton": 2, "levenshtein": 1}
    matches.sort(
        key=lambda x: (
            level_priority.get(x["match_level"], 0),
            x["word_count_match"],
            x["score"],
        ),
        reverse=True,
    )

    # Remove overlapping matches (keep best)
    filtered = []
    used_indices = set()
    for m in matches:
        m_indices = set(m["token_indices"])
        if not m_indices & used_indices:
            filtered.append(m)
            used_indices.update(m_indices)

    return filtered


# ============================================================
# PROJECTION: NER ENTITIES -> EDITION TOKENS
# ============================================================


def project_ner_to_edition(ner_entities, edition_tokens, threshold=None):
    """
    Project NER entities from a summary onto edition tokens.

    Args:
        ner_entities: list of entity dicts from run_ner_on_summaries
        edition_tokens: list of token dicts from tokenize_edition
        threshold: fuzzy match threshold (defaults to config)

    Returns:
        annotations: dict mapping token_idx -> annotation dict
        log: dict with projected, unmatched, and counts
    """
    if threshold is None:
        threshold = S1_FUZZY_THRESHOLD

    word_tokens = [t for t in edition_tokens if t["is_word"]]

    annotations = {}
    log_projected = []
    log_unmatched = []

    for entity in ner_entities:
        entity_text = entity["text"]
        entity_label = entity["mapped_label"]

        matches = progressive_fuzzy_match(entity_text, word_tokens, threshold)

        if matches:
            best = matches[0]

            for i, tok_idx in enumerate(best["token_indices"]):
                if tok_idx not in annotations:
                    annotations[tok_idx] = {
                        "label": entity_label,
                        "source": "signal_1_ner",
                        "summary_entity": entity_text,
                        "match_score": best["score"],
                        "match_level": best["match_level"],
                        "ner_confidence": entity["score"],
                        "bio_position": "B" if i == 0 else "I",
                    }

            log_projected.append(
                {
                    "summary_entity": entity_text,
                    "label": entity_label,
                    "ner_confidence": entity["score"],
                    "best_match": {
                        "edition_text": best["matched_text"],
                        "score": best["score"],
                        "match_level": best["match_level"],
                        "token_indices": best["token_indices"],
                    },
                    "all_matches_count": len(matches),
                    "all_matches": matches[:10],
                }
            )
        else:
            log_unmatched.append(
                {
                    "summary_entity": entity_text,
                    "label": entity_label,
                    "ner_confidence": entity["score"],
                    "reason": "no_match_above_threshold",
                }
            )

    log = {
        "projected": log_projected,
        "unmatched": log_unmatched,
        "projected_count": len(log_projected),
        "unmatched_count": len(log_unmatched),
    }

    return annotations, log
