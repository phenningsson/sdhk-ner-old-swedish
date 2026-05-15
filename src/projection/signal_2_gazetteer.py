"""
Signal 2: Gazetteer Lookup.

Match edition tokens against TORA (medieval place names) and SMP
(medieval person names) gazetteers using exact + fuzzy matching,
with stoplists to reduce false positives.
"""

import re

from rapidfuzz import fuzz, process

from config import (
    S2_FUZZY_THRESHOLD,
    S2_FUZZY_MIN_LENGTH,
    S2_LENGTH_RATIO,
    S2_SWEDISH_STOPLIST,
    S2_LATIN_STOPLIST,
    ROMAN_NUMERAL_RE,
)


def _is_stoplisted(tok_lower):
    """Check if a lowercase token is in any Signal 2 stoplist."""
    if tok_lower in S2_SWEDISH_STOPLIST:
        return "swedish_stoplist"
    if tok_lower in S2_LATIN_STOPLIST:
        return "latin_stoplist"
    if ROMAN_NUMERAL_RE.match(tok_lower):
        return "roman_numeral"
    return None


# ============================================================
# GAZETTEER LOOKUP
# ============================================================

def signal_2_gazetteer(edition_tokens, tora_exact, tora_fuzzy, tora_sources,
                       smp_exact, smp_fuzzy, smp_sources):
    """
    Match edition tokens against TORA and SMP gazetteers.

    Args:
        edition_tokens: list of token dicts from tokenize_edition
        tora_exact:     set of lowercase TORA names for exact matching
        tora_fuzzy:     list of lowercase TORA names for fuzzy matching
        tora_sources:   dict mapping lowercase name -> original TORA entry
        smp_exact:      set of lowercase SMP names for exact matching
        smp_fuzzy:      list of lowercase SMP names for fuzzy matching
        smp_sources:    dict mapping lowercase name -> original SMP form

    Returns:
        annotations: dict mapping token_idx -> annotation dict
        log: dict with match details and statistics
    """
    word_tokens = [t for t in edition_tokens if t["is_word"]]

    annotations = {}
    log_matches = []
    log_stoplisted = []

    for token in word_tokens:
        tok_text = token["text"]
        tok_lower = tok_text.lower()
        tok_idx = token["idx"]
        is_capitalised = tok_text[0].isupper() if tok_text else False

        # --- Check stoplists first ---
        stop_reason = _is_stoplisted(tok_lower)
        if stop_reason:
            would_match_tora = tok_lower in tora_exact
            would_match_smp = tok_lower in smp_exact
            if would_match_tora or would_match_smp:
                log_stoplisted.append({
                    "token_text": tok_text,
                    "token_idx": tok_idx,
                    "reason": stop_reason,
                    "would_match_tora": would_match_tora,
                    "would_match_smp": would_match_smp,
                })
            continue

        tora_match = None
        smp_match = None

        # --- Exact matching ---
        if tok_lower in tora_exact:
            tora_match = {
                "gazetteer_name": tora_sources.get(tok_lower, tok_lower),
                "score": 100,
                "match_level": "exact",
            }
        if tok_lower in smp_exact:
            smp_match = {
                "gazetteer_name": smp_sources.get(tok_lower, tok_lower),
                "score": 100,
                "match_level": "exact",
            }

        # --- Fuzzy matching (capitalised tokens only, if long enough) ---
        if is_capitalised and len(tok_text) >= S2_FUZZY_MIN_LENGTH:
            if not tora_match:
                result = process.extractOne(
                    tok_lower,
                    tora_fuzzy,
                    scorer=fuzz.ratio,
                    score_cutoff=S2_FUZZY_THRESHOLD,
                )
                if result:
                    matched_name, score, _ = result
                    len_ratio = min(len(tok_lower), len(matched_name)) / max(len(tok_lower), len(matched_name))
                    if not _is_stoplisted(matched_name) and len_ratio >= S2_LENGTH_RATIO:
                        tora_match = {
                            "gazetteer_name": tora_sources.get(matched_name, matched_name),
                            "score": score,
                            "match_level": "fuzzy",
                        }

            if not smp_match:
                result = process.extractOne(
                    tok_lower,
                    smp_fuzzy,
                    scorer=fuzz.ratio,
                    score_cutoff=S2_FUZZY_THRESHOLD,
                )
                if result:
                    matched_name, score, _ = result
                    len_ratio = min(len(tok_lower), len(matched_name)) / max(len(tok_lower), len(matched_name))
                    if not _is_stoplisted(matched_name) and len_ratio >= S2_LENGTH_RATIO:
                        smp_match = {
                            "gazetteer_name": smp_sources.get(matched_name, matched_name),
                            "score": score,
                            "match_level": "fuzzy",
                        }

        # --- Skip if no match ---
        if not tora_match and not smp_match:
            continue

        # --- Resolve label when both match ---
        match_info = {
            "token_text": tok_text,
            "token_idx": tok_idx,
            "is_capitalised": is_capitalised,
        }

        if tora_match:
            match_info["tora_match"] = tora_match
        if smp_match:
            match_info["smp_match"] = smp_match

        conflict = bool(tora_match and smp_match)
        match_info["conflict"] = conflict

        if tora_match and smp_match:
            if tora_match["score"] >= smp_match["score"]:
                label = "Location"
                source = "TORA"
                score = tora_match["score"]
            else:
                label = "Person"
                source = "SMP"
                score = smp_match["score"]
        elif tora_match:
            label = "Location"
            source = "TORA"
            score = tora_match["score"]
        else:
            label = "Person"
            source = "SMP"
            score = smp_match["score"]

        match_info["resolved_label"] = label
        match_info["resolved_source"] = source
        match_info["resolved_score"] = score

        annotations[tok_idx] = {
            "label": label,
            "source": f"signal_2_{source.lower()}",
            "gazetteer_name": (
                tora_match["gazetteer_name"] if source == "TORA"
                else smp_match["gazetteer_name"]
            ),
            "gazetteer_score": score,
            "match_level": (
                tora_match["match_level"] if source == "TORA"
                else smp_match["match_level"]
            ),
            "bio_position": "B",
        }

        log_matches.append(match_info)

    # --- Build log ---
    log = {
        "matches": log_matches,
        "stoplisted": log_stoplisted,
        "match_count": len(log_matches),
        "stoplisted_count": len(log_stoplisted),
        "tora_matches": sum(
            1 for m in log_matches if m.get("resolved_source") == "TORA"
        ),
        "smp_matches": sum(
            1 for m in log_matches if m.get("resolved_source") == "SMP"
        ),
        "conflicts": sum(1 for m in log_matches if m.get("conflict")),
        "exact_matches": sum(
            1
            for m in log_matches
            if (m.get("tora_match", {}).get("match_level") == "exact"
                or m.get("smp_match", {}).get("match_level") == "exact")
        ),
        "fuzzy_matches": sum(
            1
            for m in log_matches
            if (m.get("tora_match", {}).get("match_level") == "fuzzy"
                or m.get("smp_match", {}).get("match_level") == "fuzzy")
            and m.get("tora_match", {}).get("match_level") != "exact"
            and m.get("smp_match", {}).get("match_level") != "exact"
        ),
    }

    return annotations, log
