"""
Signal 3: Capitalisation Extraction.

Extract capitalised tokens from Old Swedish edition texts as entity
candidates, applying stoplists (Latin/Swedish function words) and
type heuristics (preposition context, patronymics, titles, admin suffixes).
"""

import re

from config import (
    S3_LATIN_STOPLIST,
    S3_SWEDISH_STOPLIST,
    S3_LOCATIVE_PREPOSITIONS,
    S3_TITLE_WORDS,
    S3_ADMIN_SUFFIXES,
    ROMAN_NUMERAL_RE,
)
from src.preprocessing.clean_text import is_sentence_initial


# ============================================================
# CAPITALISATION EXTRACTION
# ============================================================

def signal_3_capitalisation(edition_tokens):
    """
    Extract capitalised tokens as entity candidates, applying stoplists
    and type heuristics.

    Args:
        edition_tokens: list of token dicts from tokenize_edition

    Returns:
        annotations: dict mapping token_idx -> {
            label, source, heuristic, bio_position
        }
        log: dict with candidates, stoplisted, sentence_initial, and stats
    """
    annotations = {}
    log_candidates = []
    log_stoplisted = []
    log_sentence_initial = []

    word_tokens = [t for t in edition_tokens if t["is_word"]]

    for token in word_tokens:
        tok_text = token["text"]
        tok_idx = token["idx"]

        # Only consider capitalised tokens
        if not tok_text or not tok_text[0].isupper():
            continue

        # Skip Roman numerals
        if ROMAN_NUMERAL_RE.match(tok_text):
            log_stoplisted.append({
                "token_text": tok_text,
                "token_idx": tok_idx,
                "reason": "roman_numeral",
            })
            continue

        # Skip sentence-initial tokens (capitalised by convention, not entity)
        if is_sentence_initial(edition_tokens, tok_idx):
            log_sentence_initial.append({
                "token_text": tok_text,
                "token_idx": tok_idx,
            })
            continue

        # Skip Latin stoplisted words
        if tok_text in S3_LATIN_STOPLIST:
            log_stoplisted.append({
                "token_text": tok_text,
                "token_idx": tok_idx,
                "reason": "latin_stoplist",
            })
            continue

        # Skip Swedish stoplisted words
        if tok_text in S3_SWEDISH_STOPLIST:
            log_stoplisted.append({
                "token_text": tok_text,
                "token_idx": tok_idx,
                "reason": "swedish_stoplist",
            })
            continue

        # --- This token is a capitalised entity candidate ---
        # Apply type heuristics

        label = "Unknown"
        heuristic = None

        # Heuristic 1: preceded by locative preposition -> Location
        prev_word = _get_prev_word(edition_tokens, tok_idx)
        if prev_word and prev_word.lower() in S3_LOCATIVE_PREPOSITIONS:
            label = "Location"
            heuristic = f"preceded_by_{prev_word}"

        # Heuristic 2: token itself is a patronymic -> Person
        if label == "Unknown":
            if _is_patronymic(tok_text):
                label = "Person"
                heuristic = "patronymic_suffix"

        # Heuristic 3: followed by a patronymic -> this token is likely a first name
        if label == "Unknown":
            next_word = _get_next_word(edition_tokens, tok_idx)
            if next_word and _is_patronymic(next_word):
                label = "Person"
                heuristic = f"followed_by_patronymic_{next_word}"

        # Heuristic 4: preceded by title "härrä"/"herra" -> Person
        if label == "Unknown":
            if prev_word and prev_word.lower() in S3_TITLE_WORDS:
                label = "Person"
                heuristic = f"preceded_by_title_{prev_word}"

        # Heuristic 5: followed by admin suffix -> Location
        if label == "Unknown":
            next_word = _get_next_word(edition_tokens, tok_idx)
            if next_word and next_word.lower() in S3_ADMIN_SUFFIXES:
                label = "Location"
                heuristic = f"followed_by_{next_word}"

        annotations[tok_idx] = {
            "label": label,
            "source": "signal_3_capitalisation",
            "heuristic": heuristic,
            "bio_position": "B",
        }

        log_candidates.append({
            "token_text": tok_text,
            "token_idx": tok_idx,
            "label": label,
            "heuristic": heuristic,
        })

    # Build log
    log = {
        "candidates": log_candidates,
        "stoplisted": log_stoplisted,
        "sentence_initial": log_sentence_initial,
        "candidate_count": len(log_candidates),
        "stoplisted_count": len(log_stoplisted),
        "sentence_initial_count": len(log_sentence_initial),
        "typed_count": sum(
            1 for c in log_candidates if c["label"] != "Unknown"
        ),
        "untyped_count": sum(
            1 for c in log_candidates if c["label"] == "Unknown"
        ),
        "person_count": sum(
            1 for c in log_candidates if c["label"] == "Person"
        ),
        "location_count": sum(
            1 for c in log_candidates if c["label"] == "Location"
        ),
    }

    return annotations, log


# ============================================================
# HELPERS
# ============================================================

def _get_prev_word(tokens, tok_idx):
    """Get the text of the nearest preceding word token, or None."""
    for i in range(tok_idx - 1, -1, -1):
        if tokens[i]["is_word"]:
            return tokens[i]["text"]
    return None


def _get_next_word(tokens, tok_idx):
    """Get the text of the nearest following word token, or None."""
    for i in range(tok_idx + 1, len(tokens)):
        if tokens[i]["is_word"]:
            return tokens[i]["text"]
    return None


def _is_patronymic(text):
    """
    Check if a token looks like a patronymic name ending.
    Common Old Swedish patterns: -son, -sson, -dotter, -dottir, -dottor.
    """
    lower = text.lower()
    return bool(re.search(
        r"(?:s?son|s?sons|dotter|dottir|dottor|dottær|dottærs)$",
        lower,
    ))
