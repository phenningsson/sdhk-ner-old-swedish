"""
Load and clean the TORA (place names) and SMP (person names) gazetteers
for use in Signal 2 of the entity projection pipeline.

TORA: Topographical register from Riksarkivet (~34,400 entries).
SMP:  Sveriges Medeltida Personnamn from Isof (~3,500 rows -> ~6,000+ names).
"""

import csv
import re

from config import (
    TORA_SKIP_LINES,
    TORA_FILTER_TERMS,
    TORA_MAX_SPLIT_WORDS,
    TORA_COMMON_WORDS,
    TORA_ADMIN_WORDS,
)


def _is_valid_tora_word(word):
    """
    Check if a word extracted from a TORA entry is a plausible place name.

    Rejects:
    - Words starting with ( or [ (parenthetical context)
    - Pure numbers or year-like strings (e.g. "1012", "1630-1655")
    - Words starting with ... (truncated entries)
    - Words that are all punctuation/symbols
    """
    if not word:
        return False
    if word[0] in "([":
        return False
    if word.startswith("..."):
        return False
    if re.match(r"^[\d\-\u2013:°.]+$", word):
        return False
    if not re.search(r"[a-zA-ZåäöæøþðẏÅÄÖÆØ]", word):
        return False
    stripped = re.sub(r"[^a-zA-ZåäöæøþðẏÅÄÖÆØ]", "", word)
    if len(stripped) < 3:
        return False
    return True


def load_tora(path):
    """
    Load TORA place name gazetteer.

    Processing:
    1. Skip first TORA_SKIP_LINES lines (military units)
    2. Filter entries containing modern admin suffixes
    3. For single-word entries: add directly to lookup
    4. For multi-word entries: extract significant words

    Returns:
        tora_exact:   set of lowercase name strings for O(1) exact matching
        tora_fuzzy:   list of lowercase name strings for fuzzy matching
        tora_sources: dict mapping lowercase name -> original TORA entry
        tora_stats:   dict with loading statistics
    """
    raw_entries = []
    filtered_entries = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < TORA_SKIP_LINES:
                continue
            name = line.strip()
            if not name:
                continue

            name_lower = name.lower()
            if any(term in name_lower for term in TORA_FILTER_TERMS):
                filtered_entries.append(name)
                continue

            raw_entries.append(name)

    tora_exact = set()
    tora_sources = {}

    for entry in raw_entries:
        words = entry.split()

        if len(words) == 1:
            lower = entry.lower()
            if _is_valid_tora_word(entry):
                tora_exact.add(lower)
                tora_sources[lower] = entry
        elif len(words) <= TORA_MAX_SPLIT_WORDS:
            for word in words:
                wl = word.lower()
                if (
                    len(word) > 2
                    and wl not in TORA_COMMON_WORDS
                    and wl not in TORA_ADMIN_WORDS
                    and _is_valid_tora_word(word)
                ):
                    if wl not in tora_exact:
                        tora_sources[wl] = entry
                    tora_exact.add(wl)

    tora_fuzzy = sorted(tora_exact)

    stats = {
        "raw_file_entries": len(raw_entries) + len(filtered_entries),
        "after_filtering": len(raw_entries),
        "filtered_out": len(filtered_entries),
        "unique_lookup_names": len(tora_exact),
        "skipped_lines": TORA_SKIP_LINES,
        "filter_terms": TORA_FILTER_TERMS,
    }

    return tora_exact, tora_fuzzy, tora_sources, stats


# ============================================================
# DF — Diplomatarium Fennicum Place Name Gazetteer
# ============================================================


def load_df(path):
    """
    Load Diplomatarium Fennicum place name gazetteer.

    Simple one-name-per-line txt file (already cleaned at conversion).
    Names are merged into the TORA sets by the caller so that Signal 2
    treats them identically to TORA entries.

    Returns:
        df_exact:   set of lowercase name strings
        df_sources: dict mapping lowercase name -> original form
        df_stats:   dict with loading statistics
    """
    df_exact = set()
    df_sources = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or len(name) <= 1:
                continue
            lower = name.lower()
            df_exact.add(lower)
            if lower not in df_sources:
                df_sources[lower] = name

    stats = {
        "unique_names": len(df_exact),
    }

    return df_exact, df_sources, stats


# ============================================================
# SMP — Person Name Gazetteer
# ============================================================


def load_smp(path):
    """
    Load SMP person name gazetteer.

    Processing:
    1. Read CSV with two columns: search_form, retrieved_form
    2. Split comma-separated variants
    3. Clean each variant
    4. Discard entries that are too short (<= 1 char) or empty
    5. Discard pure prefix fragments (end with -)

    Returns:
        smp_exact:   set of lowercase name strings
        smp_fuzzy:   list of lowercase name strings
        smp_sources: dict mapping lowercase name -> original (cleaned) form
        smp_stats:   dict with loading statistics
    """
    names = set()
    sources = {}
    raw_row_count = 0
    discarded_fragments = 0
    discarded_short = 0
    discarded_empty = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            raw_row_count += 1
            for field in row:
                variants = field.split(",")
                for variant in variants:
                    name = _clean_smp_name(variant)

                    if not name:
                        discarded_empty += 1
                        continue
                    if len(name) <= 1:
                        discarded_short += 1
                        continue
                    if name.endswith("-"):
                        discarded_fragments += 1
                        continue

                    lower = name.lower()
                    names.add(lower)
                    if lower not in sources:
                        sources[lower] = name

    smp_exact = names
    smp_fuzzy = sorted(names)

    stats = {
        "raw_rows": raw_row_count,
        "unique_names": len(names),
        "discarded_fragments": discarded_fragments,
        "discarded_short": discarded_short,
        "discarded_empty": discarded_empty,
    }

    return smp_exact, smp_fuzzy, sources, stats


def _clean_smp_name(raw):
    """
    Clean a single SMP name variant.

    Strips: ±, ?, (, ), [, ], leading *, leading ..., extra whitespace.
    Does NOT strip trailing - here (caller decides whether to keep fragments).
    """
    name = raw.strip()
    name = re.sub(r"[±?()\[\]\"]", "", name)
    name = name.lstrip("* ")
    if name.startswith("..."):
        name = name[3:]
    name = re.sub(r"\s+", " ", name).strip()
    return name
