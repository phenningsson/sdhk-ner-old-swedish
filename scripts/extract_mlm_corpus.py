#!/usr/bin/env python3
"""
Extract Old Swedish edition texts from the full SDHK CSV for MLM pre-training.

Applies the same text cleaning as the scraper's clean_transcription() to ensure
consistency between the scraped JSON editions and the CSV editions.

Usage:
    .venv/bin/python3 scripts/extract_mlm_corpus.py
    .venv/bin/python3 scripts/extract_mlm_corpus.py --csv sdhk_2411.csv --output data/raw/sdhk_all_swedish.json
    .venv/bin/python3 scripts/extract_mlm_corpus.py --min-words 20 --latin-threshold 0.5
"""

import argparse
import csv
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ── Text cleaning (mirrors scrape_sdhk.py::clean_transcription) ──────────

def clean_edition_text(text):
    """Clean a raw CSV edition text using the same rules as the scraper.

    The CSV editions haven't been through HTML cleaning (they're plain text
    from the database export), but they still contain:
    - Parenthesised editorial notes/footnote references
    - Square brackets around reconstructed letters
    - º and ° from Roman numeral formatting
    - Extra whitespace
    - Various invisible/special Unicode characters
    """
    # Remove parentheses WITH contents (editorial notes, footnote refs)
    text = re.sub(r'\([^)]*\)', '', text)

    # Remove bracket CHARACTERS but keep contents (reconstructed letters)
    text = text.replace('[', '')
    text = text.replace(']', '')

    # Remove º and ° (Roman numeral markers)
    text = text.replace('\u00ba', ' ')  # º masculine ordinal
    text = text.replace('\u00b0', ' ')  # ° degree sign

    # Remove invisible/special characters
    for ch in ['\u00ad', '\u200b', '\u200c', '\u200d', '\ufeff']:
        text = text.replace(ch, '')

    # Normalize special spaces → regular space
    for ch in ['\u2006', '\u2007', '\u2008', '\u2009', '\u200a', '\u00a0']:
        text = text.replace(ch, ' ')

    # Remove Private Use Area characters
    text = re.sub(r'[\uE000-\uF8FF]', '', text)

    # The CSV export lost both middle dots (U+2027) and ẏ (U+1E8F) → literal '?'
    # We cannot reliably distinguish them, so leave ? as-is.
    # The MLM tokenizer handles ? fine as a regular token.

    # Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Latin detection ───────────────────────────────────────────────────────

LATIN_MARKERS = {
    "et", "in", "de", "ad", "cum", "per", "pro", "quod", "qui", "que",
    "est", "sunt", "nos", "ego", "item", "anno", "domini", "ecclesie",
    "omnibus", "presentes", "litteras", "datum", "nostrum", "sigillum",
    "terre", "dominus", "rex", "regni", "noverint", "universi", "super",
    "sub", "vel", "ac", "sicut", "ita", "dicti", "dictus", "predicti",
    "videlicet", "necnon", "testibus", "sigillis", "presentibus",
}


def latin_fraction(text):
    words = text.lower().split()
    if not words:
        return 1.0
    return sum(1 for w in words if w.rstrip(".,;:") in LATIN_MARKERS) / len(words)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract Old Swedish editions from SDHK CSV for MLM training"
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(PROJECT_ROOT, "sdhk_2411.csv"),
        help="Path to SDHK CSV file",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "data", "raw", "sdhk_all_swedish.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--min-words", type=int, default=20,
        help="Minimum word count after cleaning (default: 20)",
    )
    parser.add_argument(
        "--latin-threshold", type=float, default=0.5,
        help="Max Latin word fraction to include (default: 0.5)",
    )
    parser.add_argument(
        "--encoding", default="latin-1",
        help="CSV file encoding (default: latin-1)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SDHK Edition Extraction for MLM")
    print("=" * 60)
    print(f"  CSV:             {args.csv}")
    print(f"  Output:          {args.output}")
    print(f"  Min words:       {args.min_words}")
    print(f"  Latin threshold: {args.latin_threshold}")
    print()

    stats = {
        "total": 0,
        "no_edition": 0,
        "not_swedish": 0,
        "too_short": 0,
        "too_latin": 0,
        "accepted": 0,
    }

    charters = []

    with open(args.csv, encoding=args.encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            stats["total"] += 1
            cid = row.get("Id", "").strip()
            lang = row.get("Lang", "").strip().lower()
            edition = row.get("Edition", "").strip()

            if not edition:
                stats["no_edition"] += 1
                continue
            if "svenska" not in lang and "swedish" not in lang:
                stats["not_swedish"] += 1
                continue

            edition = clean_edition_text(edition)
            words = edition.split()

            if len(words) < args.min_words:
                stats["too_short"] += 1
                continue
            if latin_fraction(edition) > args.latin_threshold:
                stats["too_latin"] += 1
                continue

            stats["accepted"] += 1
            charters.append({
                "Id": cid,
                "Date": row.get("Date", "").strip(),
                "Language": lang,
                "Edition": edition,
                "source": "sdhk_csv",
            })

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(charters, f, ensure_ascii=False, indent=2)

    total_words = sum(len(c["Edition"].split()) for c in charters)
    total_chars = sum(len(c["Edition"]) for c in charters)

    print(f"  Total rows:       {stats['total']:,}")
    print(f"  No edition:       {stats['no_edition']:,}")
    print(f"  Not Swedish:      {stats['not_swedish']:,}")
    print(f"  Too short (<{args.min_words}w): {stats['too_short']:,}")
    print(f"  Too Latin:        {stats['too_latin']:,}")
    print(f"  Accepted:         {stats['accepted']:,}")
    print()
    print(f"  Total words:      {total_words:,}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Saved to:         {args.output}")


if __name__ == "__main__":
    main()
