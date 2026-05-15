"""
Text cleaning, tokenization, CoNLL parsing, and shared helpers.

Provides the foundational utilities used by all pipeline stages.
"""

import json
import re

from config import MIDDLE_DOT


# ============================================================
# DATA LOADING
# ============================================================

def load_charter_data(path):
    """
    Load charter data from JSON.

    Returns list of dicts, each with keys including Id, Summary, Edition.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ============================================================
# EDITION TOKENIZATION
# ============================================================

def clean_edition_text(text):
    """
    Pre-process edition text before tokenization.

    - Retains middle dots (‧ U+2027) as standalone tokens — they serve as
      editorial separators (like commas) in SDHK witness lists and are
      useful punctuation context for NER. tokenize_edition() handles them
      automatically via the [^\w\s] branch (non-word, non-space character).
    - Replaces degree-like abbreviation markers (º U+00BA, ° U+00B0) with
      spaces — these appear only in Roman numeral date components (e.g.
      "Mºcccºlxxº" -> "M ccc lxx") and carry no linguistic value for NER.
      Using space (not empty string) preserves word boundaries to match
      gold-standard tokenization
    - Collapses multiple spaces into single spaces
    - Strips leading/trailing whitespace
    """
    text = text.replace("\u00ba", " ")  # º (masculine ordinal indicator)
    text = text.replace("\u00b0", " ")  # ° (degree sign)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def tokenize_edition(text):
    """
    Tokenize a (cleaned) edition text into tokens with character positions.

    Each token is a dict:
        {
            'idx':     int,   # sequential token index
            'text':    str,   # the token string
            'start':   int,   # character start position in the cleaned text
            'end':     int,   # character end position in the cleaned text
            'is_word': bool,  # True if the token contains word characters
        }

    Punctuation is split from words so that e.g. "Siggadottir." becomes
    two tokens: "Siggadottir" and ".".

    Handles Old Swedish characters: æ, ø, þ, ð, ẏ, and combining diacritics.
    """
    tokens = []
    idx = 0

    for ws_match in re.finditer(r"\S+", text):
        chunk = ws_match.group()
        chunk_start = ws_match.start()

        for sub_match in re.finditer(r"[\w\u0300-\u036f]+|[^\w\s]", chunk):
            sub_text = sub_match.group()
            sub_offset = sub_match.start()
            char_start = chunk_start + sub_offset
            char_end = char_start + len(sub_text)

            tokens.append(
                {
                    "idx": idx,
                    "text": sub_text,
                    "start": char_start,
                    "end": char_end,
                    "is_word": bool(re.match(r"\w", sub_text)),
                }
            )
            idx += 1

    return tokens


# ============================================================
# CONLL PARSING
# ============================================================

def parse_gold_conll(conll_path):
    """
    Parse a gold-standard CoNLL file (BIO-tagged, one token per line).

    Returns two things:
        tokens_with_labels: list of (token_text, bio_label) tuples
        entities: list of entity dicts:
            {
                'text':   str,   # full entity string, e.g. "Bo Jonsson"
                'label':  str,   # entity type, e.g. "Person"
                'tokens': list,  # list of individual token strings
            }
    """
    tokens_with_labels = []
    entities = []
    current_entity_label = None
    current_entity_tokens = []

    with open(conll_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_entity_label:
                    entities.append(
                        {
                            "text": " ".join(current_entity_tokens),
                            "label": current_entity_label,
                            "tokens": list(current_entity_tokens),
                        }
                    )
                    current_entity_label = None
                    current_entity_tokens = []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            token = parts[0]
            tag = parts[-1]

            tokens_with_labels.append((token, tag))

            if tag.startswith("B-"):
                if current_entity_label:
                    entities.append(
                        {
                            "text": " ".join(current_entity_tokens),
                            "label": current_entity_label,
                            "tokens": list(current_entity_tokens),
                        }
                    )
                current_entity_label = tag[2:]
                current_entity_tokens = [token]

            elif tag.startswith("I-") and current_entity_label:
                current_entity_tokens.append(token)

            else:
                if current_entity_label:
                    entities.append(
                        {
                            "text": " ".join(current_entity_tokens),
                            "label": current_entity_label,
                            "tokens": list(current_entity_tokens),
                        }
                    )
                    current_entity_label = None
                    current_entity_tokens = []

    if current_entity_label:
        entities.append(
            {
                "text": " ".join(current_entity_tokens),
                "label": current_entity_label,
                "tokens": list(current_entity_tokens),
            }
        )

    return tokens_with_labels, entities


# ============================================================
# HELPERS
# ============================================================

def consonant_skeleton(text):
    """
    Extract consonant skeleton from text (strip vowels).
    Used for progressive fuzzy matching of Old Swedish name variants.

    Example: "Nilsson" -> "nlssn", "Niclison" -> "nclsn"
    """
    vowels = set("aeiouyåäöæøẏ")
    return "".join(c for c in text.lower() if c.isalpha() and c not in vowels)


def is_sentence_initial(tokens, token_idx):
    """
    Check if the token at token_idx is likely sentence-initial.

    Returns True if:
    - It is the first word token in the text, OR
    - The nearest preceding word token is followed by a sentence-ending
      punctuation mark (. ! ?)

    This is a rough heuristic — Old Swedish texts do not have consistent
    sentence boundaries.
    """
    if token_idx == 0:
        return True

    for i in range(token_idx - 1, -1, -1):
        tok = tokens[i]
        if tok["is_word"]:
            return False
        if tok["text"] in ".!?":
            return True

    return True
