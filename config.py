"""
Central configuration for the Old Swedish NER pipeline.

All thresholds, stoplists, file paths, and model settings live here
so that every module imports from a single source of truth.
"""

import os
import re

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# PATHS
# ============================================================

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sdhk_1380_1382.json")
PILOT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "pilot_20_scraped.json")
TORA_PATH = os.path.join(PROJECT_ROOT, "data", "gazetteers", "tora_gazetteer.txt")
DF_PATH = os.path.join(PROJECT_ROOT, "data", "gazetteers", "df_gazetteer.txt")
SMP_PATH = os.path.join(PROJECT_ROOT, "data", "gazetteers", "smp3.csv")
GOLD_TRANSC_DIR = os.path.join(PROJECT_ROOT, "data", "gold", "transc")
GOLD_MODERN_DIR = os.path.join(PROJECT_ROOT, "data", "gold", "modern")
SILVER_DIR = os.path.join(PROJECT_ROOT, "data", "silver")
SILVER_V2_DIR = os.path.join(PROJECT_ROOT, "data", "silver_v2")
VERIFIED_SILVER_DIR = os.path.join(PROJECT_ROOT, "data", "verified_silver")
EXPERT_GOLD_DIR = os.path.join(PROJECT_ROOT, "data", "expert_gold", "annotated")
EXPERT_GOLD_PRE_DIR = os.path.join(PROJECT_ROOT, "data", "expert_gold", "pre_annotated")
EXPERT_GOLD_SELECTION = os.path.join(PROJECT_ROOT, "data", "expert_gold", "selected_charters.json")

# ============================================================
# ENTITY TYPES
# ============================================================

ENTITY_TYPES = ("Person", "Location")

# ============================================================
# UNICODE CONSTANTS
# ============================================================

MIDDLE_DOT = "\u2027"  # ‧  — editorial separator in SDHK editions

# ============================================================
# NER MODEL (Signal 1)
# ============================================================

NER_MODEL_NAME = "KBLab/bert-base-swedish-cased-ner"
NER_AGGREGATION = "first"

# ============================================================
# SIGNAL 1 — NER PROJECTION THRESHOLDS
# ============================================================

S1_FUZZY_THRESHOLD = 60       # Levenshtein threshold for progressive match
S1_CONSONANT_THRESHOLD = 80   # Consonant-skeleton match threshold
S1_CONSONANT_MIN_LEN = 3      # Min consonant skeleton length

# ============================================================
# SIGNAL 2 — GAZETTEER THRESHOLDS
# ============================================================

S2_FUZZY_THRESHOLD = 80       # Fuzzy match score cutoff (0-100)
S2_FUZZY_MIN_LENGTH = 4       # Min token length for fuzzy matching
S2_LENGTH_RATIO = 0.7         # Min length ratio token/match

# ============================================================
# GAZETTEER LOADING (prepare_gazetteers)
# ============================================================

TORA_SKIP_LINES = 32          # Header lines to skip in TORA file
TORA_MAX_SPLIT_WORDS = 4      # Max words before entry is kept whole only

TORA_FILTER_TERMS = ["församling", "kommun", "pastorat"]

TORA_COMMON_WORDS = {
    # Swedish prepositions / articles / conjunctions
    "i", "och", "oc", "ok", "av", "af", "på", "vid", "den", "det", "de",
    "med", "medh", "til", "till", "for", "fore", "från", "som", "the",
    "att", "eller", "alla", "alle", "hans", "hen", "sin", "sitt",
    "under", "over", "övre", "nedre", "norra", "södra", "östra", "västra",
    "stora", "lilla", "gamla", "nya", "s:t", "s:ta",
    # Common Latin words appearing in TORA document titles
    "anno", "domini", "die", "post", "ante", "cum", "sub",
    "sancti", "sancta", "sancte", "beati", "beate",
    "in", "et", "ad", "per", "pro",
}

TORA_ADMIN_WORDS = {
    "härad", "socken", "sokn", "köping", "stad",
    "sätesgård", "gård", "kyrka", "kloster",
}

# ============================================================
# SIGNAL 2 — STOPLISTS
# ============================================================

# Common Old Swedish words that exist in SMP/TORA as names but appear
# far more frequently as ordinary words in charter texts.
S2_SWEDISH_STOPLIST = {
    # Pronouns / demonstratives / determiners
    "hans", "hennes", "sin", "sitt", "sina",
    "thet", "thæt", "them", "then", "the", "thetta",
    "alla", "alle", "allom", "allum", "alt",
    "minna", "mina", "minne", "när", "Gørum", "gørum",
    # Common nouns that are also place/person names in gazetteers
    "mark", "jord", "land", "fasta", "sten", "eng",
    "dagh", "dag",
    "manne", "man",
    "rike",
    # Common nouns extracted from multi-word TORA entries
    "akra", "gardh", "gaard", "gard",
    "byn", "by",
    "hus", "husit",
    "bol", "bole",
    "aker",
    "borg",
    "holm",
    "vik",
    "dal",
    "bro",
    "eke",
    "ask",
    "alm", "nor",
    # Prepositions / titles / formulaic words
    "til", "till",
    "herra", "herrä", "herræ", "hærra", "herre",
    "kennis", "kænnis", "kænnas", "kennes",
    "køpp", "køp", "köp",
    # Religious / divine (not person entities in context)
    "gudh", "gudi", "gudhii", "gwz",
    # Verbs / adjectives that match SMP entries
    "ægher", "ægha",
    "somar", "sumar",
    "skal", "skall",
    "kan",
    "vita", "vitha",
    "war", "vara",
    # Common function words
    "halla", "halle",
    "dom", "Giör", "giör",
}

# Latin formulaic words — dating clauses, witness lists, colophons.
S2_LATIN_STOPLIST = {
    # Dating formulae
    "datum", "anno", "domini", "dominj", "domino",
    "feria", "kalendas", "octaua", "crastino",
    "scriptum", "item", "jtem",
    "ante", "post", "cum", "sub", "per", "pro",
    # Religious / documentary
    "sancti", "sancto", "sancta", "sancte",
    "beate", "beati", "beatae",
    "christi", "christo",
    "dei", "deo", "deum",
    "testamentum", "testimonium",
    "sigillo", "sigillis", "sigillum",
    "virginis", "virgo",
    # Calendar/liturgical terms (NOT saint names — those are kept)
    "pasche", "pascha",
    "assumpcionis", "ascensionis",
    "exaltacionis", "exultacionis",
    "purificationem", "purificacionem",
    "natiuitatem", "natiuitatis", "nativitate",
    "omnium",
    # Other Latin words
    "augusti", "septembris", "octobris", "novembris", "decembris",
    "januarii", "februarii", "martii", "aprilis", "maii", "junii", "julii",
    "latinam", "latine",
    "ecclesie", "ecclesia",
    "prepositi", "curati",
}

# Roman numeral pattern (shared by Signal 2 and Signal 3)
ROMAN_NUMERAL_RE = re.compile(
    r"^[ivxlcdmIVXLCDM]+"
    r"(?:primo|secundo|tercio|quarto|quinto|sexto|septimo|octauo|nono)?"
    r"$",
    re.IGNORECASE,
)

# ============================================================
# SIGNAL 3 — STOPLISTS
# ============================================================

# Latin words commonly capitalised in SDHK edition texts.
S3_LATIN_STOPLIST = {
    # Dating / temporal
    "Datum", "Anno", "Domini", "Dominj", "Domino",
    "Mº", "M", "Mccclxxviijº", "Mccclxxxprimo",
    "CCC", "IIII", "III", "II",
    "Feria", "Kalendas", "Octaua", "Crastino",
    "Scriptum", "Item", "Jtem",
    # Religious (NOT personal names like Marie/Johannis — those are kept)
    "Sancti", "Sancto", "Sancta", "Sancte",
    "Beate", "Beati", "Beatae",
    "Virginis",
    "Christi", "Christo",
    "Dei", "Deo", "Deum",
    # Legal / documentary
    "Testamentum", "Testimonium",
    "Sigillo", "Sigillis", "Sigillum",
    "Jn", "In", "Et", "Sub", "Cum", "Pro", "Per", "Post", "Ante",
    # Calendar/liturgical terms (NOT saint names — those are kept)
    "Pasche", "Pascha",
    "Assumpcionis", "Ascensionis",
    "Exaltacionis", "Exultacionis",
    "Purificationem", "Purificacionem",
    "Natiuitatem", "Natiuitatis", "Nativitate",
    "Omnium",
}

# Swedish function words / formulaic terms that sometimes appear capitalised.
S3_SWEDISH_STOPLIST = {
    # Conjunctions / adverbs / prepositions
    "Ok", "Oc", "Och",
    "Til", "Till",
    "Fore", "For", "Ffor", "Ffore",
    "Tha", "Thy", "Thẏ",
    "Swa", "Sua", "Sva",
    "Medh", "Mædh",
    "Aff", "Af",
    "Wm", "Um", "Vm",
    "Som",
    "Ther", "Thær",
    # Demonstratives / pronouns
    "The", "Then", "Them", "Thom", "Thøm", "Thet", "Thæt",
    # Opening formulae
    "Alla", "Allom", "Allum", "Alle", "Alt",
    "Kennis", "Kænnis", "Kænnas", "Kennes",
    "Kungør", "Kunnoghom",
    "Giör", "Gørom", "Gørum", "Gør", "Giør", "Göör",
    # Pronouns
    "Wi", "Wj",
    "Jak", "Jac", "Iak", "Jäk", "Jæk",
    # Misc capitalised function words
    "Kan", "Skal",
    "Huilkin", "Huilka", "Hwilkin",
    "Framledhis", "Framplethis", "Framledis",
    # Religious / title (not entities themselves)
    "Gudhi", "Gudh", "Gudi", "Gudhii", "Gwz",
    "Herra", "Herrä", "Herræ",
}

# ============================================================
# SIGNAL 3 — TYPE HEURISTIC WORD LISTS
# ============================================================

# Locative prepositions: preceding token → heuristic for Location type
S3_LOCATIVE_PREPOSITIONS = {
    "i", "j", "a", "aa", "ij", "jj",
    "jnnan", "innan", "wider", "widher",
    "naar", "nær", "aff", "af",
    "ii",
}

# Title words: preceding token → heuristic for Person type
S3_TITLE_WORDS = {
    "herra", "herrä", "herræ", "hærra",
    "herre", "her",
}

# Administrative suffixes: following token → heuristic for Location type
S3_ADMIN_SUFFIXES = {
    "sokn", "sokne", "socken", "hæradhe", "hæradh",
    "härad", "häradhe",
}

# ============================================================
# VOTING — POST-PROCESSING
# ============================================================

# Patronymic suffixes — tokens ending with these continue a Person span
PATRONYMIC_RE = re.compile(
    r"(?:s?son|s?sons|dott[eä]r|dottir|dottor|dottærs?)$",
    re.IGNORECASE,
)

# Title words stripped from entity span starts
TITLE_WORDS = {
    "herra", "herrä", "herræ", "hærra", "herre", "her",
    "härrä", "härra",
    "fru", "frw",
    "herr",
}

# Prepositions stripped from Location span starts
LOCATION_PREFIX_STOP = {
    "til", "till", "i", "j", "a", "aa",
    "af", "aff",
    "jnnan", "innan",
}

# ============================================================
# EVALUATION
# ============================================================

GOLD_EXCLUDED = {"9752", "10413"}

# ============================================================
# TRAINING — NER FINE-TUNING
# ============================================================

TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "1380_1382_dataset")
EXTERNAL_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "external_test_set")
NER_FINETUNED_DIR = os.path.join(PROJECT_ROOT, "models", "ner_finetuned")
MLM_PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "models", "mlm_pretrained_v3")

TRAIN_SEED = 42

LABEL_LIST = ["O", "B-Person", "I-Person", "B-Location", "I-Location"]

# Model variants for NER experiments
NER_MODEL_VARIANTS = {
    "ner-swe": "KBLab/bert-base-swedish-cased-ner",
    "bert-swe": "KBLab/bert-base-swedish-cased",
    "mlm-adapted": os.path.join(PROJECT_ROOT, "models", "mlm_pretrained_v3"),
    "xlm-roberta": "xlm-roberta-base",
    "xlm-roberta-large": "xlm-roberta-large",
}

# Raw JSON files containing edition texts for MLM pre-training
# Full SDHK Old Swedish corpus (extracted from sdhk_2411.csv)
MLM_RAW_JSON_FILES = [
    os.path.join(PROJECT_ROOT, "data", "raw", "sdhk_all_swedish_scraped.json"),
]
# Original 1375-1382 subset (for reference / smaller runs)
MLM_RAW_JSON_FILES_SMALL = [
    os.path.join(PROJECT_ROOT, "data", "raw", "sdhk_1375_1376.json"),
    os.path.join(PROJECT_ROOT, "data", "raw", "sdhk_1377_1379.json"),
    os.path.join(PROJECT_ROOT, "data", "raw", "sdhk_1380_1382.json"),
]

# ============================================================
# TRAINING — HYPERPARAMETER DEFAULTS
# ============================================================

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MAX_LENGTH = 256
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_EARLY_STOPPING_PATIENCE = 3

# Class weights for imbalanced NER data
USE_CLASS_WEIGHTS = True
O_CLASS_WEIGHT = 1.0
ENTITY_CLASS_WEIGHT = 5.0

# Evaluation frequency: evaluate every X% of training
EVAL_PERCENT = 10

# MLM defaults (aligned with Old Icelandic MLM settings)
DEFAULT_MLM_EPOCHS = 8
DEFAULT_MLM_BATCH_SIZE = 2
DEFAULT_MLM_GRADIENT_ACCUMULATION = 16  # effective batch = 32
DEFAULT_MLM_LEARNING_RATE = 3e-5
DEFAULT_MLM_PROBABILITY = 0.15
DEFAULT_MLM_MAX_LENGTH = 256            # reduced for xlm-roberta-large on 16GB GPU
DEFAULT_MLM_WARMUP_RATIO = 0.06
DEFAULT_MLM_EVAL_STEPS = 300
DEFAULT_MLM_MIN_SEQ_LENGTH = 7          # filter short sequences (prevents NaN)
