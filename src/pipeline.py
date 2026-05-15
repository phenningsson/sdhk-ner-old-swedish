"""
Pipeline orchestrator: runs all three signals on charters and combines
via voting to produce silver-standard CoNLL annotations.
"""

import os

from src.preprocessing.clean_text import (
    load_charter_data,
    clean_edition_text,
    tokenize_edition,
)
from src.preprocessing.prepare_gazetteers import load_tora, load_smp
from src.projection.signal_1_ner import run_ner_on_summaries, project_ner_to_edition
from src.projection.signal_2_gazetteer import signal_2_gazetteer
from src.projection.signal_3_capitalisation import signal_3_capitalisation
from src.projection.voting import combine_signals, labels_to_conll, extract_entity_spans


def run_pipeline(charters, tora_data, smp_data, ner_results, output_dir=None):
    """
    Run the full three-signal pipeline on a list of charters.

    Args:
        charters: list of charter dicts (Id, Summary, Edition)
        tora_data: tuple (tora_exact, tora_fuzzy, tora_sources)
        smp_data: tuple (smp_exact, smp_fuzzy, smp_sources)
        ner_results: dict from run_ner_on_summaries
        output_dir: optional directory for CoNLL output files

    Returns:
        dict mapping charter_id -> pipeline result dict
    """
    tora_exact, tora_fuzzy, tora_sources = tora_data
    smp_exact, smp_fuzzy, smp_sources = smp_data

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pipeline_results = {}

    for charter in charters:
        charter_id = str(charter["Id"])
        cleaned = clean_edition_text(charter["Edition"])
        tokens = tokenize_edition(cleaned)

        # Signal 1: NER projection
        ner_data = ner_results.get(charter_id, {})
        s1_ann, s1_log = project_ner_to_edition(
            ner_data.get("entities", []), tokens
        )

        # Signal 2: Gazetteer
        s2_ann, s2_log = signal_2_gazetteer(
            tokens, tora_exact, tora_fuzzy, tora_sources,
            smp_exact, smp_fuzzy, smp_sources
        )

        # Signal 3: Capitalisation
        s3_ann, s3_log = signal_3_capitalisation(tokens)

        # Voting
        final_labels, vote_log = combine_signals(tokens, s1_ann, s2_ann, s3_ann)

        # Extract entities and CoNLL lines
        entities = extract_entity_spans(tokens, final_labels)
        conll_lines = labels_to_conll(tokens, final_labels)

        # Write CoNLL file
        conll_path = None
        if output_dir:
            conll_path = os.path.join(output_dir, f"sdhk_{charter_id}_silver.conll")
            with open(conll_path, "w", encoding="utf-8") as f:
                f.write("\n".join(conll_lines) + "\n")

        # Store results
        pipeline_results[charter_id] = {
            "charter_id": charter_id,
            "n_tokens": len(tokens),
            "n_word_tokens": sum(1 for t in tokens if t["is_word"]),
            "signal_1": {
                "n_annotations": len(s1_ann),
                "projected": s1_log["projected_count"],
                "unmatched": s1_log["unmatched_count"],
            },
            "signal_2": {
                "n_annotations": len(s2_ann),
                "matches": s2_log["match_count"],
                "stoplisted": s2_log["stoplisted_count"],
            },
            "signal_3": {
                "n_annotations": len(s3_ann),
                "candidates": s3_log["candidate_count"],
                "stoplisted": s3_log["stoplisted_count"],
                "sentence_initial": s3_log["sentence_initial_count"],
            },
            "voting": {
                "entity_tokens": vote_log["entity_tokens"],
                "person_tokens": vote_log["person_tokens"],
                "location_tokens": vote_log["location_tokens"],
                "type_disagreements": vote_log["type_disagreements"],
                "cap_only_unknown": vote_log["cap_only_unknown"],
                "signal_combinations": vote_log["signal_combinations"],
            },
            "entities": [
                {"text": e["text"], "label": e["label"]}
                for e in entities
            ],
            "n_entities": len(entities),
            "n_person_entities": sum(1 for e in entities if e["label"] == "Person"),
            "n_location_entities": sum(1 for e in entities if e["label"] == "Location"),
            "conll_path": conll_path,
            "signal_1_log": s1_log,
            "signal_2_log": s2_log,
            "signal_3_log": s3_log,
            "voting_log": vote_log,
        }

        per_count = sum(1 for e in entities if e["label"] == "Person")
        loc_count = sum(1 for e in entities if e["label"] == "Location")
        print(
            f"  Charter {charter_id}: "
            f"{len(entities)} entities ({per_count} PER, {loc_count} LOC), "
            f"signals: S1={len(s1_ann)} S2={len(s2_ann)} S3={len(s3_ann)}"
        )

    return pipeline_results
