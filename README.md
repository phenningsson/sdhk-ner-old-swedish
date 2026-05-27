# Old Swedish Named Entity Recognition

[![Code License: GPL v3](https://img.shields.io/badge/Code%20License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/phenningsson)

This repository contains the code and data used for an MA thesis at the Linnaeus University focused on Named Entity Recognition (NER) for Old Swedish charter texts. NER is viewed as a token classification task in which the model identifies either [Person] or [Location] entities in Old Swedish digital editions of medieval charters from the *Svenskt Diplomatariums Huvudkartotek* (SDHK), the main catalogue of the Swedish Diplomatarium maintained by Riksarkivet (the Swedish National Archives).

Because no large entity-annotated NER dataset exists for Old Swedish, training data is bootstrapped from SDHK itself using a **three-signal entity projection pipeline** that uses the structure of each SDHK record: every charter has both a modern Swedish summary (regest) and an Old Swedish digital edition (roughly, a transcription). The pipeline runs a modern Swedish NER model on the modern summary, projects the resulting annotation labels back onto the Old Swedish edition by fuzzy matching, and combines that signal with gazetteer lookup (TORA, Diplomatarium Fennicum, Sveriges medeltida personnamn), and a capitalisation extraction heuristic. The three signals combined vote on a final BIO-label per token, producing a silver-standard CoNLL corpus. This corpus, consisting of 230 charters from 1380-1382, was then manually verified, and used to train an initial NER model. This initial NER model was then run on 187 unannotated charters from 1380-1382, and the model's predictions were then manually corrected and used as training data to train the final NER model. The results of the manual verification thus constitute an entity-annotated corpus of 417 Old Swedish charters from which the NER model is trained on. A held-out test set of 75 charters, annotated by domain experts, is used to evaluate both the projection pipeline and the trained NER model. For more information about the pipeline, the training and evaluation data, the annotation process, and the theoretical implications of this project, see the paper *(coming soon)*. The final NER model and the MLM-adapted base model it builds on are available on [HuggingFace](https://huggingface.co/phenningsson).

The best performing NER model on the Old Swedish charter digital editions achieves an entity-level **micro-F1 score of 0.9771** on the internal test set (43 charters held out by the 80/10/10 split on the 1380-1382 corpus), and **F1 = 0.9764** on the external expert test set (75 charters from 1375-1382 annotated independently by domain experts). The base language model is an XLM-RoBERTa-Large model which was first domain-adapted to Old Swedish through continued Masked Language Modelling (MLM) on the full SDHK Old Swedish corpus, and then fine-tuned for token classification on the manually verified training dataset of Old Swedish charters from 1380-1382.

Inter-annotator agreement on the 10 shared adjudication charters is Krippendorff's α = **0.9812** (token-level, IO labels), and mean pairwise entity-level F1 = **0.9672** (exact span + type) across the four annotation groups.

## Repository Structure

```
sdhk-ner-old-swedish/
├── config.py                       # Paths, thresholds, model variants
├── requirements.txt
├── data/
│   ├── gazetteers/                 # TORA, DF, SMP lookup tables (Signal 2)
│   ├── raw/                        # Scraped SDHK charter JSONs
│   ├── 1380_1382_dataset/          # 417 CoNLL files of Old Swedish charters (1380-1382)
│   ├── internal_test_set/          # 43 held-out CoNLL files (80/10/10 split)
│   ├── external_test_set/          # 23 external CoNLL files, not used in the paper for evaluation, but used for use case examples of the model
│   ├── expert_gold_test/           # 75 CoNLL files from the expert annotations
│   └── expert_annotations/
│       ├── pre_adjudication/       # 4 annotator files before adjudication (adjudication is made by the author)
│       └── test_set/               # 4 annotator files after adjudication (creates the expert_gold_test set)
├── src/
│   ├── pipeline.py                 # Three-signal entity projection pipeline orchestration
│   ├── scraping/scrape_sdhk.py     # SDHK scraper
│   ├── preprocessing/              # Text cleaning, tokenisation, gazetteer preparation
│   ├── projection/                 # Signals 1–3 + voting
│   ├── training/                   # CoNLL utilities, splitting
│   └── evaluation/                 # Shared metrics + JSON dump of evaluation
└── scripts/
    ├── run_projection.py           # Run the 3-signal pipeline
    ├── evaluate_pipeline.py        # Evaluate pipeline against domain expert's annotation (expert_gold_test set)
    ├── pretrain_mlm.py             # MLM domain adaptation of the base language model
    ├── train_ner_v2.py             # NER fine-tuning (O-boundary chunking)
    ├── evaluate_ner_v2.py          # NER evaluation
    ├── extract_mlm_corpus.py       # Extract editions from SDHK CSV for MLM pre-training
    ├── extract_internal_test_set.py # Reproduce the 43-file held-out split from the 1380_1382_dataset
    ├── build_test_set.py           # Adjudicate annotator files → test_set/
    ├── convert_test_set_to_conll.py # Annotator .txt → gold CoNLL
    ├── compute_iaa.py              # Krippendorff α + pairwise F1, not used to report final IAA results
    └── compute_iaa_v2.py           # Krippendorff α + pairwise F1 via NLTK AnnotationTask, used to report final IAA results
```

## Get Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/phenningsson/sdhk-ner-old-swedish
cd sdhk-ner-old-swedish
```

2. Install dependencies (a Python 3.10+ virtual environment is recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Quickstart

#### 1. Evaluate the published NER model on the expert test set

Downloads the fine-tuned model from HuggingFace on first run:
```bash
python scripts/evaluate_ner_v2.py
```

#### 2. Reproduce the projection pipeline's CoNLL output

```bash
python scripts/run_projection.py
```

#### 3. Reproduce the internal 80/10/10 split

```bash
python scripts/extract_internal_test_set.py
```

#### 4. Reproduce the expert CoNLL test set

```bash
python scripts/convert_test_set_to_conll.py
```

#### 5. Compute inter-annotator agreement

```bash
python scripts/compute_iaa_v2.py
```

#### 6. Fine-tune NER from a chosen base model (requires GPU)

```bash
python scripts/train_ner_v2.py --model mlm-adapted
```

Available `--model` variants: `ner-swe`, `bert-swe`, `mlm-adapted`,
`xlm-roberta`, `xlm-roberta-large`.

#### 7. MLM domain-adapt a base model on the full SDHK Old Swedish corpus (requires GPU)

```bash
python scripts/extract_mlm_corpus.py --csv sdhk_2411.csv  # download CSV first, available under CC BY 4.0 license from the Swedish National Archives
python scripts/pretrain_mlm.py --base-model xlm-roberta-large
```

## Entity Types

- **Person** — person names
- **Location** — place names

BIO tagging is used: `B-Person`, `I-Person`, `B-Location`, `I-Location`, `O`.

## Data Format

All CoNLL files use the same space-separated, one-token-per-line format, with
a blank line after every sentence-ending punctuation mark:

```
Token    Label
Magnus   B-Person
Eriksson I-Person
gaff     O
Stokholm B-Location
ok       O
Vpsala   B-Location
.        O
         (blank line separates sentences)
```

## Pipeline Overview

The three-signal entity projection pipeline produces BIO labels for each Old Swedish charter digital edition:

- **Signal 1 — NER projection on the modern summary.** A modern Swedish NER
  model (KB-BERT) is run on the modern regest of each charter. Entities
  found are projected onto the Old Swedish edition by progressive
  string matching since the Old Swedish orthography of a name can differ substantially from its modern
  form.

- **Signal 2 — Gazetteer lookup.** Each Old Swedish token is matched
  against TORA + Diplomatarium Fennicum (places) and SMP (persons), using
  exact match plus fuzzy match with a length-ratio. Latin and Old
  Swedish stoplists filter out common words that overlap with gazetteer entries.

- **Signal 3 — Capitalisation extraction heuristic.** Tokens capitalised in
  the edition are inspected together with their left- and right-context
  (locative prepositions, person titles, administrative suffixes, etc) to
  propose Person/Location labels for entities that the first two signals missed.

- **Voting.** A vote function combines the three signals into the final BIO
  label per token, with post-processing for patro- and metronymic suffixes and
  title/preposition stripping.

The pipeline output for the 75-charter expert gold test set reaches an
entity-level micro F1 of almost 0.70; sufficient to bootstrap
training data, but well below what the fine-tuned NER model achieves on
the same test set.

## Reproducing Results

The repository is designed so that a fresh clone (with dependencies
installed) can reproduce every committed data artifact and every reported
metric using the included scripts. The NER weights are not part of this repository, meaning that the
fine-tuned model is hosted on HuggingFace and is downloaded on the first run
of `evaluate_ner_v2.py`. Some scripts need GPUs (or a lot of patience if running on CPU) and external CSV files
(noted in their docstrings).

## Resources

- **Paper:** *(coming soon)*
- **Models:** [HuggingFace Hub — phenningsson](https://huggingface.co/phenningsson)
  - Fine-tuned NER: `phenningsson/sdhk-ner-old-swedish-v2`
  - MLM-adapted base: `phenningsson/sdhk-mlm-pretrained-full`
- **Data sources:** SDHK, TORA, Diplomatarium Fennicum, Sveriges medeltida
  personnamn (see [LICENSE.txt](LICENSE.txt) for full attributions)

## Citation

If you want to reference this work in any way, please cite:

```bibtex
@key{comingsoon}
```

## License

For complete licensing details, attributions, and citations, please see the [LICENSE.txt](LICENSE.txt) file. In short, the **GNU General Public License v3.0 (GPL-3.0)** is used for all the source code (Python scripts), and the **Creative Commons Attribution 4.0 (CC BY 4.0)** for the derived data.

**Note:** Charter texts and gazetteers are derived from external sources (SDHK, TORA, Diplomatarium Fennicum, SMP). Please also respect the licensing terms of those upstream sources when reusing data from this repository.

## Acknowledgments

Sincere gratitude is extended to the excellent resources and their contributors below, which has made it possible for this academic research and development of open source NER models for Old Swedish to be conducted. Special thanks is given to the expert annotators who graciously annotated Person and Location entities in the 75 charters that form the expert gold test set used for evaluation in this project.

- **SDHK — Svenskt Diplomatariums Huvudkartotek:** [Riksarkivet](https://sok.riksarkivet.se/sdhk)
- **TORA — Topografiskt Register:** [Riksarkivet](https://riksarkivet.se/tora)
- **Diplomatarium Fennicum:** [Kansallisarkisto](http://df.narc.fi)
- **Sveriges medeltida personnamn (SMP):** [Institutet för språk och folkminnen](https://smp.isof.se/)
- **KB-BERT Swedish models:** [KBLab](https://huggingface.co/KBLab)
- **XLM-RoBERTa-large:** [Conneau et al. 2020](https://aclanthology.org/2020.acl-main.747)

## Contact

For questions or issues, please open a GitHub issue or contact:
[phenningsson@me.com]
