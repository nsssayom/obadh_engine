# Obadh Autocorrect Data Policy

This directory stores auditable autocorrect lexicon sources and the compact
runtime FST artifact. It does not store raw corpora or ML training scratch data.

The deterministic Obadh transliterator is still dictionary-free. Autocorrect is
a product layer above it: Obadh produces a Bengali baseline, then the
autocorrect layer retrieves valid lexicon candidates and ranks them without
changing the core transliteration contract.

## Directory Contract

```text
data/autocorrect/
  README.md
  lexicons/
    raw/
      epub_bn.tsv
      wiki_bn.tsv
      news_bn.tsv
    curated/
      epub_bn.tsv
      wiki_bn.tsv
      news_bn.tsv
    loanwords/
      en_bn_loanwords.tsv
    derived/
      loan_bn.tsv
    merged/
      bn.tsv
  models/
    bn.fst
    en_bn_loanwords.fst
```

- `lexicons/raw/epub_bn.tsv`: word-frequency TSV extracted from locally curated
  EPUB books.
- `lexicons/raw/wiki_bn.tsv`: word-frequency TSV extracted from the accepted
  Bangla Wikipedia source.
- `lexicons/raw/news_bn.tsv`: word-frequency TSV extracted from the accepted
  Furcifer Bangla newspaper source.
- `lexicons/curated/epub_bn.tsv`: curated EPUB lexicon after low-frequency
  semantic/noise filtering.
- `lexicons/curated/wiki_bn.tsv`: curated Wikipedia lexicon after
  low-frequency semantic/noise filtering.
- `lexicons/curated/news_bn.tsv`: curated newspaper lexicon after
  low-frequency semantic/noise filtering.
- `lexicons/loanwords/en_bn_loanwords.tsv`: curated English spelling to Bangla
  loanword TSV for words such as `university`, `engineering`, and `train`.
- `lexicons/derived/loan_bn.tsv`: generated Bangla-only TSV exported from the
  curated loanword source so those Bangla surfaces also exist in the main
  lexicon.
- `lexicons/merged/bn.tsv`: unified, auditable Bangla word-frequency TSV.
- `models/bn.fst`: production finite-state lexicon built from
  `lexicons/merged/bn.tsv`.
- `models/en_bn_loanwords.fst`: compact English-key FST for exact and bounded
  fuzzy loanword correction.

Raw corpus inputs stay outside this directory. The local `epubs/` directory,
Kaggle caches, downloaded JSON, temporary candidate dumps, and training exports
are not repo artifacts. Use ignored scratch directories such as:

```text
data/autocorrect/tmp/
data/autocorrect/candidates/
data/autocorrect/training/
```

## Runtime Artifact Contract

Runtime autocorrect loads `models/bn.fst` and, when available,
`models/en_bn_loanwords.fst`; it does not parse CSV, raw corpora, or large TSV
structures on the keyboard hot path. The Bangla FST maps each NFC-normalized
Bangla word to its unigram frequency. The loanword FST maps lowercase ASCII
English keys to one or more curated Bangla spellings; runtime loanword queries
are ASCII case-normalized before lookup.

Native tools memory-map the FST so the lexicon is not expanded into a
heap-resident trie. WASM loads the same compact byte blob and queries it in
place. This is the production path for the playground and keyboard-time lookup.

Input TSV format:

```text
বাংলা_শব্দ<TAB>frequency
```

The frequency column is optional and defaults to `1`. TSV ingestion normalizes
words to NFC and validates Bangla-only surfaces before audit, merge, and artifact
builds.

## Build Path

Extract a source TSV:

```bash
cargo run --bin obadh-autocorrect -- extract-lexicon \
  --input epubs \
  --output data/autocorrect/lexicons/raw/epub_bn.tsv \
  --min-frequency 1
```

Audit a source TSV:

```bash
cargo run --bin obadh-autocorrect -- audit-lexicon \
  --input data/autocorrect/lexicons/raw/epub_bn.tsv --pretty
```

Extract the accepted newspaper source:

```bash
python3 tools/autocorrect/extract_news_json_lexicon.py \
  --input ~/.cache/kagglehub/datasets/furcifer/bangla-newspaper-dataset/versions/2/data_v2/data_v2.json \
  --output data/autocorrect/lexicons/raw/news_bn.tsv
```

Generate derived loanword surfaces, curate noisy sources, and merge accepted
TSV sources:

```bash
cargo run --bin obadh-autocorrect -- export-loanword-bangla-lexicon \
  --input data/autocorrect/lexicons/loanwords/en_bn_loanwords.tsv \
  --output data/autocorrect/lexicons/derived/loan_bn.tsv \
  --frequency 16

python3 tools/autocorrect/curate_lexicon_sources.py \
  --epub data/autocorrect/lexicons/raw/epub_bn.tsv \
  --wiki data/autocorrect/lexicons/raw/wiki_bn.tsv \
  --news data/autocorrect/lexicons/raw/news_bn.tsv \
  --loan data/autocorrect/lexicons/derived/loan_bn.tsv \
  --epub-output data/autocorrect/lexicons/curated/epub_bn.tsv \
  --wiki-output data/autocorrect/lexicons/curated/wiki_bn.tsv \
  --news-output data/autocorrect/lexicons/curated/news_bn.tsv \
  --quarantine-output data/autocorrect/tmp/curation_quarantine.tsv

cargo run --bin obadh-autocorrect -- merge-lexicon \
  --input data/autocorrect/lexicons/curated/epub_bn.tsv \
  --input data/autocorrect/lexicons/curated/wiki_bn.tsv \
  --input data/autocorrect/lexicons/curated/news_bn.tsv \
  --input data/autocorrect/lexicons/derived/loan_bn.tsv \
  --output data/autocorrect/lexicons/merged/bn.tsv \
  --drop-invalid
```

Build and inspect the runtime FST:

```bash
cargo run --bin obadh-autocorrect -- build-fst-lexicon \
  --input data/autocorrect/lexicons/merged/bn.tsv \
  --output data/autocorrect/models/bn.fst

cargo run --bin obadh-autocorrect -- build-loanword-lexicon \
  --input data/autocorrect/lexicons/loanwords/en_bn_loanwords.tsv \
  --output data/autocorrect/models/en_bn_loanwords.fst \
  --frequency 16

cargo run --bin obadh-autocorrect -- inspect-fst-lexicon \
  --input data/autocorrect/models/bn.fst --pretty

cargo run --bin obadh-autocorrect -- inspect-loanword-lexicon \
  --input data/autocorrect/models/en_bn_loanwords.fst --pretty
```

Probe the runtime path:

```bash
cargo run --bin obadh-autocorrect -- suggest-fst \
  --lexicon data/autocorrect/models/bn.fst \
  --loanwords data/autocorrect/models/en_bn_loanwords.fst \
  --input cad \
  --max-distance 2 \
  --max-candidates 512 \
  --max-prefix-candidates 24 \
  --response-candidates 12 \
  --pretty
```

Expected behavior for `cad` is that Obadh's exact baseline `চাদ` remains visible
and the FST-backed mark candidate `চাঁদ` is surfaced without scanning the
lexicon or generating global spelling mutations.

## Candidate Generation Contract

The runtime FST path is deliberately bounded:

- Exact baseline lookup if Obadh's output is already a lexicon word.
- A tiny Obadh-aware Roman repair beam that inserts missing lowercase `o`
  separators at tokenizer-detected conjunct boundaries, including repeated
  clusters such as `khnn` -> `khnon`, permits one bounded second separator pass
  for omitted inherent vowels, repairs `nz` and lower-trust lowercase `ng`
  before front vowels into the deterministic `nj` palatal nasal-ja route, and
  probes corpus-gated velar/anusvara-ga routes such as `rongin` -> `roNgin`,
  `jongi` -> `jonggi`, and `songit` -> `songgIt` / `soMgIt`. Repaired forms
  must survive Obadh transliteration and exact FST lookup.
- Bounded Unicode edit search intersected with the FST.
- Bangla-unit weighted edit scoring after retrieval, where nasal marks are
  cheaper than ordinary unrelated substitutions.
- Bounded stem-suffix completion for common Bengali determiner, case/focus, and
  plural suffixes, validated by exact FST lookup.
- Exact-baseline-only chandrabindu rescue over plausible vowel-bearing positions,
  validated by exact FST lookup.
- English loanword exact lookup against the compact loanword FST. This handles
  properly spelled English loanword input without touching the deterministic
  transliterator. Runtime query case is normalized, so `university`,
  `University`, and `UNIVERSITY` share the same stored key.
- Bounded English-key repair against the compact loanword FST for slight
  misspellings. The default path uses exact lookup first, adjacent transposition
  probes, then a tiny ASCII edit automaton with dynamic distance: no fuzzy search
  for short ambiguous keys, distance `1` for medium keys, and distance `2` for
  longer keys. Fuzzy loanword repairs are suppressed when Obadh's Bangla baseline
  is already an exact lexicon hit; exact loanword matches remain allowed.
- Bounded prefix lookup for autocomplete-style suggestions.

No word-specific correction table belongs in this layer. If a candidate is not
in the FST, the runtime candidate generator should not invent it. Future neural
models may rerank or contextualize candidates, but lexicon retrieval remains the
valid-word gate.

## Corpus Extraction

`extract-lexicon` accepts one or more UTF-8 text/HTML files, EPUB files, JSON
files, or directories. Directory inputs are expanded recursively and
deterministically, admitting only:

```text
.epub .html .htm .xhtml .json .txt .text .md .markdown
```

EPUB inputs prefer OPF spine reading order and skip navigation or non-linear
items when package metadata is available. Malformed or simple EPUBs fall back to
text-like publication members. JSON inputs are parsed structurally and extract
known prose fields such as `title`, `headline`, `content`, `text`, `body`, and
`article`. HTML-ish inputs strip markup, attributes, and script/style blocks
before tokenization.

The tokenizer keeps Bangla letters and combining signs, permits ZWNJ/ZWJ only
inside a word, and rejects digits, punctuation, Latin text, standalone marks,
and Assamese-only letters such as `ৰ` and `ৱ`.

The current Wikipedia source is the Kaggle `hurutta/bangla-wikipedia-dataset`
dataset, specifically the `wiki_bn_articles` JSON directory:

```bash
cargo run --bin obadh-autocorrect -- extract-lexicon \
  --input ~/.cache/kagglehub/datasets/hurutta/bangla-wikipedia-dataset/versions/1/wiki_bn_articles \
  --output data/autocorrect/lexicons/raw/wiki_bn.tsv \
  --min-frequency 1

cargo run --bin obadh-autocorrect -- merge-lexicon \
  --input data/autocorrect/lexicons/raw/wiki_bn.tsv \
  --output data/autocorrect/lexicons/raw/wiki_bn.tsv.clean \
  --drop-invalid

mv data/autocorrect/lexicons/raw/wiki_bn.tsv.clean data/autocorrect/lexicons/raw/wiki_bn.tsv
```

Do not merge generated artifacts back into source TSVs. Rebuild from the curated
corpus inputs so old generated frequencies do not compound.

## Pair Datasets And Evaluation

Pair datasets are for audits, benchmarks, training, and calibration. They are
not runtime artifacts and should not be committed if large.

Bangla correction pairs:

```text
observed_bangla<TAB>expected_bangla
```

Roman evaluation pairs:

```text
roman_input<TAB>expected_bangla
```

Run structural audits before using a pair file:

```bash
cargo run --bin obadh-autocorrect -- audit-pairs \
  --input path/to/pairs.tsv \
  --input-kind bangla --pretty

cargo run --bin obadh-autocorrect -- audit-pairs \
  --input path/to/roman_pairs.tsv \
  --input-kind roman --pretty
```

The production retrieval path is `suggest-fst` over `bn.fst` and, when needed,
`en_bn_loanwords.fst`. Older compact-artifact evaluation commands still exist
for compatibility, but new evaluation work should target the FST path.

Important evaluation fields to preserve when building FST-based reports:

- `baseline_accuracy`: input already equals target.
- `final_output_accuracy`: output after autocorrect equals target.
- `replacement_accuracy`: automatic replacements that were correct.
- `incorrect_replacements`: automatic replacements that changed input to a wrong
  word.
- `target_lexicon_coverage`: target exists in the loaded artifact.
- `candidate_recall_given_target_in_lexicon`: retrieval quality when coverage
  exists.
- `suggestion_recall_rate`: target is baseline or present in the candidate list.
- `mean_reciprocal_rank`: ranking quality for the expected target.

## Source Admission

Prefer sources that are extensive, auditable, license-usable, and clean enough to
produce a formal Bangla word inventory.

High-priority candidates:

- Locally curated Bangla books under `epubs/`.
- Bangladesh National Corpus if access and license permit derived artifacts.
- LDC-IL Gold Standard Bengali Raw Text Corpus if redistribution terms permit.
- Wikimedia-derived sources with proper attribution and filtering.
- Carefully inspected romanized/native pairs for evaluation only. Do not admit a
  transliteration dataset into the runtime lexicon just because it is large.

Secondary candidates:

- VACASPATI or other literary corpora, with register weighting so archaic forms
  do not dominate modern keyboard behavior.
- Web corpora only after strict script, spelling, duplication, and licensing
  audits.

Reject by default:

- Aksharantar.
- Unproven Kaggle/Hugging Face aggregations with unclear provenance.
- CommonCrawl/social/news dumps without strong audits.
- Mixed Assamese/Bangla or non-Bangla-script data.
