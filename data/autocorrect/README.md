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
    epub_bn.tsv
    wiki_bn.tsv
    bn.tsv
  models/
    bn.fst
```

- `lexicons/epub_bn.tsv`: word-frequency TSV extracted from locally curated
  EPUB books.
- `lexicons/wiki_bn.tsv`: word-frequency TSV extracted from the accepted Bangla
  Wikipedia source.
- `lexicons/bn.tsv`: unified, auditable Bangla word-frequency TSV.
- `models/bn.fst`: production finite-state lexicon built from `bn.tsv`.

Raw corpus inputs stay outside this directory. The local `epubs/` directory,
Kaggle caches, downloaded JSON, temporary candidate dumps, and training exports
are not repo artifacts. Use ignored scratch directories such as:

```text
data/autocorrect/tmp/
data/autocorrect/candidates/
data/autocorrect/training/
```

## Runtime Artifact Contract

Runtime autocorrect loads `models/bn.fst`, not CSV, raw corpora, or large parsed
TSV structures. The FST maps each NFC-normalized Bangla word to its unigram
frequency.

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
  --output data/autocorrect/lexicons/epub_bn.tsv \
  --min-frequency 1
```

Audit a source TSV:

```bash
cargo run --bin obadh-autocorrect -- audit-lexicon \
  --input data/autocorrect/lexicons/epub_bn.tsv --pretty
```

Merge accepted TSV sources:

```bash
cargo run --bin obadh-autocorrect -- merge-lexicon \
  --input data/autocorrect/lexicons/epub_bn.tsv \
  --input data/autocorrect/lexicons/wiki_bn.tsv \
  --output data/autocorrect/lexicons/bn.tsv \
  --drop-invalid
```

Build and inspect the runtime FST:

```bash
cargo run --bin obadh-autocorrect -- build-fst-lexicon \
  --input data/autocorrect/lexicons/bn.tsv \
  --output data/autocorrect/models/bn.fst

cargo run --bin obadh-autocorrect -- inspect-fst-lexicon \
  --input data/autocorrect/models/bn.fst --pretty
```

Probe the runtime path:

```bash
cargo run --bin obadh-autocorrect -- suggest-fst \
  --lexicon data/autocorrect/models/bn.fst \
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
  --output data/autocorrect/lexicons/wiki_bn.tsv \
  --min-frequency 1

cargo run --bin obadh-autocorrect -- merge-lexicon \
  --input data/autocorrect/lexicons/wiki_bn.tsv \
  --output data/autocorrect/lexicons/wiki_bn.tsv.clean \
  --drop-invalid

mv data/autocorrect/lexicons/wiki_bn.tsv.clean data/autocorrect/lexicons/wiki_bn.tsv
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

`eval` and `export-candidates` currently operate on the older compact `.lex`
artifact for offline compatibility. The production runtime path is `bn.fst`;
the evaluation/export path should be migrated before any serious model training
depends on it.

```bash
cargo run --bin obadh-autocorrect -- eval \
  --lexicon path/to/obadh.bn.lex \
  --input path/to/eval_pairs.tsv \
  --input-kind bangla --pretty

cargo run --bin obadh-autocorrect -- export-candidates \
  --lexicon path/to/obadh.bn.lex \
  --input path/to/eval_pairs.tsv \
  --output path/to/candidates.jsonl \
  --input-kind bangla \
  --max-candidates 64 \
  --max-skeleton-candidates 128
```

Important evaluation fields:

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
- Google Dakshina `bn` for romanized/native evaluation pairs, not as the only
  word inventory.

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
