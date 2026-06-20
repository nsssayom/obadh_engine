# Obadh Autocorrect Data Policy

This directory stores the checked-in autocorrect lexicon artifacts plus the
repeatable build notes for producing them.

The intended structure is:

```text
data/autocorrect/
  README.md
  lexicons/epub_bn.tsv
  lexicons/wiki_bn.tsv
  lexicons/bn.tsv
  models/bn.lex
```

`lexicons/epub_bn.tsv` and `lexicons/wiki_bn.tsv` are source-specific
word-frequency TSVs. `lexicons/bn.tsv` is the unified auditable Bangla
word-frequency TSV. `models/bn.lex` is the compact runtime binary built from the
unified TSV. Raw EPUB/text/JSON corpora stay outside this directory; the local
`epubs/` directory and Kaggle cache are corpus inputs, not repo artifacts.
Temporary candidate exports or training scratch files should go under ignored
scratch directories such as
`data/autocorrect/tmp/`, `data/autocorrect/candidates/`, or
`data/autocorrect/training/`.

## Runtime Artifact Contract

Runtime autocorrect should load a compact Obadh lexicon artifact, not CSV or raw
dataset files. Build artifacts with:

```bash
cargo run --bin obadh-autocorrect -- prepare-lexicon \
  --input epubs \
  --words-output data/autocorrect/lexicons/epub_bn.tsv \
  --lexicon-output data/autocorrect/tmp/epub_bn.lex \
  --min-frequency 1 \
  --pretty

cargo run --bin obadh-autocorrect -- extract-lexicon \
  --input path/to/clean_corpus.txt \
  --input path/to/book.epub \
  --input path/to/wiki_json_directory \
  --input path/to/book_directory \
  --output path/to/clean_bn_words.tsv \
  --min-frequency 1

cargo run --bin obadh-autocorrect -- audit-lexicon \
  --input path/to/clean_bn_words.tsv --pretty

cargo run --bin obadh-autocorrect -- merge-lexicon \
  --input path/to/dakshina_words.tsv \
  --input path/to/wiki_words.tsv \
  --output path/to/merged_bn_words.tsv \
  --drop-invalid

cargo run --bin obadh-autocorrect -- audit-pairs \
  --input path/to/bangla_pairs.tsv \
  --input-kind bangla --pretty

cargo run --bin obadh-autocorrect -- merge-lexicon \
  --input data/autocorrect/lexicons/epub_bn.tsv \
  --input data/autocorrect/lexicons/wiki_bn.tsv \
  --output data/autocorrect/lexicons/bn.tsv \
  --drop-invalid

cargo run --bin obadh-autocorrect -- build-lexicon \
  --input data/autocorrect/lexicons/bn.tsv \
  --output data/autocorrect/models/bn.lex

cargo run --bin obadh-autocorrect -- export-lexicon \
  --input www/assets/autocorrect/bn.lex \
  --output data/autocorrect/tmp/shipped_words.tsv

cargo run --bin obadh-autocorrect -- suggest \
  --lexicon path/to/obadh.bn.lex \
  --input kmn --pretty
```

Input TSV format:

```text
বাংলা_শব্দ<TAB>frequency
```

The frequency column is optional and defaults to `1`. Bangla-only validation is
enabled by default.

The current compact artifact magic is `OBACLEX3`. It stores each word's
frequency plus precomputed Bangla unit length and an interned phonetic-skeleton
table so mobile and WASM startup do not recompute all derived analysis from
scratch or keep duplicate skeleton strings in memory. The loader still accepts
legacy `OBACLEX1` and `OBACLEX2` artifacts; `OBACLEX1` rebuilds the missing
analysis on load, while `OBACLEX2` interns its per-entry skeletons at load time.

`extract-lexicon` accepts one or more `--input` UTF-8 text/HTML files, EPUB
files, JSON files, or directories and emits a sorted word-frequency TSV.
Directory inputs are expanded recursively and deterministically, but only
`.epub`, `.html`, `.htm`, `.xhtml`, `.json`, `.txt`, `.text`, `.md`, and
`.markdown` files are admitted from
directories. EPUB inputs prefer the OPF spine reading order and skip navigation
or non-linear items when the package metadata is available; malformed/simple
EPUBs fall back to text-like publication members (`.xhtml`, `.html`, `.htm`,
`.txt`). JSON inputs are parsed structurally and extract known prose fields such
as `title`, `headline`, `content`, `text`, `body`, and `article`. HTML-ish inputs
strip markup, attributes, and script/style blocks before tokenization.

The tokenizer normalizes Bangla text to NFC, keeps Bangla letters and combining
signs, permits ZWNJ and ZWJ only inside a word, and rejects digits, punctuation,
Latin text, standalone marks, and Assamese-only letters such as `ৰ` and `ৱ`.
Lexicon TSV ingestion also normalizes words before audit, merge, and artifact
build so decomposed forms do not split frequency buckets.

The extraction JSON report includes `text_inputs`, `html_inputs`, `epub_inputs`,
`json_inputs`, `epub_inputs`, `epub_spine_items`, `epub_fallback_inputs`, and
`epub_fallback_items`. `inputs` reports the requested roots and
`expanded_inputs` reports the number of files discovered under those roots.
Treat fallback EPUB extraction as lower-trust corpus input because navigation,
appendix, or unreferenced publication files may be mixed into the token stream.

`prepare-lexicon` is the repeatable local corpus ingestion path. It runs
extraction, strict Bangla-only audit, and compact lexicon artifact build in one
step. Drop additional EPUB files under `epubs/` and rerun the command above to
refresh the checked-in TSV and `.lex` artifact.

Use `export-lexicon` only when a compact runtime artifact needs to be inspected
or recovered back into TSV form. Do not use the previously shipped artifact as a
standing input for normal rebuilds; repeatedly merging it back into the corpus
will compound old generated frequencies. The clean local runtime build path is
the curated corpus itself:

```bash
cargo run --bin obadh-autocorrect -- extract-lexicon \
  --input epubs \
  --output data/autocorrect/tmp/bn.all.tsv \
  --min-frequency 1

cargo run --bin obadh-autocorrect -- prepare-lexicon \
  --input epubs \
  --words-output data/autocorrect/lexicons/epub_bn.tsv \
  --lexicon-output data/autocorrect/tmp/epub_bn.lex \
  --min-frequency 1 \
  --pretty
```

The current Wikipedia source is the Kaggle `hurutta/bangla-wikipedia-dataset`
dataset, specifically the `wiki_bn_articles` JSON directory. Download it locally
with Kaggle tooling, then extract it with:

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

`merge-lexicon` accepts one or more word-frequency TSVs, sums duplicate word
frequencies deterministically, and writes a merged TSV sorted by frequency then
word. It is strict by default. Use `--drop-invalid` only when intentionally
cleaning a source; the JSON report records dropped, malformed, non-Bangla, and
invalid-frequency rows. `build-lexicon` remains strict and should consume clean
TSVs only.

## Pair Dataset Contract

Pair datasets are for audits, benchmarks, training, and calibration. They are
not runtime artifacts and should not be committed if large.

Word-level Bangla correction pairs use:

```text
observed_bangla<TAB>expected_bangla
```

Word-level Roman evaluation pairs use:

```text
roman_input<TAB>expected_bangla
```

Run `audit-pairs` before using a pair file:

```bash
cargo run --bin obadh-autocorrect -- audit-pairs \
  --input path/to/pairs.tsv \
  --input-kind bangla

cargo run --bin obadh-autocorrect -- audit-pairs \
  --input path/to/roman_pairs.tsv \
  --input-kind roman
```

The audit is intentionally structural. It rejects mixed-script source fields,
non-Bangla targets, malformed rows, empty fields, and dirty Roman tokens, then
reports duplicates, identity pairs, and the edit-distance gap from the current
baseline. It is a gate before model/data work, not a replacement for human or
model-assisted linguistic review.

## Evaluation Metrics

Evaluate candidate artifacts with:

```bash
cargo run --bin obadh-autocorrect -- eval \
  --lexicon path/to/obadh.bn.lex \
  --input path/to/eval_pairs.tsv \
  --input-kind bangla
```

Export labeled candidate JSONL for retrieval analysis or future model training
with:

```bash
cargo run --bin obadh-autocorrect -- export-candidates \
  --lexicon path/to/obadh.bn.lex \
  --input path/to/eval_pairs.tsv \
  --output path/to/candidates.jsonl \
  --input-kind bangla \
  --max-candidates 64 \
  --max-skeleton-candidates 128
```

Each JSONL record contains the original source, expected target, Obadh/Bangla
baseline, optional replacement, target rank, and the candidate list. Each
candidate includes a fixed-width integer feature vector with this order:

```text
source_id, edit_cost, input_unit_len, candidate_unit_len, unit_len_delta,
frequency_log2, input_known, candidate_known, obadh_baseline
```

Runtime candidate generation is corpus-backed through lexicon edit search,
phonetic skeleton search, and prefix completion.

`source_id` is stable for exported data:

```text
0 = no_change
1 = lexicon_edit
2 = phonetic_skeleton
4 = prefix_completion
```

The deterministic candidate score is exported separately from the feature
vector. There is no handwritten runtime rank-bonus feature.

Evaluation input is:

```text
observed_input<TAB>expected_bangla
```

For `--input-kind roman`, `observed_input` is passed through Obadh first. For
`--input-kind bangla`, `observed_input` is treated as the already-visible Bangla
buffer.

Important output fields:

- `baseline_accuracy`: how often the input already matches the target.
- `final_output_accuracy`: how often the final output after autocorrect matches
  the target.
- `replacement_accuracy`: among automatic replacements only, how often the
  replacement was correct.
- `incorrect_replacements`: automatic replacements that changed the input to the
  wrong word. This must stay very low for keyboard trust.
- `target_lexicon_coverage`: how often the expected target is even present in
  the loaded lexicon artifact. This is the hard ceiling for lexicon-backed
  candidate generation.
- `candidate_recall_given_target_in_lexicon`: how often the correct target is
  surfaced when the lexicon contains it. Use this to judge retrieval separately
  from corpus coverage.
- `suggestion_recall_rate`: how often the correct target was either already the
  baseline or present in the candidate list.
- `mean_reciprocal_rank`: ranking quality for the expected target.

`eval`, `suggest`, and `export-candidates` share retrieval controls:
`--max-candidates`, `--max-edit-cost`, `--max-prefix-candidates`,
`--max-skeleton-candidates`, `--max-skeleton-edit-cost`, and
`--search-known-input`. Default values model a tight runtime suggestion list.
Use a wider pool for offline retrieval analysis so candidate recall is not
artificially limited by production UI constraints. Prefix completion is backed
by a tiny per-trie-node top-frequency index, so it returns bounded autocomplete
candidates without scanning the lexicon.

For `--input-kind roman`, default runtime retrieval skips Bangla-unit edit
search and relies on the folded phonetic skeleton. This avoids the expensive
edit-trie path over Obadh's intermediate output, which can be structurally far
from the expected word. Use `--max-edit-cost` only when deliberately measuring
that slower Roman-origin edit-search path.

By default, Roman-origin requests are not automatically replaced by the
lexicon-only runtime ranker. They still produce candidates and exported training
features. Automatic replacement for Roman-origin requests should be enabled only
after a production-grade ranking model proves that it improves final accuracy
without increasing incorrect replacements.

Important pair-audit fields:

- `accepted_rate`: the share of structurally usable rows.
- `non_bangla_source_rows`, `non_roman_source_rows`, and
  `non_bangla_expected_rows`: script pollution counters.
- `identity_rows`: pairs that teach no correction behavior.
- `baseline_exact_rate`: how often the current baseline already equals the
  target.
- `baseline_mean_edit_cost`: average Bangla-unit edit distance between baseline
  and target for accepted rows.

## Source Admission

Prefer sources that are extensive, auditable, license-usable, and clean enough to
produce a formal Bangla word inventory.

High-priority candidates:

- Google Dakshina `bn`: use for romanized-to-native evaluation and attested
  romanization behavior, not as the only word inventory.
- Bangladesh National Corpus (BdNC): high priority if access and license permit
  derived word-frequency artifacts.
- LDC-IL Gold Standard Bengali Raw Text Corpus: useful cleaned corpus candidate,
  with license/access conditions to verify before redistribution.
- Wikimedia-derived sources: useful for reproducible formal vocabulary, with
  license attribution requirements.

Secondary candidates:

- VACASPATI: high-quality literary corpus, but should be register-weighted so
  older literary forms do not dominate modern keyboard behavior.
- bnWaC or similar web corpora: use only after strict Bangla-only filtering,
  normalization, and source/license review.

Reject by default:

- Aksharantar.
- Unproven Kaggle/Hugging Face aggregations with unclear provenance.
- CommonCrawl/social/news dumps without script, spelling, duplication, and
  licensing audits.
- Mixed Assamese/Bangla or non-Bangla-script data.
