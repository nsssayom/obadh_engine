# Obadh Engine

Obadh is a deterministic Roman-to-Bangla transliteration engine for a larger
Bangla typing system. The base layer is deliberately rule-based: a Roman input
sequence maps to Bengali because a documented rule says so, not because a
dictionary guessed the word.

This project is an Avro successor in ambition, but the deterministic core is
not an Avro clone and not a word-by-word compatibility table. Users type
deliberately according to Obadh's own Roman rule contract. Correction,
suggestion, ranking, personalization, and neural context models live above that
core.

Live playground: [https://sayom.me/obadh_engine/](https://sayom.me/obadh_engine/)

## Quick Start

```bash
git clone https://github.com/nsssayom/obadh_engine.git
cd obadh_engine
./init.sh
```

`init.sh` initializes the direct data submodules, resolves Git LFS objects, and
installs web dependencies:

| Path | Data repo |
| --- | --- |
| `data/autocorrect` | [`nsssayom/obadh_autocorrect_dataset`](https://github.com/nsssayom/obadh_autocorrect_dataset) |
| `data/autosuggest` | [`nsssayom/obadh_autosuggest_dataset`](https://github.com/nsssayom/obadh_autosuggest_dataset) |

Native prerequisites that are not installed by `init.sh`:

```bash
rustup toolchain install 1.89.0
brew install wasm-pack binaryen
```

Common commands:

```bash
cargo run --bin obadh -- 'aji e probhate robir kor'
./build.sh dev
./build.sh dist
cargo test
cargo bench --bench hot_path
```

## Runtime Shape

```mermaid
flowchart LR
  R[Roman input] --> O[Deterministic Obadh core]
  O --> B[Bengali baseline]
  B --> C[Active-word autocorrect FST]
  C --> W[Correction candidates]
  T[Committed Bengali text] --> N[Next-word ngram autosuggest]
  N --> S[Suggestion candidates]
```

## Core Contract

The deterministic transliterator must remain dictionary-free.

- No whole-word correction table in the core.
- No hidden compatibility aliases just because another keyboard accepted them.
- No ML or corpus dependency on the transliteration hot path.
- Rule aliases need an Obadh-specific phonetic, orthographic, or ergonomic reason.
- Spelling correction and ranking belong in layers above the core.

Representative deliberate signals:

| Roman Signal | Bengali Rule Intent |
| --- | --- |
| `o` | inherent অ / lowercase cluster separator |
| `a` / `A` | visible আ / া, including before clusters |
| `I`, `U`, `O` | long ঈ / ঊ and ও |
| `aY` / `AY` | অ্যা / ্যা, e.g. `aYp` -> `অ্যাপ` |
| `ng`, `M`, `Ng` | anusvara / explicit anusvara escape / velar nasal |
| `ngg`, `nggh` | ঙ্গ / ঙ্ঘ shorthand |
| `jNG`, `jn`, `gg` | জ্ঞ paths |
| `NGj`, `nj`, `nJ` | ঞ্জ paths |
| `rr` + cluster | reph over a valid cluster |
| `rZy` / `rZY` | non-conjunct ZWNJ-separated র‌্য form |
| `y`, `w` | য-ফলা / ব-ফলা markers in declared clusters |
| `,,` | explicit hasant / conjunct boundary command |
| <code>t``</code> / <code>T``</code> | খণ্ড ত / ৎ |
| `^`, `:`, `.`, `$` | chandrabindu, visarga, danda, taka sign |

Rule sources live under `data/rules/` and are checked by tests.

## Autocorrect

Autocorrect is an active-word layer above the deterministic core. Obadh first
produces a Bengali baseline. The autocorrect layer then retrieves valid lexicon
candidates from compact FST artifacts and ranks them through bounded,
explainable channels.

Current runtime channels:

- exact Obadh baseline lookup
- bounded Obadh-aware Roman repair, such as missing lowercase `o` separators
- Bangla weighted edit lookup over the FST
- narrow vowel-length and nasal-mark rescue channels
- exact-stem suffix completion
- curated English-loanword exact and bounded fuzzy lookup
- bounded prefix completion

Runtime code does not parse CSV, TSV, EPUB, JSON, or large heap-resident trie
structures. Native tools can memory-map the FST; WASM loads the same compact
artifact bytes.

Inspect shipped artifacts:

```bash
cargo run --release --bin obadh-autocorrect -- inspect-fst-lexicon \
  --input data/autocorrect/models/bn.fst --pretty

cargo run --release --bin obadh-autocorrect -- inspect-loanword-lexicon \
  --input data/autocorrect/models/en_bn_loanwords.fst --pretty
```

Probe the production FST path:

```bash
cargo run --release --bin obadh-autocorrect -- suggest-fst \
  --lexicon data/autocorrect/models/bn.fst \
  --loanwords data/autocorrect/models/en_bn_loanwords.fst \
  --input sushil \
  --max-distance 2 \
  --max-candidates 512 \
  --max-prefix-candidates 24 \
  --response-candidates 8 \
  --pretty
```

The autocorrect dataset repo contains the auditable lexicon TSVs and runtime
FSTs. Build commands and corpus policy are documented in
`data/autocorrect/README.md` after `./init.sh`.

## Next-Word Autosuggest

Autosuggest is the next-word layer above committed Bengali text. It does not
run while a Roman token is active, does not transliterate Roman input, and does
not replace active-word autocorrect.

The production path starts with a compact n-gram candidate generator that
retrieves likely next Bengali words through trigram -> bigram -> unigram
backoff. A future neural layer can rerank this bounded candidate set, but the
current shipped runtime is the n-gram artifact.

Runtime integrations can layer `PersonalAutosuggest` above the static artifact
for on-device learning. It stores only bounded token-id transitions, keeps the
static corpus artifact unchanged, and can merge personal candidates with the
model using caller-owned scratch buffers. The personal layer can also round-trip
through a compact binary snapshot for local persistence.

The n-gram artifact is a fixed-width binary file designed for mmap/native and
byte-buffer/WASM loading. It stores vocabulary text, sorted token lookup rows,
bounded context rows, and candidate records in one portable blob. Native
integrations should keep resolved token IDs on the typing hot path and use
`suggest_ids_for_context_into`; materialize candidate text only for the few
words that will actually be shown. The text APIs remain available for tools,
tests, and slower integration points.
Sentence starts and explicit sentence/editor boundaries use the artifact's
learned `<bos>` row; unknown in-sentence tokens deliberately fall back without
pretending to be a boundary.

Build a local smoke artifact from the current corpus:

```bash
python3 -m tools.autosuggest.build_ngram_lm \
  --backend memory \
  --max-sentences 20000 \
  --output target/autosuggest-smoke.bin

cargo run --release --bin obadh-autosuggest -- inspect \
  --input target/autosuggest-smoke.bin \
  --pretty

cargo run --release --bin obadh-autosuggest -- suggest \
  --model target/autosuggest-smoke.bin \
  --context 'আমি আজ' \
  --top-k 5 \
  --pretty
```

For full corpus artifact generation, use the SQLite backend. It is slower than
the in-memory smoke path, but it keeps counting disk-backed, streams the final
artifact sections, and preserves exact counts:

```bash
python3 -m tools.autosuggest.build_ngram_lm \
  --backend sqlite \
  --min-count 10 \
  --max-candidates-per-prefix 5 \
  --unigram-size 4096 \
  --log-every-sentences 250000 \
  --output data/autosuggest/models/ngram/autosuggest-ngram.bin
```

When tuning profile parameters, keep the SQLite count DB and re-export without
recounting the corpus:

```bash
python3 -m tools.autosuggest.build_ngram_lm \
  --backend sqlite \
  --reuse-sqlite \
  --sqlite-path data/autosuggest/models/ngram/autosuggest-ngram.sqlite \
  --min-count 5 \
  --max-candidates-per-prefix 5 \
  --unigram-size 4096 \
  --output target/autosuggest-profile.bin
```

Evaluate and benchmark an artifact:

```bash
python3 -m tools.autosuggest.eval_ngram_lm \
  --model target/autosuggest-smoke.bin \
  --source epub --source news --source wiki \
  --skip-sentences-per-source 100000 \
  --max-sentences-per-source 5000 \
  --top-k 10

cargo run --release --bin obadh-autosuggest -- bench \
  --model target/autosuggest-smoke.bin \
  --context 'আমি আজ' \
  --context 'বাংলাদেশের মানুষ' \
  --mode context \
  --iterations 200000 \
  --pretty

cargo run --release --bin obadh-autosuggest -- bench \
  --model target/autosuggest-smoke.bin \
  --context 'আমি আজ' \
  --context 'বাংলাদেশের মানুষ' \
  --mode session \
  --iterations 200000 \
  --pretty
```

Current full-corpus candidate profiles from the 161.6M-token corpus:

| Profile | Artifact | Context rows | Replay top-5 | Native context lookup |
| --- | ---: | ---: | ---: | ---: |
| mobile candidate layer | `16.79 MB` | `371,089` | `24.84%` | `~0.10 us` |
| compact c4 baseline | `15.99 MB` | `371,089` | `23.65%` | not resampled |
| wider candidate layer | `31.94 MB` | `782,978` | `24.83%` | not sampled |
| research/wide layer | `65.22 MB` | `1,519,688` | `27.27%` | not resampled |

The replay metric is measured on corpus rows also used for counting, so it is a
profile-comparison signal rather than a held-out product accuracy claim.

Held-out profile probe, trained on the first `100,000` sentences per source and
evaluated on the next `25,000` sentences per source:

| Profile | Split artifact | Held-out top-5 | Held-out top-10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| c4 count/backoff | `1.78 MB` | `16.46%` | `18.75%` | `11.83%` |
| c5 count/backoff | `1.79 MB` | `17.23%` | `19.57%` | `12.00%` |
| c6 count/backoff | `1.81 MB` | `17.23%` | `20.28%` | `12.11%` |
| c8 count/backoff | `1.83 MB` | `17.23%` | `21.30%` | `12.25%` |
| c5 smoothed score | `1.79 MB` | `17.14%` | `19.56%` | `11.90%` |
| c5 stupid-backoff score | `1.79 MB` | `17.22%` | `19.57%` | `12.00%` |
| c5 count/backoff, min-count 20 | `1.57 MB` | `15.01%` | `17.41%` | `10.50%` |

The five-slot UI currently uses the c5 count/backoff profile. c6/c8 improve
only deeper top-10 recall in this probe, while smoothed score merging regresses
the visible top-5 candidate band. Stupid-backoff scoring tied the visible band
without improving it, and raising `min-count` removed too much useful context
even though it reduced artifact size.

The checked-in runtime artifact is the mobile n-gram profile:

| Item | Value |
| --- | --- |
| model family | bounded n-gram LM |
| task | next Bengali word prediction |
| context | newest `2` committed Bengali tokens |
| short context behavior | `<bos>` at sentence start, then bigram/unigram backoff |
| vocabulary | `32,768` tokens |
| unigram fallback | top `4,096` tokens |
| bigram rows | `32,266` |
| trigram rows | `338,823` |
| candidate rows | `798,834` |
| max candidates per context | `5` |
| artifact bytes | `16,792,106` |
| native runtime | mmap binary artifact plus `AutosuggestContext` |
| playground runtime | Obadh WASM parser over a binary artifact |

Keyboard integrations should keep an `AutosuggestContext` as words are
committed and call `suggest_ids_for_context_into` with a reused candidate-ID
buffer. For personalized suggestions, keep an `AutosuggestSession` per editor
surface and call `suggest_ids` when the platform can stay token-ID-first.
Commit resolved vocabulary IDs through `commit_token_id` on the hot path, use
`commit_token` only when text still needs lookup, and reuse the session-owned
personal/model/output buffers. Persist personalization with
`write_personal_snapshot_into` when the platform can reuse a save buffer. The
personal store grows lazily, so empty editor sessions do not reserve the full
history cap.
The text-based APIs remain useful for tools and tests, but they intentionally
include token parsing and lookup overhead that a keyboard does not need on each
suggestion request. Sentence-ending punctuation (`।`, `॥`, `.`, `!`, `?`, and
ellipsis) clears the recent context so the next sentence backs off cleanly
instead of inheriting the previous sentence's last words. Native editor
integrations can also call `AutosuggestContext::push_boundary()` on explicit
line or paragraph breaks.

Corpus snapshot:

| Source | Documents | Sentences | Tokens |
| --- | ---: | ---: | ---: |
| curated EPUB | `13` | `159,068` | `1,472,288` |
| Bangla Wikipedia | `169,736` | `4,297,804` | `54,560,642` |
| Bangla newspaper | `408,471` | `8,887,488` | `105,605,338` |
| total | `578,220` | `13,344,360` | `161,638,268` |

The vocabulary is built with `min_frequency = 3`, covers `148,611,832` corpus
tokens, and reaches `91.94%` token coverage.

Mobile artifact replay snapshot:

| Metric | Value |
| --- | ---: |
| top-1 | `12.04%` |
| top-3 | `20.48%` |
| top-5 | `24.84%` |
| top-10 | `26.96%` |
| MRR | `17.00%` |

These numbers are for reproducibility and regression checks only. Replay is
measured against corpus rows used for counting, so it is not a held-out product
accuracy claim.

## Data Policy

The main repo owns source code, docs, tests, the playground, and generated
GitHub Pages assets. Heavy data lives in data-only submodules mounted at the
same paths used by the tools:

```text
data/autocorrect   -> obadh_autocorrect_dataset
data/autosuggest   -> obadh_autosuggest_dataset
```

Those dataset repos may use Git LFS for corpora, TSVs, FSTs, and binary model
artifacts. They must not contain training/runtime code. GitHub Pages runtime
files under `docs/` are real bytes, not LFS pointers, because Pages branch
deploys do not serve LFS objects as ordinary static assets.

Manual submodule recovery:

```bash
git submodule update --init --recursive -- data/autocorrect data/autosuggest
git -C data/autocorrect lfs pull
git -C data/autosuggest lfs pull
```

## Web / WASM Usage

```javascript
import init, { ObadhaWasm } from './js/obadh_engine.js';

await init();
const engine = new ObadhaWasm();

console.log(engine.transliterate('aji e probhate robir kor'));
// আজি এ প্রভাতে রবির কর
```

Strict transliteration returns the original text unchanged when unsupported
characters are present. Use `transliterate_lenient` only when the caller
deliberately wants unsupported characters removed before transliteration.

## Rust Library Usage

```rust
use obadh_engine::ObadhEngine;

let engine = ObadhEngine::new();
let bangla = engine.transliterate("aji e probhate robir kor");

assert_eq!(bangla, "আজি এ প্রভাতে রবির কর");
```

For editor integrations, reuse buffers where possible:

```rust
use obadh_engine::{ObadhEngine, PhoneticUnit};

let engine = ObadhEngine::new();
let mut units: Vec<PhoneticUnit> = Vec::new();

engine.tokenize_phonetic_into("rrkSh", &mut units);
engine.tokenize_phonetic_into("praNer", &mut units);
```

## Current Metrics

| Check | Result |
| --- | --- |
| transliteration sample average | `0.002815 ms` |
| sample iterations | `100,000` |
| Bangla FST entries | `845,461` |
| Bangla FST bytes | `8,847,897` |
| English loanword keys | `1,776` |
| English loanword FST bytes | `89,427` |
| optimized WASM | about `280 KB` |
| autosuggest ngram artifact | `16,792,106` bytes |
| autosuggest vocab | `1,058,854` bytes |
| autosuggest native ID context lookup sample | `~0.060 us` |
| autosuggest native text context lookup sample | `~0.134 us` |

Autocorrect CLI process timings are not reported as keyboard latency because
process startup dominates those measurements. Keyboard-time performance should
be measured inside loaded runtimes.

## Project Layout

```text
src/engine/                 deterministic tokenizer/transliterator
src/definitions/            compiled rule tables
src/autocorrect/            FST candidate generation and ranking primitives
src/autosuggest/            static ngram runtime and bounded personal overlay
src/wasm/                   WebAssembly bindings
src/bin/                    CLI binaries
data/rules/                 documented deterministic rule sources
data/autocorrect/           direct data submodule: lexicon TSVs and FSTs
data/autosuggest/           direct data submodule: corpus, vocab, ngram model
tools/autocorrect/          corpus and loanword data utilities
tools/autosuggest/          sentence corpus, vocab, and ngram model utilities
www/                        playground source
docs/                       generated GitHub Pages distribution
tests/                      regression suite
benches/                    Criterion hot-path benchmarks
```

`docs/` is generated by `./build.sh dist`. Do not edit generated CSS, WASM, or
copied distribution files directly.

## Release Checklist

```bash
cargo test
cargo bench --bench hot_path --no-run
./build.sh dist
git status --short
```

For a tagged release, bump the Cargo/npm versions together, rebuild `docs/`,
commit source plus generated artifacts, push, then tag the exact commit.
