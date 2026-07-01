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
cargo run --features cli --bin obadh -- 'aji e probhate robir kor'
./build.sh dev
./build.sh dist
cargo test
cargo bench --bench hot_path
```

## Crate Features

The crates.io package is the Rust SDK. It includes source code, CLI/WASM source
behind explicit features, tests, and the small deterministic rule fixtures. It
does not bundle large runtime model artifacts.

```toml
# Native Rust SDK: deterministic core, autocorrect/autosuggest runtimes,
# personal autosuggest, and native model handoff helpers.
obadh_engine = "0.5"

# CLI tools and artifact builders.
obadh_engine = { version = "0.5", features = ["cli"] }

# Browser playground bindings.
obadh_engine = { version = "0.5", features = ["wasm"] }
```

The default feature set is empty. Native downstreams, including the planned iOS
wrapper, do not pay for `clap`, ZIP/EPUB tooling, `wasm-bindgen`, or browser
APIs unless they opt into those features.

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
cargo run --release --features cli --bin obadh-autocorrect -- inspect-fst-lexicon \
  --input data/autocorrect/models/bn.fst --pretty

cargo run --release --features cli --bin obadh-autocorrect -- inspect-loanword-lexicon \
  --input data/autocorrect/models/en_bn_loanwords.fst --pretty
```

Probe the production FST path:

```bash
cargo run --release --features cli --bin obadh-autocorrect -- suggest-fst \
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

The production path starts with a bounded n-gram candidate generator that
retrieves likely next Bengali words through suffix backoff. The current browser
runtime uses a compact c16 fourgram profile plus the bounded personal overlay.
Native integrations can use the wider c64 retriever plus the packaged GRU256
generator: the model emits a top-128 vocabulary proposal set, scores the static
c64 pool, and a fixed scored-union policy merges both streams after a Bengali
word is committed. The neural path is not part of Roman keystroke
transliteration.

Runtime integrations can layer `PersonalAutosuggest` above the static artifact
for on-device learning. It stores only bounded token-id transitions, keeps the
static corpus artifact unchanged, and can merge personal candidates with the
model using caller-owned scratch buffers. The personal layer can also round-trip
through a compact binary snapshot for local persistence.
The WASM playground exercises this path: committed Bengali words update the
bounded personal model, and the ribbon can surface personal next-word
suggestions without modifying the static artifact.

Personal autosuggest has two separate lifetimes. The live session context is
short-lived: it is the recent words in the current editor flow and should be
cleared at editor/session boundaries. The personal dictionary is longer-lived:
it is a bounded local overlay that survives app restarts only if the host
platform exports and saves its snapshot, then imports it again during startup.
Obadh owns the compact snapshot format, vocabulary-fingerprint validation,
bounded learning rules, and merge behavior. Downstream keyboards own the storage
policy: where the snapshot lives, when it is saved, when it is cleared, and
whether a user-facing privacy/reset control is exposed. A missing or
fingerprint-mismatched snapshot must be treated as an empty personal dictionary;
the static autosuggest artifact is never rewritten by user learning.

The playground follows the same contract as a platform integration. It stores
the exported personal snapshot in browser `localStorage`, keyed by the
autosuggest vocabulary fingerprint, reloads it when the WASM autosuggest module
starts, and drops it if import validation fails. Native integrations should do
the equivalent with platform-local storage: load once when the keyboard/editor
session is initialized, call `commit_token_id`/`commit_token` as Bengali words
are accepted, export the snapshot on a debounce or lifecycle event, and import
it before serving suggestions in the next session.

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

cargo run --release --features cli --bin obadh-autosuggest -- inspect \
  --input target/autosuggest-smoke.bin \
  --pretty

cargo run --release --features cli --bin obadh-autosuggest -- suggest \
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
  --min-count 8 \
  --max-candidates-per-prefix 16 \
  --unigram-size 4096 \
  --log-every-sentences 250000 \
  --max-context-order 3 \
  --fourgram-min-count 14 \
  --compact-count-records \
  --output data/autosuggest/models/ngram/autosuggest-ngram.bin
```

Use `--compact-count-records` for size-only profile experiments,
`--score-mode kneser-ney` for single-discount Kneser-Ney, and
`--score-mode modified-kneser-ney` for modified-discount Kneser-Ney profile
research. The SQLite export path streams final artifact sections and uses
cached lower-order lookups for Kneser-Ney, so full-corpus scored exports do not
materialize the entire n-gram table in Python heap. The checked-in mobile
artifact currently uses compact count records because it keeps the visible
top-5 rate tied with the smaller c5 profile while expanding the hidden pool and
shrinking the model versus scored c16 profiles.

When tuning profile parameters, keep the SQLite count DB and re-export without
recounting the corpus:

```bash
python3 -m tools.autosuggest.build_ngram_lm \
  --backend sqlite \
  --reuse-sqlite \
  --sqlite-path data/autosuggest/models/ngram/autosuggest-ngram.sqlite \
  --min-count 8 \
  --max-candidates-per-prefix 16 \
  --unigram-size 4096 \
  --max-context-order 3 \
  --fourgram-min-count 14 \
  --compact-count-records \
  --output target/autosuggest-profile.bin
```

SQLite count DBs are keyed by vocabulary token IDs and now record the vocabulary
fingerprint and context order used during counting. Reuse is only valid with the
same vocab artifact; changing `vocab.tsv` requires rebuilding the count DB.

Use repeatable `--source-weight source=integer` flags for corpus-prior
experiments. The weights are applied during counting, remain integer-valued in
the artifact, and are recorded in the manifest. Source weighting is useful for
audits, but it is not currently part of the shipped profile because EPUB-heavy
split profiles hurt the visible top-5 band.

Evaluate and benchmark an artifact:

```bash
python3 -m tools.autosuggest.eval_ngram_lm \
  --model target/autosuggest-smoke.bin \
  --source epub --source news --source wiki \
  --skip-sentences-per-source 100000 \
  --max-sentences-per-source 5000 \
  --top-k 10

cargo run --release --features cli --bin obadh-autosuggest -- bench \
  --model target/autosuggest-smoke.bin \
  --context 'আমি আজ' \
  --context 'বাংলাদেশের মানুষ' \
  --mode context \
  --iterations 200000 \
  --pretty

cargo run --release --features cli --bin obadh-autosuggest -- bench \
  --model target/autosuggest-smoke.bin \
  --context 'আমি আজ' \
  --context 'বাংলাদেশের মানুষ' \
  --mode session \
  --iterations 200000 \
  --pretty
```

The evaluator reports both product-facing all-target rates and diagnostic
in-vocabulary rates. Use `top*_all_targets` for profile comparisons because
unknown vocabulary targets count as misses; use `top*` and
`skipped_unknown_ratio` to diagnose whether the vocabulary cap is hiding useful
words. When `--top-k` is larger than the visible suggestion band, the evaluator
also reports `topN_to_topK_headroom*` fields. Those are the candidates a future
reranker could promote without changing the retrieval artifact.

Current full-corpus candidate profiles from the 161.6M-token corpus, measured
on the same replay probe:

| Profile | Artifact | Context rows | Replay top-1 | Replay top-5 | Native context lookup |
| --- | ---: | ---: | ---: | ---: | ---: |
| trigram c5 min8 | `16.69 MB` | `466,003` | `11.61%` | `23.89%` | `~0.11 us` |
| fourgram c5 b8/t8/f14 | `21.47 MB` | `618,590` | `11.98%` | `24.03%` | `~0.12 us` |
| fourgram c5 b8/t8/f14 scored | `26.34 MB` | `618,590` | `12.58%` | `24.97%` | `~0.14 us` |
| fourgram c5 b8/t8/f14 Kneser-Ney | `26.34 MB` | `618,590` | `12.42%` | `24.69%` | `~0.18 us` |
| fourgram c16 b8/t8/f14 | `25.20 MB` | `618,590` | `11.98%` | `24.03%` | `~0.185 us` |
| fourgram c16 b8/t8/f14 scored | `31.93 MB` | `618,590` | `12.58%` | `24.97%` | `~0.14 us` |
| fourgram c16 b8/t8/f14 Kneser-Ney | `31.93 MB` | `618,590` | `12.42%` | `24.69%` | `~0.19 us` |
| fourgram c5 min8 | `27.13 MB` | `796,737` | `12.13%` | `24.12%` | `~0.12 us` |

The replay metric is measured on corpus rows also used for counting, so it is a
profile-comparison signal rather than a held-out product accuracy claim.

Held-out profile probe, trained on the first `100,000` sentences per source and
evaluated on the next `25,000` sentences per source:

| Profile | Split artifact | Held-out top-1 | Held-out top-5 | Held-out top-10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| trigram c5 min8 count/backoff | `1.79 MB` | `8.12%` | `16.64%` | `18.22%` | `11.43%` |
| trigram c5 min8 Kneser-Ney | `1.92 MB` | `8.12%` | `16.80%` | `18.38%` | `11.51%` |
| fourgram c5 min4 count/backoff | `2.69 MB` | `8.97%` | `18.28%` | `19.91%` | `12.73%` |
| fourgram c5 min4 scored backoff | `3.01 MB` | `8.97%` | `18.28%` | `19.91%` | `12.73%` |
| fourgram c5 min4 Kneser-Ney | `3.01 MB` | `8.95%` | `18.39%` | `20.20%` | `12.79%` |
| fourgram c16 min4 count/backoff | `2.89 MB` | `8.97%` | `18.28%` | `22.87%` | `13.25%` |
| fourgram c16 min4 Kneser-Ney | `3.31 MB` | `8.95%` | `18.39%` | `22.96%` | `13.30%` |
| fourgram c16 min4, EPUB weight 2 | `3.60 MB` | `8.75%` | `18.21%` | `22.72%` | `13.02%` |

The five-slot UI currently uses the full-corpus fourgram b8/t8/f14 compact
count/backoff profile with a wider hidden candidate pool. The visible top-5
rate ties the c5 compact profile, but top-10 all-target recall improves from
`25.72%` to `29.67%` while the artifact stays smaller than the older scored c5
profile. The native c64 profile gives the neural generator a bounded static
pool without changing the browser artifact.

Neural packaging lives under `tools/autosuggest/`. A model is only packaged when
it improves the static c64 pool under held-out probes, passes source-balanced
EPUB/news/Wikipedia checks, and keeps the runtime bounded:

| Gate | Current result |
| --- | --- |
| `train_candidate_reranker.py` | candidate-feature GRU/Transformer trials were not strong enough to ship |
| `train_next_word_lm.py` | 8.82M-parameter GRU256 over the 32k vocabulary; packaged as a top-128 generator plus c64 scorer head |

The packaged production path keeps static n-gram retrieval and slot one stable,
then lets the GRU proposal stream compete for the remaining visible slots. The
runtime contract is fixed: `uint32` token IDs inside Obadh, `int64` ONNX inputs,
`int32` Core ML inputs, `float32` scores, 16 context IDs, 128 generated token
IDs, and 64 static c64 candidate IDs.

Export and package that deployment-shaped generator:

```bash
python3 -m tools.autosuggest.export_next_word_lm \
  --checkpoint target/autosuggest-next-word-lm-gru256-c64-3m-continued2.pt \
  --artifact data/autosuggest/models/ngram/autosuggest-ngram-c64.bin \
  --output target/autosuggest-full-vocab-topk128-static64-gru256-balanced-combined.onnx \
  --quantized-output target/autosuggest-full-vocab-topk128-static64-gru256-balanced-combined.int8.onnx \
  --coreml-output target/autosuggest-full-vocab-topk128-static64-gru256-balanced-combined.mlpackage \
  --report target/autosuggest-full-vocab-topk128-static64-gru256-balanced-combined-report.json \
  --export-kind full-vocab-topk-scorer \
  --pool-k 64 \
  --top-k-output 128 \
  --max-examples-per-source 30000 \
  --benchmark-iterations 3000

python3 -m tools.autosuggest.package_scorer \
  --ngram data/autosuggest/models/ngram/autosuggest-ngram-c64.bin \
  --ngram-manifest data/autosuggest/models/ngram/autosuggest-ngram-c64.manifest.json \
  --scorer-report target/autosuggest-full-vocab-topk128-static64-gru256-balanced-combined-report.json \
  --onnx data/autosuggest/models/neural/autosuggest-generator-gru256-topk128-c64-balanced.onnx \
  --quantized-onnx data/autosuggest/models/neural/autosuggest-generator-gru256-topk128-c64-balanced.int8.onnx \
  --coreml data/autosuggest/models/neural/autosuggest-generator-gru256-topk128-c64-balanced.mlpackage \
  --output data/autosuggest/models/neural/autosuggest-generator-gru256-topk128-c64-balanced.manifest.json \
  --scored-union-profile balanced_by_mrr

cargo run --release --features cli --bin obadh-autosuggest -- validate-generator \
  --model data/autosuggest/models/ngram/autosuggest-ngram-c64.bin \
  --manifest data/autosuggest/models/neural/autosuggest-generator-gru256-topk128-c64-balanced.manifest.json \
  --pretty

python3 -m tools.autosuggest.verify_generator_package \
  --output target/autosuggest-generator-production-verify.json
```

`validate-generator` checks both the loaded n-gram compatibility and the
referenced package asset sizes/hashes. `verify_generator_package` additionally
replays the packager in `--check` mode and enforces the release-mode Rust
handoff latency/heap budgets. Use `--asset-root` when the manifest paths are
mounted under an app bundle or staging directory.

Current fixed-batch generator gate:

| Model | File | top-1 all | top-5 all | top-10 all | Local runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| static c64 pool | `29.49 MB` | `16.84%` | `31.34%` | `37.99%` | `~0.83 us` u32 candidate input |
| scored-union GRU256 | `17.67 MB` Core ML, `18.49 MB` INT8 ONNX | `16.84%` | `32.89%` | `39.42%` | `~14.38 us` release personal-aware Rust handoff, `~459 us` Core ML graph |

The selected scored-union profile improves top-5 and MRR on EPUB, news, and
Wikipedia held-out slices. It is packaged in `data/autosuggest/models/neural`;
the final hardware gate remains measurement inside real keyboard-extension
constraints on Apple devices.

The checked-in runtime artifact is the mobile n-gram profile:

| Item | Value |
| --- | --- |
| model family | bounded n-gram LM |
| task | next Bengali word prediction |
| artifact version | `3` compact fourgram format |
| artifact context | newest `3` committed Bengali tokens |
| runtime context buffer | newest `3` committed Bengali tokens |
| short context behavior | `<bos>` at sentence start, then bigram/unigram backoff |
| vocabulary | `32,768` tokens |
| unigram fallback | top `4,096` tokens |
| bigram rows | `32,533` |
| trigram rows | `433,470` |
| fourgram rows | `152,587` |
| candidate rows | `1,679,620` |
| candidate record | `8` bytes, token + count |
| max candidates per context | `16` hidden, UI usually requests `5` |
| min count | bigram/trigram `8`, fourgram `14` |
| artifact bytes | `25,195,978` |
| eval top-5 all targets | `24.03%` |
| eval top-10 all targets | `29.67%` |
| eval MRR all targets | `17.17%` |
| native runtime | mmap binary artifact plus `AutosuggestContext` |
| playground runtime | Obadh WASM parser over a binary artifact |

Keyboard integrations should keep an `AutosuggestContext` as words are
committed and call `suggest_ids_for_context_into` with a reused candidate-ID
buffer. For personalized suggestions, keep an `AutosuggestSession` per editor
surface and call `suggest_ids` when the platform can stay token-ID-first.
Commit resolved vocabulary IDs through `commit_token_id` on the hot path, use
`commit_token` only when text still needs lookup, and reuse the session-owned
personal/model/output buffers. Persist personalization with
`write_personal_snapshot_into` when the platform can reuse a save buffer, and
restore it with `import_personal_snapshot` so model compatibility and token IDs
are validated before the session mutates. The personal store grows lazily, so
empty editor sessions do not reserve the full history cap.
The personal overlay remains capped to its compact two-token snapshot format.
Use `estimated_heap_bytes` and `personal_snapshot_len` for current session
resource use; use `heap_limit_bytes` and `personal_snapshot_limit_bytes` for
conservative caps under the current personal-store and candidate-count
configuration.
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

Mobile artifact replay-probe snapshot:

| Metric | Value |
| --- | ---: |
| top-1 | `11.98%` |
| top-3 | `20.08%` |
| top-5 | `24.03%` |
| top-10 | `29.67%` |
| MRR | `17.17%` |

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

## Data Artifacts

Runtime data is packaged outside the crates.io tarball. The crate stays small
and auditable; data repos carry the large LFS artifacts:

| Runtime | Required artifacts |
| --- | --- |
| deterministic core | none |
| autocorrect | `data/autocorrect/models/bn.fst` |
| autocorrect loanwords | `data/autocorrect/models/en_bn_loanwords.fst` |
| browser autosuggest | `data/autosuggest/models/ngram/autosuggest-ngram.bin` |
| native autosuggest | `data/autosuggest/models/ngram/autosuggest-ngram-c64.bin` |
| neural next-word package | generator manifest plus Core ML or ONNX model |

The main repo pins exact data commits through submodules. A fresh checkout
should run `./init.sh`, which initializes both submodules, resolves LFS files,
and verifies the required runtime artifacts are present. Future data updates
should land in the data repo first, then the main repo should update the
submodule pointer and validation docs in the same engine release commit.

For iOS, `obadh-ios` should bundle only the runtime artifacts it uses:
autocorrect FSTs, the c64 n-gram, the generator manifest, and a compiled Core
ML model. Corpora, raw TSVs, training checkpoints, and builder outputs should
not ship inside the keyboard target.

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

For WASM autosuggest, `suggestSession` and `suggestSessionCandidates` use the
current live session context. `commitTokenId`, `commitToken`, and
`commitUnknown` advance that context and update the bounded personal overlay
when the committed word is eligible. `exportPersonalSnapshot` returns the bytes
that a host should persist for this user; `importPersonalSnapshot` restores
those bytes and validates that they belong to the loaded vocabulary.
`clearSession` clears only the recent context. `clearPersonal` clears the
learned local dictionary.

Local playground workflow:

```bash
./build.sh wasm
npm --prefix www run serve
```

`./build.sh dev` runs the Tailwind watcher plus the same lightweight `www/`
server. The playground server is intentionally outside the Rust crate dependency
graph.

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
| autosuggest ngram artifact | `25,195,978` bytes |
| autosuggest artifact fingerprint | `381a5b7821e7c187` |
| autosuggest c64 candidate artifact | `29,486,274` bytes |
| autosuggest c64 fingerprint | `b311c36a29c4579b` |
| autosuggest INT8 generator | `18,492,708` bytes |
| autosuggest Core ML generator package | `17,668,804` bytes |
| autosuggest vocab | `1,058,854` bytes |
| autosuggest native model load sample | `~1.10-2.16 ms` |
| autosuggest native context lookup sample | `~0.185 us` |
| autosuggest c64 candidate-input sample | `~0.83 us` |
| autosuggest generator scored-union handoff | `~14.38 us` release, personal-aware |
| autosuggest Core ML generator sample | `~459 us` |
| autosuggest native session lookup sample | `~0.188 us` |
| autosuggest native text context lookup sample | `~0.463 us` |

Autocorrect CLI process timings are not reported as keyboard latency because
process startup dominates those measurements. Keyboard-time performance should
be measured inside loaded runtimes.

## Project Layout

```text
src/engine/                 deterministic tokenizer/transliterator
src/definitions/            compiled rule tables
src/autocorrect/            FST candidate generation and ranking primitives
src/autosuggest/            ngram runtime, personal overlay, neural handoff
src/wasm/                   WebAssembly bindings
src/bin/                    CLI binaries
data/rules/                 documented deterministic rule sources
data/autocorrect/           direct data submodule: lexicon TSVs and FSTs
data/autosuggest/           direct data submodule: corpus, vocab, models
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
cargo test --features cli
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm --lib
cargo bench --bench hot_path --no-run
cargo publish --dry-run
./build.sh dist
git status --short
```

For a tagged release, bump the Cargo/npm versions together, rebuild `docs/`,
commit source plus generated artifacts, push, publish the crate, then tag the
exact published commit.
