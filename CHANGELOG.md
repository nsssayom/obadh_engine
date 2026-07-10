# Changelog

All notable changes to `obadh_engine` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
Semantic Versioning (with the `0.x` caveat that the minor version carries breaking changes).

Releases before `0.7.0` predate this file; see the git history and tags for those.

## [0.7.1]

A patch release: three orthography fixes in the deterministic core, a terminology sweep, and the
test and CI infrastructure that found the bugs. **The public API is unchanged.** Every output that
changes was previously a cluster Bangla does not permit.

### Fixed

- **A vowel after a conjunct, reph or numeral could render as a second kar.**
  `transliterate_word_units_into` carries one flag, `previous_unit_accepts_dependent_vowel`, that
  decides whether the next vowel is a dependent sign or an independent letter. Seven of its sixteen
  match arms never assigned it, so a stale `true` leaked across them from a preceding bare
  consonant: `pxiE` → পক্সি**ে** instead of পক্সিএ, `krrkie` → কর্কি**ে** instead of কর্কিএ, and
  `k1i` → ক১**ি** — a vowel sign hanging off a Bengali digit. The suite already asserted this
  contract for `,,`, `^`, `:`, `ng`, <code>t``</code> and `rr`; conjuncts, reph and numerals were
  exempt only because their arms forgot.
  ([#9](https://github.com/nsssayom/obadh_engine/issues/9))

- **An explicit hasant with no consonant to sit on is now dropped.** A hasant suppresses a
  consonant's *inherent* vowel; after an explicit kar there is nothing left to suppress, so `ka,,`
  emitted কা**্**. The same hole let it land on chandrabindu (`k^,,`), anusvar (`kng,,`), bisarga
  (`k:,,`), khanda ta (<code>kt``,,</code> — already a dead consonant), a numeral (`k1,,`), and
  even another hasant (`rr,,` → র্**্**). The rule is now one predicate: **a hasant attaches only
  to a consonant.** As a standalone marker `,,` still renders on its own, and everything the rule
  sources document keeps working — `k,,` → ক্, `ko,,` → ক্, `kk,,` → ক্ক্, `rr,,ka` → র্কা,
  <code>rrk,,Sh</code> → র্ক্ষ, `kR,,` → কড়্.
  ([#10](https://github.com/nsssayom/obadh_engine/issues/10))

- **`ন্স` (`ns`) is now a conjunct.** It was absent from the inventory, so `laisens` produced
  লাইসেনস instead of লাইসেন্স, and `kansa` gave কানসা instead of কান্সা. The whole `-ence`/
  `-ance`/`-ense` loanword family was affected (ডিফেন্স, ব্যালেন্স, সেন্স, রেসপন্স,
  ফাইন্যান্স), as was `ট্রান্সফার`. It behaves like every other conjunct: adjacency joins and
  the inherent vowel breaks, so `nsa` → ন্সা while `nosa` → নসা, exactly as `kta` → ক্তা and
  `kota` → কতা.

  `ন্স` is a **loanword-only** cluster. Native and tatsama Bangla writes the same sequence with
  anusvar — ধ্বংস, হংস, বংশ — and those are unaffected.

### Changed

- **Scope of the change: roman input where `n` is immediately followed by `s`. Nothing else
  moves.** No other conjunct, cluster, or romanization is affected.

  Within that scope, `ns` now behaves like every other conjunct rather than like a pair the
  engine could not spell. Adjacency joins, and the inherent vowel breaks — the same rule that
  has always governed `amra` → আম্রা vs `amora` → আমরা. So words that take the cluster now come
  out right on their own (`respons` → রেস্পন্স, `TrAnsphar` → ট্রান্সফার), and words that spell
  ন and স separately type the `o` (`anosar` → আনসার, `monosur` → মনসুর, `inosTol` → ইনস্টল,
  `konosarrT` → কনসার্ট).

  Previously ন্স was simply unreachable, so `ansar` → আনসার fell out by default. That default is
  gone, not because the convention changed, but because `ns` finally has a conjunct to form.

- **Bangla terminology replaces Sanskrit, Devanagari and Arabic terms** throughout the source,
  tests and rule docs: `virama` → `hasant` (হসন্ত), `visarga` → `bisarga` (বিসর্গ), `anusvara` →
  `anusvar` (অনুস্বার), `matra` → `kar` (কার), `nukta` → `phota` (ফোটা). In Bangla, মাত্রা is the
  headstroke on a letter, not the vowel sign — that is a কার; and নুক্তা is Arabic (نقطة), while
  the Bangla name for the subscript dot is ফোটা. `chandrabindu`, `khanda ta`, `reph`, `phola` and
  `juktoborno` were already Bangla and are untouched.

  The renamed constants are `pub(crate)` or private, so **the public API does not change**.
  Verified behavior-neutral: 19,110 generated inputs transliterate byte-identically before and
  after. ([#11](https://github.com/nsssayom/obadh_engine/issues/11))

### Added

- **`tests/property_tests.rs`** — seven reference-free properties over a fixed-seed corpus of ~45k
  supported inputs and ~14k inputs that deliberately leave the sanitizer's allowed set. None of
  them needs to know the expected Bengali for a given Roman input, so they hold for inputs nobody
  tabulated: `transliterate` is idempotent on its own output; `transliterate(x)` equals
  `transliterate_tokens(tokenize(x))` on supported input; `transliterate_lenient(x)` equals
  `transliterate(clean(x))`; tokenizer spans reassemble the input; no render path emits a dotted
  circle. Each property was mutation-tested — an injected bug in the token render path, in
  `transliterate_lenient`, and in word-internal numeral rendering each failed exactly one property
  and no others.

- **Structural sweeps in `tests/orthography_tests.rs` and `tests/hasant_tests.rs`.** Over 4,320 and
  7,072 generated inputs respectively, no output may stack two vowel signs, hang one off a numeral,
  or place a hasant on a sign that cannot carry it. Both fail on the pre-fix engine (839/4,320 and
  6,051/7,072), so they are regression nets rather than rubber stamps.

- **CI (`.github/workflows/ci.yml`).** `cargo test`, `cargo test --features cli`, the `wasm32`
  target check and the benchmark compile now run on every push and pull request, with
  `RUSTFLAGS: -D warnings`. Submodules are deliberately not checked out, so CI enforces rather than
  assumes the claim that the suite passes without the Git-LFS data.

- **`tests/juktoborno_corpus_tests.rs`** — an exhaustive conjunct corpus (~20k assertions)
  driven from `data/conjuncts.csv` rather than hand-transcribed. It guards both directions of
  the contract: every inventory conjunct must form word-initially, medially and finally and
  accept every dependent vowel; and every consonant pair *outside* the inventory must **not**
  join on its own (1,023 negative cases), since silent over-production is as much a defect as a
  missing conjunct. It also pins the inherent-vowel break, the `,,` explicit-hasant escape for
  all 1,024 pairs, the Unicode khanda-ta ligature rule (ত renders as ৎ except before
  ত/থ/ন/ব/ম/য/র), and a well-formedness sweep asserting the engine never emits a dotted circle,
  a dangling hasant, a hasant before a kar, a doubled hasant, or a hasant on a non-joining sign.

### Notes

- `data/rules/conjunct.wiki` mirrors the Bengali Wikipedia যুক্তবর্ণ list, which claims its 306
  entries are exhaustive (*"এর বাইরে কোন যুক্তবর্ণ সম্ভবত বাংলায় প্রচলিত নয়"*). That claim does not
  hold for modern loanwords — bn.wikipedia's own article title লাইসেন্স uses `ন্স`. The mirror is
  left byte-identical to upstream; `data/conjuncts.csv` is the engine's inventory and now
  intentionally carries one conjunct the wiki does not. The `wiki ⊆ csv` contract test still holds.

- `transliterate` and `transliterate_tokens` disagree on **unsupported** input: the former is
  strict and returns the whole input unchanged, the latter renders the supported tokens and passes
  the rest through (`ké` → `ké` vs `কé`). The behavior is unchanged in this release and is now
  pinned by a test; resolving it is a public-API decision tracked in
  [#16](https://github.com/nsssayom/obadh_engine/issues/16).

## [0.7.0]

Three grounded recovery channels are added to the **active-word autocorrect layer**
(`FstLexicon` suggest path). Each is data-grounded rather than hand-tuned, runs only when the
typed word is a weak signal, and is precision-capped so it can never demote a word the input
actually spells. The deterministic transliteration hot path is unchanged.

### Added

- **Dropped-vowel ("skeleton") channel.** Recovers words typed with vowels omitted
  (e.g. `krlm` → করলাম) by matching the consonant skeleton. It reads skeleton-mates directly
  out of the existing lexicon FST via a custom `fst::Automaton` — **no second index, no extra
  artifact, no added memory or bundle size**, and it always stays consistent with the lexicon.
  Only identical modern-Bengali phonemes are folded (শ/ষ, জ/য, ণ/ন, ড়/ঢ়), grounded in the
  documented grapheme→IPA inventory. Verified over the full 845k-word vocabulary:
  **72% recall@1 / 95% recall@5** (frequency-weighted).
- **Consonant-confusion channel.** Fixes same/near-sound spelling slips (e.g. মানুশ → মানুষ)
  by substituting a baseline consonant with a near phoneme, cost **graded by Bengali IPA
  articulatory-feature distance** (`src/autocorrect/phoneme.rs`) — not a hand-listed confusion
  set.
- **QWERTY key-slip ("fat-finger") channel.** Corrects a physically adjacent-key slip in the
  roman input before transliteration (e.g. `banhla` → বাংলা, `desj` → দেশ). Grounded in
  **FFitts law** (Bi, Li & Zhai, CHI 2013): a finger's touch lands as a 2-D Gaussian around
  the key, so slip cost is quadratic in key distance with σ ≈ 0.5 key-widths. It fires **only
  when the untouched baseline is not itself a lexicon word**, so a validly-typed word is never
  second-guessed. Measured recall when the channel acts (non-word typo): **94% @1 / 99% @5**,
  at **100% precision** on correctly-typed inputs.
- **`key_slip_repaired_outputs(input, baseline, baseline_frequency, transliterate, is_word)`**
  — public helper that produces lexicon-validated key-slip repaired baselines. It mirrors the
  existing `roman_repaired_outputs`: the caller passes only plumbing closures; every numeric
  parameter (touch σ, neighbour radius, variant cap, scoring, gate) is baked into the engine
  and is **not** a client tunable. Wire it into a suggest pipeline exactly where you already
  build `roman_repaired_outputs` (see `src/bin/obadh_autocorrect/fst_cli.rs` and
  `src/wasm/mod.rs`); the WASM `ObadhAutocorrectWasm.suggest` path already includes it.

### Changed

- **`FstCandidateSource` is now `#[non_exhaustive]`.** Future suggestion channels can add
  variants without a breaking change. Downstream `match` statements on this enum must now
  include a wildcard arm.
- **Recovery channels are subordinated to confident readings.** When a word is produced by
  more than one channel it now ranks by its best score but is *labelled* by the most
  authoritative channel that found it, and the skeleton/confusion channels are capped just
  below the best *confident* candidate (exact word, cheap roman-repair, or exact loanword) —
  so a heuristic guess can never relabel or outrank a word the input actually spells.

### API

- **Added variants** to `FstCandidateSource`: `SkeletonVowelDrop` (`"fst_skeleton_vowel_drop"`)
  and `ConsonantConfusion` (`"fst_consonant_confusion"`). The QWERTY channel surfaces as the
  existing `RomanRepairExact` source with `roman_repair_kind = "qwerty_key_slip"`.
- **Added function**: `key_slip_repaired_outputs` (re-exported from the crate root).
- No existing function signatures were removed or changed.

### Migration

- Adding a wildcard arm (`_ => …`) to any exhaustive `match` on `FstCandidateSource` is the
  only source change downstreams may need. Everything else is additive.
