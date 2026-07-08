# Changelog

All notable changes to `obadh_engine` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
Semantic Versioning (with the `0.x` caveat that the minor version carries breaking changes).

Releases before `0.7.0` predate this file; see the git history and tags for those.

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
