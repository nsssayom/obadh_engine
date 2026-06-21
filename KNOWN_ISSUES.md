# Known Issues

This file tracks current limitations for Obadh v0.3.0. It should not list behavior that is already fixed or behavior that belongs only to historical experiments.

## Deterministic Core

1. **Broader Rule-Corpus Validation**

   The core has regression coverage for vowels, consonants, numerals, punctuation, explicit hasant notation, reph, phola forms, valid conjunct filtering, case fallback, mixed-script boundaries, and documented rule probes. The remaining gap is scale: the deliberate input corpus should grow into a larger matrix of linguistically meaningful Roman patterns so awkward but rule-correct spellings are found early.

2. **Conjunct Boundary Ergonomics**

   Implicit conjuncts are filtered through compiled valid-conjunct data, and explicit `,,` remains the deliberate boundary tool. Some legal but rare cluster cases still need more corpus-backed review to decide whether the default path should be implicit composition or explicit user signaling.

3. **Alias Admission Discipline**

   Accepted aliases must continue to be justified by Obadh's own phonetic, orthographic, or ergonomic contract. External keyboard behavior is comparison data only. Broad aliases that weaken deliberate typing should stay out of the core.

## Autocorrect Workbench

1. **Lexicon Coverage**

   The FST candidate generator can only surface words present in the runtime lexicon or loanword lexicon. More curated Bangla sources are still needed before production accuracy can be judged seriously.

2. **Source Weighting**

   The unified lexicon currently combines curated EPUB, Wikipedia, newspaper, and loanword sources. Source weighting is still basic; future rebuilds should keep high-quality formal and literary sources from being drowned out by noisy or register-specific corpus counts.

3. **Ranking Calibration**

   The v0.3.0 ranker is intentionally conservative and explainable: channel priors, weighted Bangla-unit edit cost, unigram frequency, bounded Roman repairs, suffix/prefix completions, and loanword lookup. Contextual ranking and personalization are future layers, not part of the deterministic core.

4. **Evaluation Migration**

   `suggest-fst` is the production runtime path. Older compact `.lex` evaluation/export flows still exist for compatibility and tests, but serious retrieval evaluation should move fully onto the FST candidate generator.

5. **Keyboard-Time Performance Profiling**

   Native CLI process timings include startup and are not a substitute for loaded keyboard-runtime latency. The next profiling pass should measure the loaded FST path inside the long-lived runtime, especially on mobile-class hardware and WASM.

## Playground

1. **Mobile Interaction Validation**

   The v0.3.0 mobile composer is substantially cleaner, but it still needs hands-on testing on real mobile browsers and iOS keyboard-adjacent constraints.

2. **Debug Surface Scope**

   The inspector now exposes useful live/autocorrect data, but the exact split between user-facing playground controls and deeper developer diagnostics should keep evolving as the autocorrect layer matures.

## Future Work

- Expand the deliberate input probe corpus.
- Add loaded-runtime FST latency benchmarks.
- Improve corpus admission and source weighting.
- Keep the deterministic rule docs synchronized with every accepted alias.
- Add a future neural/contextual reranker only after lexicon retrieval and deterministic behavior are stable.
