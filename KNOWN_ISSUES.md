# Known Issues in Obadh Engine

This document tracks current limitations and planned future work for the deterministic Obadh Engine core. It should not list already-fixed behavior as open work.

## Tokenizer Issues

1. **Conjunct Formation Policy**: The tokenizer filters implicit conjuncts through a compiled valid-conjunct trie. This prevents arbitrary consonant fusion, but some legal clusters still need better syllable-boundary policy so the engine can decide when implicit clustering is helpful and when the user should type an explicit `,,` boundary.

2. **Alias Admission**: Common "chh" and uppercase "C"/"Ch"/"Chh"/"CH"/"CHH" aliases are handled for ছ, including c+ছ conjunct aliases. Titlecase and all-caps aspirated digraphs such as "Kh"/"KH", "Gh"/"GH", "Jh"/"JH", "Th"/"TH", "Dh"/"DH", "Ph"/"PH", and "Bh"/"BH" compose through normal consonant, vowel, and canonical conjunct rules. The pronounced "kkh" alias family maps to orthographic ক্ষ alongside "ksh"/"kSh". Future aliases need an Obadh-specific linguistic or usability reason; external keyboard layouts are comparison data, not acceptance criteria.

3. **Complex Rule Handling**: The tokenizer needs more sophisticated rules to handle special cases like:
   - When two consonants should form conjuncts vs. when they should remain separate
   - Proper handling of less common consonant clusters and cluster boundaries

4. **Consonant Cluster Recognition**: Current regression coverage protects representative valid clusters, explicit hasant behavior, reph, phola forms, and anusvara-bounded clusters. Broader implicit-cluster behavior still needs corpus-driven validation against deliberate Roman input patterns.

5. **Duplicate Source Signal**: `data/conjuncts.csv` contains the duplicate Roman key `rrt` for both `র্ত` and `র্ৎ`. The engine keeps `rrt` mapped to `র্ত` and requires the composable signal <code>rrt``</code> for `র্ৎ`; <code>rrT``</code> remains an accepted alias.

## Transliterator Issues

1. **Advanced Orthography Rules**: Some spellings need explicit, deterministic Roman input rather than whole-word exceptions. The engine should prefer composable rules and documented user input patterns over dictionary-style word overrides.

2. **Corpus Validation**: Representative cluster, vowel, hasant, phola, mixed-script, CLI, and WASM cases are covered. The next validation gap is a larger corpus of deliberate Roman input patterns that can expose awkward but rule-correct spellings before higher-level suggestion systems exist.

3. **Documentation Completeness**: The main deliberate-input contract is documented, but every accepted alias should continue to be tied back to a canonical rule signal in user-facing docs.

## Future Work

1. Implement a more linguistically accurate algorithm for forming conjuncts based on Bengali orthography rules.

2. Expand deterministic phonetic and orthographic rules without hardcoded whole-word mappings.

3. Add tooling to audit a larger corpus of deliberate Roman input patterns against the deterministic engine.

4. Maintain and expand the Criterion benchmark suite for tokenizer/transliterator hot paths as new deterministic rules are added.

5. Expand test coverage to ensure all edge cases are handled correctly.

6. Add a phonetic rule system that better matches Bengali orthography's special cases while preserving one canonical deliberate signal wherever possible.

7. Consider implementing explicit normalization passes for documented Roman rule patterns before tokenization.

## Notes

The current version has regression coverage for basic vowel and consonant composition, explicit hasant notation, valid conjunct filtering, phola forms, the non-conjunct `র‌্য` ZWNJ signal, mixed-script preservation, numerals, and the CLI/library path. Runtime vowel, consonant, diacritic, and symbol signals have source-contract tests against their documented rule tables or deliberate-input contract; documented arrow examples in `data/rules/simplified_rules.md` are also checked against the public engine path. Every source `data/conjuncts.csv` Roman conjunct key is checked through the public engine rendering path, with only the documented duplicate `rrt` source conflict handled explicitly; those same source conjunct keys are also checked with canonical dependent vowel signs and explicit `,,` hasant between source components. The direct-rendering path and tokenized debug path share text-boundary predicates and have parity coverage for decimal separators, explicit hasant markers, khanda-ta notation, standalone marks, and mixed-script boundaries. The project also has a Criterion hot-path benchmark target for tokenizer and transliterator rule-stress inputs.

More complex cases involving conjuncts, vowel ambiguity, and deliberate input conventions need broader corpus-driven validation. That validation should expand deterministic rules, not introduce dictionary-style word overrides into the core engine.
