# Transliteration Data Source Policy

The first Obadh ML model must optimize for high-quality Bangladeshi Bangla
typing, mobile latency, and deterministic fallback behavior. No dataset enters
training because it is large or popular. Every source needs provenance,
licensing, row-level audit, deduplication, and split-leakage checks.

## Current Source Decisions

- Dakshina Bengali (`bn`): candidate benchmark. Useful for clean eval and
  limited training; not enough by itself for conversational BN-BD typing.
- BanglaTLit: candidate conversational source. Needs segmentation,
  domain/noise filtering, and row audit before training.
- SKNahin/bengali-transliteration-data: candidate. Small social-media style
  source; needs license confirmation, row audit, and deduplication.
- Kaggle Bangla transliteration FPT dataset: candidate/manual. Requires
  authenticated download, checksum registration, source inspection, and license
  review.
- Aksharantar: excluded. Do not use for this model.

## Source Notes

Dakshina is documented as a 12-language South Asian dataset containing native
Wikipedia text, a romanization lexicon with attested romanizations, and some
sentence parallel data. Its Bengali split is useful as a clean benchmark, but it
is Wikipedia-derived and not enough to represent conversational Bangladeshi
typing alone. The published license is CC BY-SA 4.0.

BanglaTLit is an EMNLP 2024 Findings dataset for back-transliteration of
Romanized Bangla. The project reports 245,727 romanized Bangla samples for
further pre-training and 42,705 paired romanized/Bangla examples. Public samples
show useful BN-BD conversational language, but also sentence-level rows, English
technical/product terms, punctuation noise, spelling variation, and occasional
bad target choices. It should feed a sentence-to-word extraction pipeline, not
the word-level model directly.

SKNahin/bengali-transliteration-data has about 5k rows on Hugging Face and
visible conversational examples. It is relevant but small and noisy; license and
provenance need explicit confirmation before training use.

The Kaggle dataset appears related to BanglaTLit further pre-training. Kaggle
access can require authentication, so it must be manually downloaded into
`ml/data/raw/`, registered with a checksum, and audited before use.

## Admission Gates

Required before training:

1. Source metadata is recorded: URL, license, download method, checksum, and
   intended use.
2. Pair-level audit passes configured thresholds for the model stage.
3. Word-level training rows are single-token pairs after segmentation.
4. Target text is Bengali-script dominant and does not contain unsupported
   foreign-script leakage for the word model.
5. Roman input is Latin-script dominant and does not contain native-script
   leakage.
6. Duplicates and conflicting labels are reported before split assignment.
7. Dev/test sets are source-separated or deduplicated against training.
8. Dakshina dev/test remains clean for benchmark reporting.

The audit is intentionally broad. It checks script, Unicode, punctuation, row
shape, length ratio, duplicates, and obvious domain/markup/code leakage. It does
not assume one specific contamination mode.
