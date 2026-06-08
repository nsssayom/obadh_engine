# Obadh ML Stage

This directory is for models that sit above the deterministic Obadh engine. The
deterministic engine remains the always-available typing path; ML is a local,
low-latency suggestion/reranking layer that must never block text entry.

## Runtime Contract

- Run fully on device. Do not require network access for keyboard inference.
- Keep deterministic Obadh output as the fallback candidate.
- Predict Bengali character or orthographic-piece sequences, not fixed word IDs.
- Run on the active word only. Cache prefix results and debounce where a UI can.
- Treat rejected feature documents as data-quality rows to skip or audit; do not
  silently clean training input.

Targets for the first model:

| Constraint | Target |
| --- | --- |
| Model family | 1-layer BiGRU + CTC |
| Quantized model size | under 1 MB preferred |
| p50 inference | under 0.5 ms per word |
| p95 inference | under 2 ms per word |
| Resident memory | under 4-8 MB |
| iOS deployment | Core ML preferred after benchmarking |
| Portable deployment | ONNX for Rust/desktop/Android experiments |

## Feature Schema

Rust exposes `ObadhEngine::ml_features(text)` and the
`obadh-ml-features` binary. The schema is versioned as:

```text
obadh.ml.features.v0
```

Each word token contains Obadh phonetic units and an expanded CTC slot sequence.
Every unit produces three slots: `before`, `main`, and `after`. This keeps the
input time axis long enough for CTC insertions such as latent vowels without
changing the deterministic core.

Example:

```bash
cargo run --bin obadh-ml-features -- 'aYp'
printf 'aYp\nbiggan\n' | cargo run --bin obadh-ml-features
```

## Data

Do not commit downloaded datasets or generated feature corpora. The ignored data
roots are:

```text
ml/data/raw/
ml/data/processed/
ml/runs/
```

Recommended dataset order:

1. Dakshina Bengali (`bn`) for clean benchmark and evaluation.
2. BanglaTLit-style conversational Bangla sources only after source-specific
   audit, sentence segmentation, deduplication, and licensing review.
3. SKNahin/Kaggle/community datasets only after manual source inspection,
   checksum registration, and admission reports.
4. Future private keyboard telemetry only if it is opt-in, local/privacy-safe,
   and clearly separated from this public training pipeline.

Excluded source:

- Aksharantar must not be used for Obadh's Bengali model. It is too risky for
  Bangladeshi Bangla quality and is not part of the approved data path.

Every non-Dakshina source starts as `candidate` until an audit report proves it
is suitable for the specific model stage. Popularity, row count, or a dataset
card language label is not enough.

## Typical Flow

```bash
# Download Dakshina into ignored local storage.
python ml/scripts/download_dakshina.py --output-dir ml/data/raw --extract

# Build Obadh feature JSONL from the Bengali lexicon split.
python ml/scripts/build_obadh_features.py \
  --dakshina-root ml/data/raw/dakshina_dataset_v1.0 \
  --split train \
  --output ml/data/processed/dakshina_bn_train.features.jsonl \
  --release

# Audit any candidate pair file before feature extraction/training.
python ml/scripts/audit_pairs.py \
  --input candidate.tsv \
  --format tsv \
  --latin-column roman \
  --target-column bangla \
  --source-id candidate_source \
  --mode word \
  --report ml/data/processed/candidate.audit.json
```

Training should only begin after the feature vocabulary, target piece
vocabulary, and evaluation split policy are checked in.
