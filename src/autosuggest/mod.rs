//! Compact next-word autosuggest runtime.
//!
//! The runtime is deliberately a candidate generator: it retrieves likely next
//! Bengali words from a bounded n-gram artifact with suffix backoff. v1
//! artifacts use trigram/bigram/unigram rows; v2 artifacts can add fourgram
//! rows. A bounded personal overlay can promote on-device habits without
//! changing the static artifact, and a neural model can rerank this small
//! candidate set later without forcing every keystroke through a full-vocabulary
//! softmax.

mod adaptive;
// `pub(crate)` so the `#[cfg(test)] test_support` fixture builder is reachable
// from crate-level tests (e.g. the C ABI e2e tests). No item is re-exported at
// the crate root, so the public API is unchanged.
pub(crate) mod artifact;
mod generator;
mod lm;
mod open_vocab;
mod rerank;
mod scorer;

#[cfg(feature = "wasm")]
pub(crate) use adaptive::{
    repetition_observation_for_raw_token, session_repetition_guard_pool_limit,
    AutosuggestRepetitionHistory,
};
pub use adaptive::{
    AutosuggestSession, PersonalAutosuggest, PersonalAutosuggestConfig, PersonalAutosuggestContext,
    PersonalAutosuggestError, PersonalAutosuggestSnapshotError, PersonalAutosuggestSuggestion,
    PersonalAutosuggestTextSuggestion, DEFAULT_PERSONAL_AUTOSUGGEST_ENTRIES,
    DEFAULT_PERSONAL_AUTOSUGGEST_MIN_COUNT,
};
pub use artifact::AutosuggestArtifactError;
pub use generator::{
    materialize_generated_candidates_into, materialize_merged_candidates_into,
    materialize_scored_union_candidates_into, merge_static_and_generated_candidates_into,
    scored_union_static_and_generated_candidates_into, AutosuggestGeneratedCandidateId,
    AutosuggestGeneratorCompatibility, AutosuggestGeneratorFile, AutosuggestGeneratorHandoff,
    AutosuggestGeneratorHandoffError, AutosuggestGeneratorManifest,
    AutosuggestGeneratorManifestError, AutosuggestGeneratorMergeOptions,
    AutosuggestGeneratorMergedQuality, AutosuggestGeneratorModel, AutosuggestGeneratorNgram,
    AutosuggestGeneratorQuality, AutosuggestGeneratorQualityMetrics,
    AutosuggestGeneratorRuntimeContract, AutosuggestGeneratorScoredUnionPolicy,
    AutosuggestGeneratorScoredUnionQuality, AutosuggestGeneratorSession,
    AutosuggestMaterializedGeneratedCandidate, AutosuggestMaterializedMergedCandidate,
    AutosuggestMaterializedScoredUnionCandidate, AutosuggestMergedCandidateId,
    AutosuggestMergedCandidateSource, AutosuggestScoredUnionCandidateId,
    AUTOSUGGEST_GENERATOR_COREML_INPUT_DTYPE, AUTOSUGGEST_GENERATOR_MANIFEST_VERSION,
    AUTOSUGGEST_GENERATOR_ONNX_INPUT_DTYPE, AUTOSUGGEST_GENERATOR_PACKAGE_KIND,
    AUTOSUGGEST_GENERATOR_RUNTIME_ROLE, AUTOSUGGEST_GENERATOR_SCORE_DTYPE,
    AUTOSUGGEST_GENERATOR_TOKEN_ID_DTYPE, DEFAULT_AUTOSUGGEST_GENERATOR_LOCKED_STATIC_PREFIX,
};
#[cfg(not(target_arch = "wasm32"))]
pub use generator::{AutosuggestGeneratorAsset, AutosuggestGeneratorAssetReport};
#[cfg(not(target_arch = "wasm32"))]
pub use lm::AutosuggestLoadError;
pub use lm::{
    scorer_candidate_i32s_for_candidates_into, scorer_candidate_ids_for_candidates_into,
    AutosuggestCandidate, AutosuggestCandidateId, AutosuggestCandidatePrior, AutosuggestContext,
    AutosuggestContextPriorMetadata, AutosuggestContextPriorOptions, AutosuggestLm,
    AutosuggestMetadata, AutosuggestModelInfo, AutosuggestOptions, AutosuggestRerankInputMetadata,
    AutosuggestResult, AutosuggestSource, AUTOSUGGEST_ARTIFACT_KIND, AUTOSUGGEST_BOS_I32,
    AUTOSUGGEST_BOS_ID, AUTOSUGGEST_PAD_I32, AUTOSUGGEST_PAD_ID, AUTOSUGGEST_UNK_I32,
    AUTOSUGGEST_UNK_ID, DEFAULT_AUTOSUGGEST_CANDIDATES, MAX_AUTOSUGGEST_CONTEXT_TOKENS,
    MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS,
};
pub use open_vocab::{
    accept_open_vocab_texts_into, merge_static_generated_and_open_vocab_candidates_into,
    validate_open_vocab_text, AutosuggestOpenVocabError, AutosuggestOpenVocabPolicy,
    AutosuggestOpenVocabRejectionKind, AutosuggestOpenVocabValidationReport,
    AutosuggestUnifiedCandidate, AutosuggestUnifiedCandidateKind,
    AutosuggestValidatedTextCandidate, DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TEXT_PENALTY,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TEXT_RANK_PENALTY,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TOKEN_PENALTY,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TOKEN_RANK_PENALTY,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_CANDIDATES, DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_SCALAR_RUN,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_WORD_CHARS, DEFAULT_AUTOSUGGEST_OPEN_VOCAB_OVERLAP_BONUS,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_BONUS,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_LOG_COUNT_SCALE,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_RANK_PENALTY,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_SOURCE_BONUS,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_TEXT_HEAP_BYTES,
};
pub use rerank::{
    rerank_candidate_ids_with_fixed_scores_into, rerank_candidate_ids_with_scores_into,
    AutosuggestRerankError, AutosuggestRerankOptions, AutosuggestScoredCandidateId,
    DEFAULT_AUTOSUGGEST_RERANK_LOCKED_PREFIX, DEFAULT_AUTOSUGGEST_RERANK_RANK_PENALTY,
};
pub use scorer::{
    AutosuggestMaterializedScoredCandidate, AutosuggestScorerCompatibility, AutosuggestScorerFile,
    AutosuggestScorerHandoff, AutosuggestScorerHandoffError, AutosuggestScorerManifest,
    AutosuggestScorerManifestError, AutosuggestScorerModel, AutosuggestScorerNgram,
    AutosuggestScorerQuality, AutosuggestScorerQualityMetrics, AutosuggestScorerRuntimeContract,
    AutosuggestScorerSession, AUTOSUGGEST_SCORER_COREML_INPUT_DTYPE,
    AUTOSUGGEST_SCORER_MANIFEST_VERSION, AUTOSUGGEST_SCORER_ONNX_INPUT_DTYPE,
    AUTOSUGGEST_SCORER_PACKAGE_KIND, AUTOSUGGEST_SCORER_RUNTIME_ROLE,
    AUTOSUGGEST_SCORER_SCORE_DTYPE, AUTOSUGGEST_SCORER_TOKEN_ID_DTYPE,
};
#[cfg(not(target_arch = "wasm32"))]
pub use scorer::{AutosuggestScorerAsset, AutosuggestScorerAssetReport};
