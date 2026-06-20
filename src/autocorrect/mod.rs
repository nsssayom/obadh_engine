//! Runtime autocorrect primitives for Bangla text produced above Obadh.
//!
//! This module implements the deterministic runtime shell for autocorrect:
//! candidate generation, Bangla-aware edit scoring, and conservative ranking.
//! Learned model weights can be added behind the same candidate/ranker boundary
//! later without changing the transliteration hot path.

mod artifact;
mod bangla;
mod char_reranker;
mod edit;
mod lexicon;
mod mlp;
mod ranker;

pub use artifact::LexiconArtifactError;
pub use char_reranker::{
    CharCandidateReranker, CharCandidateRerankerError, CharReplacementPolicy,
    ScoredCorrectionCandidate,
};
pub use edit::{weighted_edit_distance, EditCost};
pub use lexicon::{Lexicon, LexiconEntry, LexiconStats};
pub use mlp::{MlpReranker, MlpRerankerError};
pub use ranker::{
    AutocorrectConfig, AutocorrectDecision, AutocorrectEngine, CandidateFeatures,
    CorrectionCandidate, CorrectionRequest, CorrectionSource, AUTOCORRECT_FEATURE_DIM,
};
