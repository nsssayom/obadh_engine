//! Compact next-word autosuggest runtime.
//!
//! The runtime is deliberately a candidate generator: it retrieves likely next
//! Bengali words from a bounded n-gram artifact with trigram/bigram/unigram
//! backoff. A neural model can rerank this small candidate set later without
//! forcing every keystroke through a full-vocabulary softmax.

mod artifact;
mod lm;

pub use artifact::AutosuggestArtifactError;
pub use lm::{
    AutosuggestCandidate, AutosuggestLm, AutosuggestMetadata, AutosuggestOptions,
    AutosuggestResult, AutosuggestSource, DEFAULT_AUTOSUGGEST_CANDIDATES,
};
