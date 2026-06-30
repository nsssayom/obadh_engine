//! Compact next-word autosuggest runtime.
//!
//! The runtime is deliberately a candidate generator: it retrieves likely next
//! Bengali words from a bounded n-gram artifact with trigram/bigram/unigram
//! backoff. A bounded personal overlay can promote on-device habits without
//! changing the static artifact, and a neural model can rerank this small
//! candidate set later without forcing every keystroke through a full-vocabulary
//! softmax.

mod adaptive;
mod artifact;
mod lm;

pub use adaptive::{
    AutosuggestSession, PersonalAutosuggest, PersonalAutosuggestConfig, PersonalAutosuggestError,
    PersonalAutosuggestSuggestion, DEFAULT_PERSONAL_AUTOSUGGEST_ENTRIES,
    DEFAULT_PERSONAL_AUTOSUGGEST_MIN_COUNT,
};
pub use artifact::AutosuggestArtifactError;
pub use lm::{
    AutosuggestCandidate, AutosuggestContext, AutosuggestLm, AutosuggestMetadata,
    AutosuggestOptions, AutosuggestResult, AutosuggestSource, DEFAULT_AUTOSUGGEST_CANDIDATES,
    MAX_AUTOSUGGEST_CONTEXT_TOKENS,
};
