use super::lm::{
    analyze_context_token, scorer_candidate_i32s_for_candidates_into,
    scorer_candidate_ids_for_candidates_into, AutosuggestCandidate, AutosuggestCandidateId,
    AutosuggestContext, AutosuggestLm, AutosuggestMetadata, AutosuggestOptions,
    AutosuggestRerankInputMetadata, AutosuggestResult, AutosuggestSource, BOS_ID,
    MAX_AUTOSUGGEST_CONTEXT_TOKENS, UNK_ID,
};
use crate::autosuggest::AutosuggestArtifactError;
use std::error::Error;
use std::fmt;
use std::mem;

#[cfg(test)]
use super::lm::PAD_ID;

const PERSONAL_MAGIC: &[u8; 16] = b"OBPERSUGLM_V1\0\0\0";
const PERSONAL_VERSION_V1: u32 = 1;
const PERSONAL_VERSION_V2: u32 = 2;
const PERSONAL_VERSION: u32 = 3;
const PERSONAL_V1_V2_HEADER_LEN: usize = 32;
const PERSONAL_HEADER_LEN: usize = 40;
const PERSONAL_V1_CONTEXT_TOKENS: usize = 2;
const PERSONAL_V1_ENTRY_LEN: usize = 24;
const PERSONAL_ENTRY_LEN: usize = 28;
const PERSONAL_TEXT_ENTRY_HEADER_LEN: usize = 28;
const PERSONAL_INITIAL_ENTRY_CAPACITY: usize = 16;
const PERSONAL_TEXT_INITIAL_ENTRY_CAPACITY: usize = 8;
const PERSONAL_UNIGRAM_CACHE_LIMIT: usize = 16;
const MAX_PERSONAL_CONTEXT_TOKENS: usize = MAX_AUTOSUGGEST_CONTEXT_TOKENS;
const DEFAULT_PERSONAL_TEXT_AUTOSUGGEST_ENTRIES: usize = 512;
const PERSONAL_TEXT_TOKEN_MAX_BYTES: usize = 72;
const PERSONAL_TEXT_TOTAL_MAX_BYTES: usize = 16 * 1024;
const PERSONAL_TEXT_CONTEXT_ID_MARKER: u32 = 0x8000_0000;
const PERSONAL_TEXT_CONTEXT_ID_MASK: u32 = 0x7fff_ffff;
const PERSONAL_TEXT_CONTEXT_HASH_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PERSONAL_TEXT_CONTEXT_HASH_PRIME: u64 = 0x0000_0100_0000_01b3;
const SESSION_REPETITION_GUARD_MIN_NGRAM: usize = 3;
const SESSION_REPETITION_GUARD_MAX_POOL: usize = 64;
const SESSION_REPETITION_HISTORY_TOKENS: usize = 256;

pub const DEFAULT_PERSONAL_AUTOSUGGEST_ENTRIES: usize = 4096;
pub const DEFAULT_PERSONAL_AUTOSUGGEST_MIN_COUNT: u16 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AutosuggestRepetitionHistory {
    ids: [u32; SESSION_REPETITION_HISTORY_TOKENS],
    len: usize,
}

impl AutosuggestRepetitionHistory {
    pub(crate) fn new() -> Self {
        Self {
            ids: [0; SESSION_REPETITION_HISTORY_TOKENS],
            len: 0,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    pub(crate) fn recent_token_ids(&self) -> &[u32] {
        &self.ids[..self.len]
    }

    pub(crate) fn observe_resolved_token(
        &mut self,
        token_id: Option<u32>,
        had_text: bool,
        boundary_after: bool,
    ) {
        match token_id {
            Some(id) if id > UNK_ID => {
                self.push_token_id(id);
                if boundary_after {
                    self.clear();
                }
            }
            Some(_) => self.clear(),
            None if had_text => {
                self.push_token_id(UNK_ID);
                if boundary_after {
                    self.clear();
                }
            }
            None if boundary_after => self.clear(),
            None => {}
        }
    }

    fn push_token_id(&mut self, token_id: u32) {
        if self.len < self.ids.len() {
            self.ids[self.len] = token_id;
            self.len += 1;
        } else {
            self.ids.copy_within(1.., 0);
            self.ids[self.ids.len() - 1] = token_id;
        }
    }
}

pub(crate) fn repetition_observation_for_raw_token<D: AsRef<[u8]>>(
    lm: &AutosuggestLm<D>,
    raw_token: &str,
) -> Result<(Option<u32>, bool, bool), AutosuggestArtifactError> {
    let token = analyze_context_token(raw_token);
    let token_id = match token.text {
        Some(text) => lm.token_id(text)?,
        None => None,
    };
    Ok((token_id, token.text.is_some(), token.boundary_after))
}

/// How strongly a committed word counts as user-established vocabulary.
///
/// The caller classifies the event — an ordinary commit, a rejected correction,
/// an explicit add — which is a UI concern. The engine owns the mapping from
/// class to evidence weight ([`CommitStrength::weight`]), a model concern, so
/// the weights can be retuned centrally without every downstream picking its own
/// scalar. This mirrors why the autocorrect ranking score is not exposed as a
/// raw confidence: a free per-downstream number fragments and stops being
/// explainable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitStrength {
    /// An ordinary commit. Weight 1 — identical to the pre-existing observe path,
    /// so default commits are unchanged.
    Committed,
    /// The user rejected an offered correction to keep this word. Strong intent:
    /// a single event establishes the word past the default suggestion threshold
    /// ([`DEFAULT_PERSONAL_AUTOSUGGEST_MIN_COUNT`]).
    CorrectionRejected,
    /// The user added the word explicitly (e.g. a personal dictionary). Strongest.
    ManuallyAdded,
}

impl CommitStrength {
    /// Evidence weight this strength contributes to a personal entry. `Committed`
    /// is 1, so an ordinary commit behaves exactly as before this method existed.
    /// The larger weights are calibration points, not a public contract — a
    /// downstream's measured frecency can retune them without changing this enum.
    pub const fn weight(self) -> u16 {
        match self {
            CommitStrength::Committed => 1,
            CommitStrength::CorrectionRejected => 3,
            CommitStrength::ManuallyAdded => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersonalAutosuggestConfig {
    pub max_entries: usize,
    pub min_count: u16,
}

impl Default for PersonalAutosuggestConfig {
    fn default() -> Self {
        Self {
            max_entries: DEFAULT_PERSONAL_AUTOSUGGEST_ENTRIES,
            min_count: DEFAULT_PERSONAL_AUTOSUGGEST_MIN_COUNT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersonalAutosuggestSuggestion {
    pub token_id: u32,
    pub context_len: usize,
    pub count: u16,
    pub last_seen: u32,
    pub score: i32,
}

const EMPTY_PERSONAL_SUGGESTION: PersonalAutosuggestSuggestion = PersonalAutosuggestSuggestion {
    token_id: 0,
    context_len: 0,
    count: 0,
    last_seen: 0,
    score: 0,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersonalAutosuggestTextSuggestion {
    pub entry_index: usize,
    pub context_len: usize,
    pub count: u16,
    pub last_seen: u32,
    pub score: i32,
}

const EMPTY_PERSONAL_TEXT_SUGGESTION: PersonalAutosuggestTextSuggestion =
    PersonalAutosuggestTextSuggestion {
        entry_index: usize::MAX,
        context_len: 0,
        count: 0,
        last_seen: 0,
        score: 0,
    };

#[derive(Debug)]
pub struct AutosuggestSession<'lm, D: AsRef<[u8]>> {
    lm: &'lm AutosuggestLm<D>,
    personal: PersonalAutosuggest,
    context: AutosuggestContext,
    personal_context: PersonalAutosuggestContext,
    repetition_history: AutosuggestRepetitionHistory,
    options: AutosuggestOptions,
    personal_scratch: Vec<PersonalAutosuggestSuggestion>,
    personal_text_scratch: Vec<PersonalAutosuggestTextSuggestion>,
    model_scratch: Vec<AutosuggestCandidate<'lm>>,
    model_id_scratch: Vec<AutosuggestCandidateId>,
    candidates: Vec<AutosuggestCandidate<'lm>>,
    id_candidates: Vec<AutosuggestCandidateId>,
}

impl<'lm, D: AsRef<[u8]>> AutosuggestSession<'lm, D> {
    pub fn new(
        lm: &'lm AutosuggestLm<D>,
        personal: PersonalAutosuggest,
        options: AutosuggestOptions,
    ) -> Self {
        let capacity = session_repetition_guard_pool_limit(options.max_candidates);
        Self {
            lm,
            personal,
            context: AutosuggestContext::new(),
            personal_context: PersonalAutosuggestContext::new(),
            repetition_history: AutosuggestRepetitionHistory::new(),
            options,
            personal_scratch: Vec::with_capacity(capacity),
            personal_text_scratch: Vec::with_capacity(capacity),
            model_scratch: Vec::with_capacity(capacity),
            model_id_scratch: Vec::with_capacity(capacity),
            candidates: Vec::with_capacity(capacity),
            id_candidates: Vec::with_capacity(capacity),
        }
    }

    pub fn with_personal_config(
        lm: &'lm AutosuggestLm<D>,
        config: PersonalAutosuggestConfig,
        options: AutosuggestOptions,
    ) -> Self {
        Self::new(lm, PersonalAutosuggest::new(config), options)
    }

    pub fn context(&self) -> AutosuggestContext {
        self.context
    }

    pub fn clear_context(&mut self) {
        self.context.clear();
        self.personal_context.clear();
        self.repetition_history.clear();
        self.clear_cached_suggestions();
    }

    pub fn push_boundary(&mut self) {
        self.context.push_boundary();
        self.personal_context.push_boundary();
        self.repetition_history.clear();
        self.clear_cached_suggestions();
    }

    pub fn personal(&self) -> &PersonalAutosuggest {
        &self.personal
    }

    pub fn personal_mut(&mut self) -> &mut PersonalAutosuggest {
        &mut self.personal
    }

    pub fn replace_personal(&mut self, personal: PersonalAutosuggest) {
        self.personal = personal;
        self.clear_cached_suggestions();
    }

    pub fn try_replace_personal(
        &mut self,
        mut personal: PersonalAutosuggest,
    ) -> Result<(), AutosuggestArtifactError> {
        personal.validate_for_model(self.lm.vocab_size(), self.lm.vocab_fingerprint())?;
        personal.stamp_model_fingerprint(self.lm.vocab_fingerprint());
        self.replace_personal(personal);
        Ok(())
    }

    pub fn import_personal_snapshot(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), PersonalAutosuggestSnapshotError> {
        let personal = PersonalAutosuggest::from_compact_bytes_for_model(
            self.personal.config(),
            bytes,
            self.lm.vocab_size(),
            self.lm.vocab_fingerprint(),
        )?;
        self.replace_personal(personal);
        Ok(())
    }

    pub fn options(&self) -> AutosuggestOptions {
        self.options
    }

    pub fn set_options(&mut self, options: AutosuggestOptions) {
        self.options = options;
        self.ensure_candidate_capacity();
        self.clear_cached_suggestions();
    }

    pub fn candidates(&self) -> &[AutosuggestCandidate<'lm>] {
        &self.candidates
    }

    pub fn candidate_ids(&self) -> &[AutosuggestCandidateId] {
        &self.id_candidates
    }

    pub fn personal_text_suggestions(&self) -> &[PersonalAutosuggestTextSuggestion] {
        &self.personal_text_scratch
    }

    pub fn personal_text_suggestion_text(
        &self,
        suggestion: PersonalAutosuggestTextSuggestion,
    ) -> Option<&str> {
        self.personal.text_suggestion_text(suggestion)
    }

    pub fn commit_token(&mut self, raw_token: &str) -> Result<bool, AutosuggestArtifactError> {
        let (token_id, had_text, boundary_after) =
            repetition_observation_for_raw_token(self.lm, raw_token)?;
        let learned = self
            .personal
            .observe_committed_token_with_personal_context(
                self.lm,
                &mut self.context,
                &mut self.personal_context,
                raw_token,
            )?;
        self.repetition_history
            .observe_resolved_token(token_id, had_text, boundary_after);
        self.clear_cached_suggestions();
        Ok(learned)
    }

    /// [`commit_token`](Self::commit_token) with an explicit [`CommitStrength`].
    /// `CommitStrength::Committed` is identical to `commit_token`.
    pub fn commit_token_with_strength(
        &mut self,
        raw_token: &str,
        strength: CommitStrength,
    ) -> Result<bool, AutosuggestArtifactError> {
        let (token_id, had_text, boundary_after) =
            repetition_observation_for_raw_token(self.lm, raw_token)?;
        let learned = self
            .personal
            .observe_committed_token_with_personal_context_and_strength(
                self.lm,
                &mut self.context,
                &mut self.personal_context,
                raw_token,
                strength,
            )?;
        self.repetition_history
            .observe_resolved_token(token_id, had_text, boundary_after);
        self.clear_cached_suggestions();
        Ok(learned)
    }

    /// Cumulative post-decay evidence that the user has established `word` as
    /// their own vocabulary, checking both the known-vocabulary and
    /// out-of-vocabulary personal paths. Returns 0 for a word the user has never
    /// committed. See [`PersonalAutosuggest::committed_text_weight`].
    pub fn established_weight(&self, word: &str) -> u16 {
        let text_weight = self.personal.committed_text_weight(word);
        let token_weight = match self.lm.token_id(word) {
            Ok(Some(id)) => self.personal.committed_token_weight(id),
            _ => 0,
        };
        text_weight.max(token_weight)
    }

    /// Whether the user has established `word` with at least `min_weight`
    /// post-decay evidence. A downstream protecting learned words from
    /// auto-correction gates on this instead of maintaining a parallel store.
    /// `min_weight` is clamped to at least 1.
    pub fn is_word_established(&self, word: &str, min_weight: u16) -> bool {
        self.established_weight(word) >= min_weight.max(1)
    }

    /// Commit a token ID that was already resolved against this session's LM.
    ///
    /// Keyboard integrations can resolve a committed Bengali token once, then
    /// stay on this path for personalization and future suggestions. `None`
    /// represents a committed unknown token and clears recent context.
    pub fn commit_token_id(
        &mut self,
        token_id: Option<u32>,
        boundary_after: bool,
    ) -> Result<bool, AutosuggestArtifactError> {
        if let Some(id) = token_id {
            self.validate_token_id(id)?;
        }
        let learned = self
            .personal
            .observe_resolved_token_id_with_personal_context(
                &mut self.context,
                &mut self.personal_context,
                token_id,
                boundary_after,
            );
        self.repetition_history.observe_resolved_token(
            token_id,
            token_id.is_some(),
            boundary_after,
        );
        self.clear_cached_suggestions();
        Ok(learned)
    }

    /// Commit a token that is not represented in the autosuggest vocabulary.
    pub fn commit_unknown(&mut self, boundary_after: bool) {
        self.personal
            .observe_resolved_token_id_with_personal_context(
                &mut self.context,
                &mut self.personal_context,
                None,
                boundary_after,
            );
        self.repetition_history
            .observe_resolved_token(None, true, boundary_after);
        self.clear_cached_suggestions();
    }

    pub fn suggest(&mut self) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        self.ensure_candidate_capacity();
        self.personal.suggest_with_lm_for_personal_context_into(
            self.lm,
            self.context,
            self.personal_context,
            self.repetition_history.recent_token_ids(),
            self.options,
            &mut self.personal_scratch,
            &mut self.model_scratch,
            &mut self.candidates,
        )
    }

    pub fn suggest_ids(&mut self) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        self.suggest_ids_with_options(self.options)
    }

    pub fn suggest_personal_text(&mut self) -> AutosuggestMetadata {
        self.ensure_candidate_capacity();
        self.personal.suggest_text_for_personal_context_into(
            self.personal_context,
            self.options.max_candidates,
            &mut self.personal_text_scratch,
        );
        AutosuggestMetadata {
            context_token_count: self.context.token_count(),
            matched_context_token_count: self.context.matched_token_count(),
        }
    }

    /// Build fixed-shape scorer inputs for the current session context.
    ///
    /// This preserves the session's personal/static merge policy and avoids
    /// materializing candidate text before a platform reranker chooses the small
    /// visible set.
    pub fn rerank_input_into(
        &mut self,
        scorer_context_ids: &mut [u32],
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestArtifactError> {
        let scorer_context_token_count = self
            .lm
            .scorer_context_ids_for_context_into(self.context, scorer_context_ids)?;
        let metadata = self.suggest_ids_with_options(self.options)?;

        Ok(AutosuggestRerankInputMetadata {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            scorer_context_token_count,
            candidate_count: self.id_candidates.len(),
        })
    }

    pub(crate) fn rerank_u32_input_with_options_into(
        &mut self,
        options: AutosuggestOptions,
        scorer_context_ids: &mut [u32],
        scorer_candidate_ids: &mut [u32],
        output_candidates: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestArtifactError> {
        let scorer_context_token_count = self
            .lm
            .scorer_context_ids_for_context_into(self.context, scorer_context_ids)?;
        let metadata = self.suggest_ids_with_options_into(options, output_candidates)?;
        scorer_candidate_ids_for_candidates_into(output_candidates, scorer_candidate_ids);

        Ok(AutosuggestRerankInputMetadata {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            scorer_context_token_count,
            candidate_count: output_candidates.len(),
        })
    }

    pub(crate) fn rerank_coreml_input_with_options_into(
        &mut self,
        options: AutosuggestOptions,
        scorer_context_ids: &mut [i32],
        scorer_candidate_ids: &mut [i32],
        output_candidates: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestArtifactError> {
        let scorer_context_token_count = self
            .lm
            .scorer_context_i32s_for_context_into(self.context, scorer_context_ids)?;
        let metadata = self.suggest_ids_with_options_into(options, output_candidates)?;
        scorer_candidate_i32s_for_candidates_into(output_candidates, scorer_candidate_ids)?;

        Ok(AutosuggestRerankInputMetadata {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            scorer_context_token_count,
            candidate_count: output_candidates.len(),
        })
    }

    pub fn estimated_heap_bytes(&self) -> usize {
        self.personal
            .estimated_heap_bytes()
            .saturating_add(
                self.personal_scratch
                    .capacity()
                    .saturating_mul(mem::size_of::<PersonalAutosuggestSuggestion>()),
            )
            .saturating_add(
                self.personal_text_scratch
                    .capacity()
                    .saturating_mul(mem::size_of::<PersonalAutosuggestTextSuggestion>()),
            )
            .saturating_add(
                self.model_scratch
                    .capacity()
                    .saturating_mul(mem::size_of::<AutosuggestCandidate<'lm>>()),
            )
            .saturating_add(
                self.model_id_scratch
                    .capacity()
                    .saturating_mul(mem::size_of::<AutosuggestCandidateId>()),
            )
            .saturating_add(
                self.candidates
                    .capacity()
                    .saturating_mul(mem::size_of::<AutosuggestCandidate<'lm>>()),
            )
            .saturating_add(
                self.id_candidates
                    .capacity()
                    .saturating_mul(mem::size_of::<AutosuggestCandidateId>()),
            )
    }

    pub fn heap_limit_bytes(&self) -> usize {
        let candidate_capacity = session_repetition_guard_pool_limit(self.options.max_candidates);
        self.personal
            .heap_limit_bytes()
            .saturating_add(scratch_heap_bytes::<PersonalAutosuggestSuggestion>(
                candidate_capacity,
                self.personal_scratch.capacity(),
            ))
            .saturating_add(scratch_heap_bytes::<PersonalAutosuggestTextSuggestion>(
                self.options.max_candidates,
                self.personal_text_scratch.capacity(),
            ))
            .saturating_add(scratch_heap_bytes::<AutosuggestCandidate<'lm>>(
                candidate_capacity,
                self.model_scratch.capacity(),
            ))
            .saturating_add(scratch_heap_bytes::<AutosuggestCandidateId>(
                candidate_capacity,
                self.model_id_scratch.capacity(),
            ))
            .saturating_add(scratch_heap_bytes::<AutosuggestCandidate<'lm>>(
                candidate_capacity,
                self.candidates.capacity(),
            ))
            .saturating_add(scratch_heap_bytes::<AutosuggestCandidateId>(
                candidate_capacity,
                self.id_candidates.capacity(),
            ))
    }

    pub fn personal_snapshot_len(&self) -> usize {
        self.personal.compact_snapshot_len()
    }

    pub fn personal_snapshot_limit_bytes(&self) -> usize {
        self.personal.compact_snapshot_limit_bytes()
    }

    pub fn write_personal_snapshot_into(&self, output: &mut Vec<u8>) {
        self.personal
            .write_compact_bytes_with_model_fingerprint_into(output, self.lm.vocab_fingerprint());
    }

    fn ensure_candidate_capacity(&mut self) {
        self.ensure_candidate_capacity_for(session_repetition_guard_pool_limit(
            self.options.max_candidates,
        ));
    }

    fn ensure_candidate_capacity_for(&mut self, max_candidates: usize) {
        let capacity = max_candidates.max(1);
        reserve_to(&mut self.personal_scratch, capacity);
        reserve_to(&mut self.personal_text_scratch, capacity);
        reserve_to(&mut self.model_scratch, capacity);
        reserve_to(&mut self.model_id_scratch, capacity);
        reserve_to(&mut self.candidates, capacity);
        reserve_to(&mut self.id_candidates, capacity);
    }

    fn suggest_ids_with_options(
        &mut self,
        options: AutosuggestOptions,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        self.ensure_candidate_capacity_for(options.max_candidates);
        self.personal.suggest_ids_with_lm_for_personal_context_into(
            self.lm,
            self.context,
            self.personal_context,
            self.repetition_history.recent_token_ids(),
            options,
            &mut self.personal_scratch,
            &mut self.model_id_scratch,
            &mut self.id_candidates,
        )
    }

    fn suggest_ids_with_options_into(
        &mut self,
        options: AutosuggestOptions,
        output: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        self.ensure_candidate_capacity_for(options.max_candidates);
        self.personal.suggest_ids_with_lm_for_personal_context_into(
            self.lm,
            self.context,
            self.personal_context,
            self.repetition_history.recent_token_ids(),
            options,
            &mut self.personal_scratch,
            &mut self.model_id_scratch,
            output,
        )
    }

    fn validate_token_id(&self, token_id: u32) -> Result<(), AutosuggestArtifactError> {
        self.lm.validate_word_token_id(token_id)
    }

    fn clear_cached_suggestions(&mut self) {
        self.candidates.clear();
        self.id_candidates.clear();
        self.personal_text_scratch.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersonalAutosuggestError {
    UnexpectedEof,
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidLayout,
}

impl fmt::Display for PersonalAutosuggestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => f.write_str("personal autosuggest snapshot is truncated"),
            Self::InvalidMagic => f.write_str("personal autosuggest snapshot has invalid magic"),
            Self::UnsupportedVersion(version) => {
                write!(
                    f,
                    "unsupported personal autosuggest snapshot version {version}"
                )
            }
            Self::InvalidLayout => f.write_str("personal autosuggest snapshot layout is invalid"),
        }
    }
}

impl Error for PersonalAutosuggestError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersonalAutosuggestSnapshotError {
    Snapshot(PersonalAutosuggestError),
    Model(AutosuggestArtifactError),
}

impl fmt::Display for PersonalAutosuggestSnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Snapshot(error) => error.fmt(f),
            Self::Model(error) => error.fmt(f),
        }
    }
}

impl Error for PersonalAutosuggestSnapshotError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Snapshot(error) => Some(error),
            Self::Model(error) => Some(error),
        }
    }
}

impl From<PersonalAutosuggestError> for PersonalAutosuggestSnapshotError {
    fn from(error: PersonalAutosuggestError) -> Self {
        Self::Snapshot(error)
    }
}

impl From<AutosuggestArtifactError> for PersonalAutosuggestSnapshotError {
    fn from(error: AutosuggestArtifactError) -> Self {
        Self::Model(error)
    }
}

impl PersonalAutosuggestConfig {
    pub fn heap_limit_bytes(self) -> usize {
        self.max_entries
            .saturating_mul(mem::size_of::<PersonalEntry>())
            .saturating_add(
                DEFAULT_PERSONAL_TEXT_AUTOSUGGEST_ENTRIES
                    .saturating_mul(mem::size_of::<PersonalTextEntry>()),
            )
            .saturating_add(PERSONAL_TEXT_TOTAL_MAX_BYTES)
    }

    pub fn compact_snapshot_limit_bytes(self) -> usize {
        PERSONAL_HEADER_LEN
            .saturating_add(self.max_entries.saturating_mul(PERSONAL_ENTRY_LEN))
            .saturating_add(
                DEFAULT_PERSONAL_TEXT_AUTOSUGGEST_ENTRIES
                    .saturating_mul(PERSONAL_TEXT_ENTRY_HEADER_LEN + PERSONAL_TEXT_TOKEN_MAX_BYTES),
            )
    }
}

#[derive(Debug, Clone)]
pub struct PersonalAutosuggest {
    config: PersonalAutosuggestConfig,
    entries: Vec<PersonalEntry>,
    text_entries: Vec<PersonalTextEntry>,
    unigram_cache: PersonalUnigramCache,
    text_unigram_cache: PersonalTextUnigramCache,
    weakest_index: Option<usize>,
    weakest_text_index: Option<usize>,
    model_fingerprint: u32,
    tick: u32,
    /// Weight applied by the observe leaves for the commit in progress. Transient
    /// (never serialized); defaults to 1 and is set only inside
    /// [`PersonalAutosuggest::with_commit_weight`], which always restores it.
    commit_weight: u16,
}

impl PersonalAutosuggest {
    pub fn new(config: PersonalAutosuggestConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            text_entries: Vec::new(),
            unigram_cache: PersonalUnigramCache::empty(),
            text_unigram_cache: PersonalTextUnigramCache::empty(),
            weakest_index: None,
            weakest_text_index: None,
            model_fingerprint: 0,
            tick: 0,
            commit_weight: 1,
        }
    }

    pub fn config(&self) -> PersonalAutosuggestConfig {
        self.config
    }

    pub fn model_fingerprint(&self) -> u32 {
        self.model_fingerprint
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn text_len(&self) -> usize {
        self.text_entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.text_entries.is_empty()
    }

    pub fn estimated_heap_bytes(&self) -> usize {
        self.entries
            .capacity()
            .saturating_mul(std::mem::size_of::<PersonalEntry>())
            .saturating_add(
                self.text_entries
                    .capacity()
                    .saturating_mul(std::mem::size_of::<PersonalTextEntry>()),
            )
            .saturating_add(
                self.text_entries
                    .iter()
                    .map(|entry| entry.text.capacity())
                    .sum(),
            )
    }

    pub fn heap_limit_bytes(&self) -> usize {
        self.config.heap_limit_bytes()
    }

    pub fn compact_snapshot_len(&self) -> usize {
        PERSONAL_HEADER_LEN
            + self.entries.len() * PERSONAL_ENTRY_LEN
            + self
                .text_entries
                .iter()
                .map(|entry| PERSONAL_TEXT_ENTRY_HEADER_LEN + entry.text.len())
                .sum::<usize>()
    }

    pub fn compact_snapshot_limit_bytes(&self) -> usize {
        self.config.compact_snapshot_limit_bytes()
    }

    pub fn validate_token_ids(&self, vocab_size: usize) -> Result<(), AutosuggestArtifactError> {
        for entry in &self.entries {
            for token_id in entry.context.ids() {
                validate_personal_context_token_id(*token_id, vocab_size)?;
            }
            validate_personal_target_token_id(entry.target_id, vocab_size)?;
        }
        for entry in &self.text_entries {
            for token_id in entry.context.ids() {
                validate_personal_context_token_id(*token_id, vocab_size)?;
            }
        }
        Ok(())
    }

    pub fn validate_for_model(
        &self,
        vocab_size: usize,
        model_fingerprint: u32,
    ) -> Result<(), AutosuggestArtifactError> {
        self.validate_token_ids(vocab_size)?;
        validate_personal_model_fingerprint(self.model_fingerprint, model_fingerprint)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.text_entries.clear();
        self.unigram_cache.clear();
        self.text_unigram_cache.clear();
        self.weakest_index = None;
        self.weakest_text_index = None;
        self.tick = 0;
    }

    pub fn decay_counts(&mut self) {
        for entry in &mut self.entries {
            entry.count /= 2;
        }
        self.entries.retain(|entry| entry.count > 0);
        for entry in &mut self.text_entries {
            entry.count /= 2;
        }
        self.text_entries.retain(|entry| entry.count > 0);
        self.rebuild_unigram_cache();
        self.rebuild_text_unigram_cache();
        self.weakest_index = None;
        self.weakest_text_index = None;
    }

    pub fn from_compact_bytes(
        config: PersonalAutosuggestConfig,
        bytes: &[u8],
    ) -> Result<Self, PersonalAutosuggestError> {
        if bytes.len() < PERSONAL_V1_V2_HEADER_LEN {
            return Err(PersonalAutosuggestError::UnexpectedEof);
        }
        if &bytes[..PERSONAL_MAGIC.len()] != PERSONAL_MAGIC {
            return Err(PersonalAutosuggestError::InvalidMagic);
        }
        let version = read_snapshot_u32(bytes, 16)?;
        if version != PERSONAL_VERSION_V1
            && version != PERSONAL_VERSION_V2
            && version != PERSONAL_VERSION
        {
            return Err(PersonalAutosuggestError::UnsupportedVersion(version));
        }
        if version == PERSONAL_VERSION && bytes.len() < PERSONAL_HEADER_LEN {
            return Err(PersonalAutosuggestError::UnexpectedEof);
        }
        let context_token_slots = personal_snapshot_context_token_slots(version)?;
        let entry_len = personal_snapshot_entry_len(version)?;
        let header_len = personal_snapshot_header_len(version)?;
        let tick = read_snapshot_u32(bytes, 20)?;
        let entry_count = read_snapshot_u32(bytes, 24)? as usize;
        let model_fingerprint = read_snapshot_u32(bytes, 28)?;
        let text_entry_count = if version == PERSONAL_VERSION {
            read_snapshot_u32(bytes, 32)? as usize
        } else {
            0
        };
        if version == PERSONAL_VERSION && read_snapshot_u32(bytes, 36)? != 0 {
            return Err(PersonalAutosuggestError::InvalidLayout);
        }
        let id_entries_len = entry_count
            .checked_mul(entry_len)
            .ok_or(PersonalAutosuggestError::InvalidLayout)?;
        let mut offset = header_len
            .checked_add(id_entries_len)
            .ok_or(PersonalAutosuggestError::InvalidLayout)?;
        if offset > bytes.len() {
            return Err(PersonalAutosuggestError::UnexpectedEof);
        }
        if version != PERSONAL_VERSION && offset != bytes.len() {
            return Err(PersonalAutosuggestError::InvalidLayout);
        }

        let mut entries = Vec::with_capacity(entry_count.min(config.max_entries));
        let mut text_entries = Vec::with_capacity(
            text_entry_count
                .min(DEFAULT_PERSONAL_TEXT_AUTOSUGGEST_ENTRIES)
                .min(PERSONAL_TEXT_INITIAL_ENTRY_CAPACITY),
        );
        let mut max_seen = tick;
        for index in 0..entry_count {
            let entry_offset = header_len + index * entry_len;
            let context_len = read_snapshot_u32(bytes, entry_offset)?;
            if context_len as usize > context_token_slots {
                return Err(PersonalAutosuggestError::InvalidLayout);
            }
            let mut context = PersonalContext::empty();
            context.len = context_len as u8;
            for slot in 0..context_token_slots {
                context.ids[slot] = read_snapshot_u32(bytes, entry_offset + 4 + slot * 4)?;
            }
            let target_offset = entry_offset + 4 + context_token_slots * 4;
            let target_id = read_snapshot_u32(bytes, target_offset)?;
            let count = read_snapshot_u32(bytes, target_offset + 4)?;
            let last_seen = read_snapshot_u32(bytes, target_offset + 8)?;
            if target_id <= UNK_ID || count == 0 || count > u32::from(u16::MAX) {
                return Err(PersonalAutosuggestError::InvalidLayout);
            }
            max_seen = max_seen.max(last_seen);
            entries.push(PersonalEntry {
                context,
                target_id,
                count: count as u16,
                last_seen,
            });
        }
        for _ in 0..text_entry_count {
            let context_len = read_snapshot_u32(bytes, offset)?;
            if context_len as usize > MAX_PERSONAL_CONTEXT_TOKENS {
                return Err(PersonalAutosuggestError::InvalidLayout);
            }
            let mut context = PersonalContext::empty();
            context.len = context_len as u8;
            for slot in 0..MAX_PERSONAL_CONTEXT_TOKENS {
                context.ids[slot] = read_snapshot_u32(bytes, offset + 4 + slot * 4)?;
            }
            let count = read_snapshot_u32(bytes, offset + 16)?;
            let last_seen = read_snapshot_u32(bytes, offset + 20)?;
            let text_len = read_snapshot_u32(bytes, offset + 24)? as usize;
            let text_start = offset
                .checked_add(PERSONAL_TEXT_ENTRY_HEADER_LEN)
                .ok_or(PersonalAutosuggestError::InvalidLayout)?;
            let text_end = text_start
                .checked_add(text_len)
                .ok_or(PersonalAutosuggestError::InvalidLayout)?;
            let text_bytes = bytes
                .get(text_start..text_end)
                .ok_or(PersonalAutosuggestError::UnexpectedEof)?;
            let text = std::str::from_utf8(text_bytes)
                .map_err(|_| PersonalAutosuggestError::InvalidLayout)?;
            if count == 0 || count > u32::from(u16::MAX) || !is_personal_text_token(text) {
                return Err(PersonalAutosuggestError::InvalidLayout);
            }
            max_seen = max_seen.max(last_seen);
            if text_entries.len() < DEFAULT_PERSONAL_TEXT_AUTOSUGGEST_ENTRIES
                && text_entries_total_bytes(&text_entries).saturating_add(text.len())
                    <= PERSONAL_TEXT_TOTAL_MAX_BYTES
            {
                text_entries.push(PersonalTextEntry {
                    context,
                    text: text.to_string(),
                    count: count as u16,
                    last_seen,
                });
            }
            offset = text_end;
        }
        if offset != bytes.len() {
            return Err(PersonalAutosuggestError::InvalidLayout);
        }

        entries.sort_by_key(|entry| {
            (
                std::cmp::Reverse(entry.count),
                std::cmp::Reverse(entry.last_seen),
                entry.target_id,
            )
        });
        entries.truncate(config.max_entries);
        entries.sort_by_key(|entry| (entry.context, entry.target_id));
        entries.dedup_by_key(|entry| (entry.context, entry.target_id));
        text_entries.sort_by_key(|entry| {
            (
                std::cmp::Reverse(entry.count),
                std::cmp::Reverse(entry.last_seen),
                entry.text.clone(),
            )
        });
        text_entries.truncate(DEFAULT_PERSONAL_TEXT_AUTOSUGGEST_ENTRIES);
        text_entries.sort_by(|left, right| {
            left.context
                .cmp(&right.context)
                .then_with(|| left.text.cmp(&right.text))
        });
        text_entries
            .dedup_by(|left, right| left.context == right.context && left.text == right.text);

        let mut personal = Self {
            config,
            entries,
            text_entries,
            unigram_cache: PersonalUnigramCache::empty(),
            text_unigram_cache: PersonalTextUnigramCache::empty(),
            weakest_index: None,
            weakest_text_index: None,
            model_fingerprint,
            tick: max_seen,
            commit_weight: 1,
        };
        personal.rebuild_unigram_cache();
        personal.rebuild_text_unigram_cache();
        Ok(personal)
    }

    pub fn from_compact_bytes_for_model(
        config: PersonalAutosuggestConfig,
        bytes: &[u8],
        vocab_size: usize,
        model_fingerprint: u32,
    ) -> Result<Self, PersonalAutosuggestSnapshotError> {
        let mut personal = Self::from_compact_bytes(config, bytes)?;
        personal.validate_for_model(vocab_size, model_fingerprint)?;
        personal.stamp_model_fingerprint(model_fingerprint);
        Ok(personal)
    }

    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.compact_snapshot_len());
        self.write_compact_bytes_into(&mut bytes);
        bytes
    }

    pub fn write_compact_bytes_into(&self, bytes: &mut Vec<u8>) {
        self.write_compact_bytes_with_model_fingerprint_into(bytes, self.model_fingerprint);
    }

    pub fn write_compact_bytes_with_model_fingerprint_into(
        &self,
        bytes: &mut Vec<u8>,
        model_fingerprint: u32,
    ) {
        bytes.clear();
        reserve_to(bytes, self.compact_snapshot_len());
        bytes.extend_from_slice(PERSONAL_MAGIC);
        write_snapshot_u32(bytes, PERSONAL_VERSION);
        write_snapshot_u32(bytes, self.tick);
        write_snapshot_u32(bytes, self.entries.len() as u32);
        write_snapshot_u32(bytes, model_fingerprint);
        write_snapshot_u32(bytes, self.text_entries.len() as u32);
        write_snapshot_u32(bytes, 0);
        for entry in &self.entries {
            write_snapshot_u32(bytes, u32::from(entry.context.len));
            for slot in 0..MAX_PERSONAL_CONTEXT_TOKENS {
                write_snapshot_u32(bytes, entry.context.ids[slot]);
            }
            write_snapshot_u32(bytes, entry.target_id);
            write_snapshot_u32(bytes, u32::from(entry.count));
            write_snapshot_u32(bytes, entry.last_seen);
        }
        for entry in &self.text_entries {
            write_snapshot_u32(bytes, u32::from(entry.context.len));
            for slot in 0..MAX_PERSONAL_CONTEXT_TOKENS {
                write_snapshot_u32(bytes, entry.context.ids[slot]);
            }
            write_snapshot_u32(bytes, u32::from(entry.count));
            write_snapshot_u32(bytes, entry.last_seen);
            write_snapshot_u32(bytes, entry.text.len() as u32);
            bytes.extend_from_slice(entry.text.as_bytes());
        }
    }

    pub(crate) fn stamp_model_fingerprint(&mut self, model_fingerprint: u32) {
        self.model_fingerprint = model_fingerprint;
    }

    pub fn observe_context_target(&mut self, context: AutosuggestContext, target_id: u32) {
        self.observe_context_target_inner(context, target_id);
    }

    pub fn observe_context_ids_target(&mut self, context_ids: &[u32], target_id: u32) {
        if self.config.max_entries == 0 || target_id <= UNK_ID {
            return;
        }

        self.tick = self.tick.wrapping_add(1);
        self.observe_key(PersonalContext::empty(), target_id);

        let usable = context_ids.len().min(MAX_PERSONAL_CONTEXT_TOKENS);
        for len in 1..=usable {
            self.observe_key(PersonalContext::from_suffix(context_ids, len), target_id);
        }
    }

    pub fn observe_context_text_target(&mut self, context: AutosuggestContext, text: &str) -> bool {
        self.observe_personal_context_text_target(
            personal_context_from_static_context(context),
            text,
        )
    }

    pub fn observe_personal_context_target(
        &mut self,
        context: PersonalAutosuggestContext,
        target_id: u32,
    ) {
        self.observe_personal_context_target_inner(context, target_id);
    }

    pub fn observe_personal_context_text_target(
        &mut self,
        context: PersonalAutosuggestContext,
        text: &str,
    ) -> bool {
        if !is_personal_text_token(text) {
            return false;
        }

        self.tick = self.tick.wrapping_add(1);
        self.observe_text_key(PersonalContext::empty(), text);

        if context.is_sentence_start() {
            self.observe_text_key(PersonalContext::sentence_start(), text);
            return true;
        }

        let context_ids = context.recent_context_ids();
        let usable = context_ids.len().min(MAX_PERSONAL_CONTEXT_TOKENS);
        for len in 1..=usable {
            self.observe_text_key(PersonalContext::from_suffix(context_ids, len), text);
        }
        true
    }

    fn observe_context_target_inner(&mut self, context: AutosuggestContext, target_id: u32) {
        if self.config.max_entries == 0 || target_id <= UNK_ID {
            return;
        }

        self.tick = self.tick.wrapping_add(1);
        self.observe_key(PersonalContext::empty(), target_id);

        if context.is_sentence_start() {
            self.observe_key(PersonalContext::sentence_start(), target_id);
            return;
        }

        let context_ids = context.recent_token_ids();
        let usable = context_ids.len().min(MAX_PERSONAL_CONTEXT_TOKENS);
        for len in 1..=usable {
            self.observe_key(PersonalContext::from_suffix(context_ids, len), target_id);
        }
    }

    fn observe_personal_context_target_inner(
        &mut self,
        context: PersonalAutosuggestContext,
        target_id: u32,
    ) {
        if self.config.max_entries == 0 || target_id <= UNK_ID {
            return;
        }

        self.tick = self.tick.wrapping_add(1);
        self.observe_key(PersonalContext::empty(), target_id);

        if context.is_sentence_start() {
            self.observe_key(PersonalContext::sentence_start(), target_id);
            return;
        }

        let context_ids = context.recent_context_ids();
        let usable = context_ids.len().min(MAX_PERSONAL_CONTEXT_TOKENS);
        for len in 1..=usable {
            self.observe_key(PersonalContext::from_suffix(context_ids, len), target_id);
        }
    }

    pub fn observe_committed_token<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        context: &mut AutosuggestContext,
        raw_token: &str,
    ) -> Result<bool, AutosuggestArtifactError> {
        let mut personal_context = personal_context_from_static_context(*context);
        self.observe_committed_token_with_personal_context(
            lm,
            context,
            &mut personal_context,
            raw_token,
        )
    }

    pub fn observe_committed_token_with_personal_context<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        context: &mut AutosuggestContext,
        personal_context: &mut PersonalAutosuggestContext,
        raw_token: &str,
    ) -> Result<bool, AutosuggestArtifactError> {
        let token = analyze_context_token(raw_token);
        let token_id = match token.text {
            Some(text) => lm.token_id(text)?,
            None => None,
        };

        let learned = match token.text {
            Some(text) => match token_id {
                Some(id) => self.observe_resolved_token_id_with_personal_context(
                    context,
                    personal_context,
                    Some(id),
                    token.boundary_after,
                ),
                None => {
                    let learned =
                        self.observe_personal_context_text_target(*personal_context, text);
                    context.push_unknown();
                    personal_context.push_text_token(text);
                    if token.boundary_after {
                        context.push_boundary();
                        personal_context.push_boundary();
                    }
                    learned
                }
            },
            None => {
                if token.boundary_after {
                    context.push_boundary();
                    personal_context.push_boundary();
                }
                false
            }
        };
        Ok(learned)
    }

    /// [`observe_committed_token`](Self::observe_committed_token) with an explicit
    /// [`CommitStrength`]. `CommitStrength::Committed` is identical to the plain
    /// method.
    pub fn observe_committed_token_with_strength<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        context: &mut AutosuggestContext,
        raw_token: &str,
        strength: CommitStrength,
    ) -> Result<bool, AutosuggestArtifactError> {
        let mut personal_context = personal_context_from_static_context(*context);
        self.observe_committed_token_with_personal_context_and_strength(
            lm,
            context,
            &mut personal_context,
            raw_token,
            strength,
        )
    }

    /// [`observe_committed_token_with_personal_context`](Self::observe_committed_token_with_personal_context)
    /// with an explicit [`CommitStrength`].
    ///
    /// The strength sets the evidence weight the observe leaves apply for this one
    /// commit, then the previous weight is restored — so the weight is a property
    /// of the commit rather than something every observe path must thread.
    pub fn observe_committed_token_with_personal_context_and_strength<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        context: &mut AutosuggestContext,
        personal_context: &mut PersonalAutosuggestContext,
        raw_token: &str,
        strength: CommitStrength,
    ) -> Result<bool, AutosuggestArtifactError> {
        let previous = self.commit_weight;
        self.commit_weight = strength.weight();
        let result = self.observe_committed_token_with_personal_context(
            lm,
            context,
            personal_context,
            raw_token,
        );
        self.commit_weight = previous;
        result
    }

    /// Cumulative post-decay evidence that the user has committed `text` as an
    /// out-of-vocabulary word.
    ///
    /// This is the count of the context-free entry, which every commit bumps
    /// exactly once by the commit's [`CommitStrength`] weight, so it reflects
    /// total weighted commits. Counts decay in place
    /// ([`decay_counts`](Self::decay_counts)), so this is already a *post-decay*
    /// measure — the honest "has the user established this word" signal, not a
    /// raw ever-seen flag that would immunize a one-off typo forever. Returns 0
    /// if the word has no personal text evidence.
    pub fn committed_text_weight(&self, text: &str) -> u16 {
        self.text_entries
            .iter()
            .find(|entry| entry.context.is_empty() && entry.text == text)
            .map_or(0, |entry| entry.count)
    }

    /// Post-decay evidence for a known-vocabulary token committed by the user,
    /// keyed by its LM token ID. The text counterpart is
    /// [`committed_text_weight`](Self::committed_text_weight); prefer the
    /// session-level [`AutosuggestSession::established_weight`] when an LM is
    /// available, since it checks both paths.
    pub fn committed_token_weight(&self, token_id: u32) -> u16 {
        self.entries
            .iter()
            .find(|entry| entry.context.is_empty() && entry.target_id == token_id)
            .map_or(0, |entry| entry.count)
    }

    /// Whether the user has established `text` with at least `min_weight`
    /// post-decay evidence. A downstream protecting names/slang from
    /// auto-correction gates on this. `min_weight` is clamped to at least 1.
    pub fn is_text_established(&self, text: &str, min_weight: u16) -> bool {
        self.committed_text_weight(text) >= min_weight.max(1)
    }

    /// Observe a token ID that has already been resolved by the caller.
    ///
    /// This low-level API does not validate the ID against a model. Prefer
    /// `AutosuggestSession::commit_token_id` when an LM is available.
    pub fn observe_resolved_token_id(
        &mut self,
        context: &mut AutosuggestContext,
        token_id: Option<u32>,
        boundary_after: bool,
    ) -> bool {
        let mut personal_context = personal_context_from_static_context(*context);
        self.observe_resolved_token_id_with_personal_context(
            context,
            &mut personal_context,
            token_id,
            boundary_after,
        )
    }

    pub fn observe_resolved_token_id_with_personal_context(
        &mut self,
        context: &mut AutosuggestContext,
        personal_context: &mut PersonalAutosuggestContext,
        token_id: Option<u32>,
        boundary_after: bool,
    ) -> bool {
        let learned = match token_id {
            Some(id) if id > UNK_ID => {
                self.observe_personal_context_target(*personal_context, id);
                context.push_token_id(Some(id));
                personal_context.push_token_id(Some(id));
                true
            }
            _ => {
                context.push_unknown();
                personal_context.push_unknown();
                false
            }
        };

        if boundary_after {
            context.push_boundary();
            personal_context.push_boundary();
        }

        learned
    }

    pub fn suggest_token_ids_into(
        &self,
        context: AutosuggestContext,
        limit: usize,
        output: &mut Vec<PersonalAutosuggestSuggestion>,
    ) {
        self.suggest_token_ids_for_personal_context_into(
            personal_context_from_static_context(context),
            limit,
            output,
        );
    }

    pub fn suggest_token_ids_for_personal_context_into(
        &self,
        context: PersonalAutosuggestContext,
        limit: usize,
        output: &mut Vec<PersonalAutosuggestSuggestion>,
    ) {
        output.clear();
        let limit = limit.max(1);
        if context.is_sentence_start() {
            self.collect_for_context(PersonalContext::sentence_start(), limit, output);
            if output.len() >= limit {
                return;
            }
        }

        let context_ids = context.recent_context_ids();
        let usable = context_ids.len().min(MAX_PERSONAL_CONTEXT_TOKENS);

        for len in (1..=usable).rev() {
            self.collect_for_context(
                PersonalContext::from_suffix(context_ids, len),
                limit,
                output,
            );
            if output.len() >= limit {
                return;
            }
        }

        self.collect_for_context(PersonalContext::empty(), limit, output);
    }

    pub fn suggest_text_into(
        &self,
        context: AutosuggestContext,
        limit: usize,
        output: &mut Vec<PersonalAutosuggestTextSuggestion>,
    ) {
        self.suggest_text_for_personal_context_into(
            personal_context_from_static_context(context),
            limit,
            output,
        );
    }

    pub fn suggest_text_for_personal_context_into(
        &self,
        context: PersonalAutosuggestContext,
        limit: usize,
        output: &mut Vec<PersonalAutosuggestTextSuggestion>,
    ) {
        output.clear();
        let limit = limit.max(1);
        if context.is_sentence_start() {
            self.collect_text_for_context(PersonalContext::sentence_start(), limit, output);
            if output.len() >= limit {
                return;
            }
        }

        let context_ids = context.recent_context_ids();
        let usable = context_ids.len().min(MAX_PERSONAL_CONTEXT_TOKENS);

        for len in (1..=usable).rev() {
            self.collect_text_for_context(
                PersonalContext::from_suffix(context_ids, len),
                limit,
                output,
            );
            if output.len() >= limit {
                return;
            }
        }

        self.collect_text_for_context(PersonalContext::empty(), limit, output);
    }

    pub fn text_suggestion_text(
        &self,
        suggestion: PersonalAutosuggestTextSuggestion,
    ) -> Option<&str> {
        self.text_entries
            .get(suggestion.entry_index)
            .map(|entry| entry.text.as_str())
    }

    pub fn suggest_with_lm<'a, D: AsRef<[u8]>>(
        &self,
        lm: &'a AutosuggestLm<D>,
        context: AutosuggestContext,
        options: AutosuggestOptions,
    ) -> Result<AutosuggestResult<'a>, AutosuggestArtifactError> {
        let limit = options.max_candidates.max(1);
        let mut personal = Vec::with_capacity(limit);
        let mut model = Vec::with_capacity(limit);
        let mut candidates = Vec::with_capacity(limit);
        let metadata = self.suggest_with_lm_into(
            lm,
            context,
            options,
            &mut personal,
            &mut model,
            &mut candidates,
        )?;
        Ok(AutosuggestResult {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            candidates,
        })
    }

    pub fn suggest_with_lm_into<'a, D: AsRef<[u8]>>(
        &self,
        lm: &'a AutosuggestLm<D>,
        context: AutosuggestContext,
        options: AutosuggestOptions,
        personal_scratch: &mut Vec<PersonalAutosuggestSuggestion>,
        model_scratch: &mut Vec<AutosuggestCandidate<'a>>,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        self.suggest_with_lm_for_personal_context_into(
            lm,
            context,
            personal_context_from_static_context(context),
            context.recent_token_ids(),
            options,
            personal_scratch,
            model_scratch,
            output,
        )
    }

    pub fn suggest_with_lm_for_personal_context_into<'a, D: AsRef<[u8]>>(
        &self,
        lm: &'a AutosuggestLm<D>,
        context: AutosuggestContext,
        personal_context: PersonalAutosuggestContext,
        repetition_context_ids: &[u32],
        options: AutosuggestOptions,
        personal_scratch: &mut Vec<PersonalAutosuggestSuggestion>,
        model_scratch: &mut Vec<AutosuggestCandidate<'a>>,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        output.clear();
        let limit = options.max_candidates.max(1);
        let pool_limit = session_repetition_guard_pool_limit(limit);
        let pool_options = AutosuggestOptions {
            max_candidates: pool_limit,
        };
        self.suggest_token_ids_for_personal_context_into(
            personal_context,
            pool_limit,
            personal_scratch,
        );
        let has_context = has_specific_personal_autosuggest_context(personal_context);

        let metadata = lm.suggest_for_context_into(context, pool_options, model_scratch)?;
        reserve_to(output, limit);
        let context_ids = repetition_context_ids;
        for admit_repeated in [false, true] {
            for suggestion in personal_scratch.iter() {
                if output.len() >= limit {
                    return Ok(metadata);
                }
                if has_context && suggestion.context_len == 0 {
                    continue;
                }
                push_personal_candidate_if_repetition_pass_matches(
                    lm,
                    context_ids,
                    suggestion,
                    admit_repeated,
                    output,
                )?;
            }

            for candidate in model_scratch.iter() {
                if output.len() >= limit {
                    return Ok(metadata);
                }
                push_model_candidate_if_repetition_pass_matches(
                    context_ids,
                    candidate,
                    admit_repeated,
                    output,
                );
            }

            if has_context {
                for suggestion in personal_scratch.iter() {
                    if output.len() >= limit {
                        return Ok(metadata);
                    }
                    if suggestion.context_len > 0 {
                        continue;
                    }
                    push_personal_candidate_if_repetition_pass_matches(
                        lm,
                        context_ids,
                        suggestion,
                        admit_repeated,
                        output,
                    )?;
                }
            }
        }

        Ok(metadata)
    }

    pub fn suggest_ids_with_lm_into<D: AsRef<[u8]>>(
        &self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
        options: AutosuggestOptions,
        personal_scratch: &mut Vec<PersonalAutosuggestSuggestion>,
        model_scratch: &mut Vec<AutosuggestCandidateId>,
        output: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        self.suggest_ids_with_lm_for_personal_context_into(
            lm,
            context,
            personal_context_from_static_context(context),
            context.recent_token_ids(),
            options,
            personal_scratch,
            model_scratch,
            output,
        )
    }

    pub fn suggest_ids_with_lm_for_personal_context_into<D: AsRef<[u8]>>(
        &self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
        personal_context: PersonalAutosuggestContext,
        repetition_context_ids: &[u32],
        options: AutosuggestOptions,
        personal_scratch: &mut Vec<PersonalAutosuggestSuggestion>,
        model_scratch: &mut Vec<AutosuggestCandidateId>,
        output: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        output.clear();
        let limit = options.max_candidates.max(1);
        let pool_limit = session_repetition_guard_pool_limit(limit);
        let pool_options = AutosuggestOptions {
            max_candidates: pool_limit,
        };
        self.suggest_token_ids_for_personal_context_into(
            personal_context,
            pool_limit,
            personal_scratch,
        );
        let has_context = has_specific_personal_autosuggest_context(personal_context);

        let metadata = lm.suggest_ids_for_context_into(context, pool_options, model_scratch)?;
        reserve_to(output, limit);
        let context_ids = repetition_context_ids;
        for admit_repeated in [false, true] {
            for suggestion in personal_scratch.iter() {
                if output.len() >= limit {
                    return Ok(metadata);
                }
                if has_context && suggestion.context_len == 0 {
                    continue;
                }
                push_personal_candidate_id_if_repetition_pass_matches(
                    context_ids,
                    suggestion,
                    admit_repeated,
                    output,
                );
            }

            for candidate in model_scratch.iter().copied() {
                if output.len() >= limit {
                    return Ok(metadata);
                }
                push_model_candidate_id_if_repetition_pass_matches(
                    context_ids,
                    candidate,
                    admit_repeated,
                    output,
                );
            }

            if has_context {
                for suggestion in personal_scratch.iter() {
                    if output.len() >= limit {
                        return Ok(metadata);
                    }
                    if suggestion.context_len > 0 {
                        continue;
                    }
                    push_personal_candidate_id_if_repetition_pass_matches(
                        context_ids,
                        suggestion,
                        admit_repeated,
                        output,
                    );
                }
            }
        }

        Ok(metadata)
    }

    fn observe_key(&mut self, context: PersonalContext, target_id: u32) {
        let mut changed_unigram = None;
        let mut removed_unigram = None;

        let weight = self.commit_weight;
        match self.find_entry(context, target_id) {
            Ok(index) => {
                let entry = &mut self.entries[index];
                entry.count = entry.count.saturating_add(weight);
                entry.last_seen = self.tick;
                if entry.context.is_empty() {
                    changed_unigram = Some(*entry);
                }
                if self.weakest_index == Some(index) {
                    self.weakest_index = None;
                }
            }
            Err(index) => {
                let entry = PersonalEntry {
                    context,
                    target_id,
                    count: weight,
                    last_seen: self.tick,
                };
                if self.entries.len() < self.config.max_entries {
                    self.ensure_entry_capacity_for_insert();
                    self.entries.insert(index, entry);
                    self.weakest_index = None;
                    if context.is_empty() {
                        changed_unigram = Some(entry);
                    }
                } else if let Some(weakest_index) = self.weakest_entry_index() {
                    if entry_precedes(entry, self.entries[weakest_index]) {
                        let removed = self.entries.remove(weakest_index);
                        if removed.context.is_empty() {
                            removed_unigram = Some(removed.target_id);
                        }
                        self.insert_entry_sorted(entry);
                        self.weakest_index = None;
                        if context.is_empty() {
                            changed_unigram = Some(entry);
                        }
                    }
                }
            }
        }

        if let Some(token_id) = removed_unigram {
            self.unigram_cache.remove_token(token_id);
        }
        if let Some(entry) = changed_unigram {
            self.unigram_cache.insert(entry.suggestion());
        }
    }

    fn observe_text_key(&mut self, context: PersonalContext, text: &str) {
        let weight = self.commit_weight;
        match self.find_text_entry(context, text) {
            Ok(index) => {
                let entry = &mut self.text_entries[index];
                entry.count = entry.count.saturating_add(weight);
                entry.last_seen = self.tick;
                if self.weakest_text_index == Some(index) {
                    self.weakest_text_index = None;
                }
            }
            Err(index) => {
                let entry = PersonalTextEntry {
                    context,
                    text: text.to_string(),
                    count: weight,
                    last_seen: self.tick,
                };
                if self.can_insert_text_entry(&entry) {
                    self.ensure_text_entry_capacity_for_insert();
                    self.text_entries.insert(index, entry);
                    self.weakest_text_index = None;
                } else if let Some(weakest_index) = self.weakest_text_entry_index() {
                    if text_entry_precedes(&entry, &self.text_entries[weakest_index])
                        && self.can_replace_text_entry(weakest_index, &entry)
                    {
                        self.text_entries.remove(weakest_index);
                        self.insert_text_entry_sorted(entry);
                        self.weakest_text_index = None;
                    }
                }
            }
        }

        self.rebuild_text_unigram_cache();
    }

    fn collect_for_context(
        &self,
        context: PersonalContext,
        limit: usize,
        output: &mut Vec<PersonalAutosuggestSuggestion>,
    ) {
        if context.is_empty() {
            self.unigram_cache
                .collect(limit, self.config.min_count, output);
            if output.len() >= limit || limit <= PERSONAL_UNIGRAM_CACHE_LIMIT {
                return;
            }
            self.collect_for_context_scan(context, limit, output);
            return;
        }

        self.collect_for_context_scan(context, limit, output);
    }

    fn collect_text_for_context(
        &self,
        context: PersonalContext,
        limit: usize,
        output: &mut Vec<PersonalAutosuggestTextSuggestion>,
    ) {
        if context.is_empty() {
            self.text_unigram_cache.collect(
                limit,
                self.config.min_count,
                output,
                &self.text_entries,
            );
            if output.len() >= limit || limit <= PERSONAL_UNIGRAM_CACHE_LIMIT {
                return;
            }
            self.collect_text_for_context_scan(context, limit, output);
            return;
        }

        self.collect_text_for_context_scan(context, limit, output);
    }

    fn collect_for_context_scan(
        &self,
        context: PersonalContext,
        limit: usize,
        output: &mut Vec<PersonalAutosuggestSuggestion>,
    ) {
        let start = self
            .entries
            .partition_point(|entry| entry.context < context);
        let end = start + self.entries[start..].partition_point(|entry| entry.context == context);
        for entry in &self.entries[start..end] {
            if entry.count < self.config.min_count
                || output
                    .iter()
                    .any(|suggestion| suggestion.token_id == entry.target_id)
            {
                continue;
            }
            insert_suggestion_bounded(entry.suggestion(), limit, output);
        }
    }

    fn collect_text_for_context_scan(
        &self,
        context: PersonalContext,
        limit: usize,
        output: &mut Vec<PersonalAutosuggestTextSuggestion>,
    ) {
        let start = self
            .text_entries
            .partition_point(|entry| entry.context < context);
        let end =
            start + self.text_entries[start..].partition_point(|entry| entry.context == context);
        for (relative_index, entry) in self.text_entries[start..end].iter().enumerate() {
            if entry.count < self.config.min_count
                || output.iter().any(|suggestion| {
                    self.text_suggestion_text(*suggestion)
                        .is_some_and(|existing| existing == entry.text)
                })
            {
                continue;
            }
            insert_text_suggestion_bounded(entry.suggestion(start + relative_index), limit, output);
        }
    }

    fn find_entry(&self, context: PersonalContext, target_id: u32) -> Result<usize, usize> {
        self.entries
            .binary_search_by_key(&(context, target_id), |entry| {
                (entry.context, entry.target_id)
            })
    }

    fn find_text_entry(&self, context: PersonalContext, text: &str) -> Result<usize, usize> {
        self.text_entries
            .binary_search_by(|entry| personal_text_key_cmp(entry, context, text))
    }

    fn insert_entry_sorted(&mut self, entry: PersonalEntry) {
        let index = self
            .find_entry(entry.context, entry.target_id)
            .expect_err("personal autosuggest replacement must be a new key");
        self.entries.insert(index, entry);
    }

    fn insert_text_entry_sorted(&mut self, entry: PersonalTextEntry) {
        let index = self
            .find_text_entry(entry.context, &entry.text)
            .expect_err("personal text autosuggest replacement must be a new key");
        self.text_entries.insert(index, entry);
    }

    fn weakest_entry_index(&mut self) -> Option<usize> {
        if let Some(index) = self
            .weakest_index
            .filter(|index| *index < self.entries.len())
        {
            return Some(index);
        }

        let index = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry_strength_key(entry))
            .map(|(index, _)| index);
        self.weakest_index = index;
        index
    }

    fn weakest_text_entry_index(&mut self) -> Option<usize> {
        if let Some(index) = self
            .weakest_text_index
            .filter(|index| *index < self.text_entries.len())
        {
            return Some(index);
        }

        let index = self
            .text_entries
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| text_entry_strength_key(entry))
            .map(|(index, _)| index);
        self.weakest_text_index = index;
        index
    }

    fn can_insert_text_entry(&self, entry: &PersonalTextEntry) -> bool {
        self.text_entries.len() < DEFAULT_PERSONAL_TEXT_AUTOSUGGEST_ENTRIES
            && text_entries_total_bytes(&self.text_entries).saturating_add(entry.text.len())
                <= PERSONAL_TEXT_TOTAL_MAX_BYTES
    }

    fn can_replace_text_entry(&self, replaced_index: usize, entry: &PersonalTextEntry) -> bool {
        let replaced_len = self
            .text_entries
            .get(replaced_index)
            .map_or(0, |replaced| replaced.text.len());
        text_entries_total_bytes(&self.text_entries)
            .saturating_sub(replaced_len)
            .saturating_add(entry.text.len())
            <= PERSONAL_TEXT_TOTAL_MAX_BYTES
    }

    fn ensure_entry_capacity_for_insert(&mut self) {
        if self.entries.len() < self.entries.capacity() || self.config.max_entries == 0 {
            return;
        }

        let target = if self.entries.capacity() == 0 {
            self.config.max_entries.min(PERSONAL_INITIAL_ENTRY_CAPACITY)
        } else {
            self.config
                .max_entries
                .min(self.entries.capacity().saturating_mul(2))
        };
        reserve_to(&mut self.entries, target);
    }

    fn ensure_text_entry_capacity_for_insert(&mut self) {
        if self.text_entries.len() < self.text_entries.capacity() {
            return;
        }

        let target = if self.text_entries.capacity() == 0 {
            DEFAULT_PERSONAL_TEXT_AUTOSUGGEST_ENTRIES.min(PERSONAL_TEXT_INITIAL_ENTRY_CAPACITY)
        } else {
            DEFAULT_PERSONAL_TEXT_AUTOSUGGEST_ENTRIES
                .min(self.text_entries.capacity().saturating_mul(2))
        };
        reserve_to(&mut self.text_entries, target);
    }

    fn rebuild_unigram_cache(&mut self) {
        self.unigram_cache.clear();
        let empty = PersonalContext::empty();
        let start = self.entries.partition_point(|entry| entry.context < empty);
        let end = start + self.entries[start..].partition_point(|entry| entry.context == empty);
        for entry in &self.entries[start..end] {
            self.unigram_cache.insert(entry.suggestion());
        }
    }

    fn rebuild_text_unigram_cache(&mut self) {
        self.text_unigram_cache.clear();
        let empty = PersonalContext::empty();
        let start = self
            .text_entries
            .partition_point(|entry| entry.context < empty);
        let end =
            start + self.text_entries[start..].partition_point(|entry| entry.context == empty);
        for (relative_index, entry) in self.text_entries[start..end].iter().enumerate() {
            self.text_unigram_cache
                .insert(entry.suggestion(start + relative_index), &self.text_entries);
        }
    }
}

impl Default for PersonalAutosuggest {
    fn default() -> Self {
        Self::new(PersonalAutosuggestConfig::default())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PersonalEntry {
    context: PersonalContext,
    target_id: u32,
    count: u16,
    last_seen: u32,
}

impl PersonalEntry {
    fn suggestion(self) -> PersonalAutosuggestSuggestion {
        PersonalAutosuggestSuggestion {
            token_id: self.target_id,
            context_len: self.context.len as usize,
            count: self.count,
            last_seen: self.last_seen,
            score: i32::from(self.count),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PersonalTextEntry {
    context: PersonalContext,
    text: String,
    count: u16,
    last_seen: u32,
}

impl PersonalTextEntry {
    fn suggestion(&self, entry_index: usize) -> PersonalAutosuggestTextSuggestion {
        PersonalAutosuggestTextSuggestion {
            entry_index,
            context_len: self.context.len as usize,
            count: self.count,
            last_seen: self.last_seen,
            score: i32::from(self.count),
        }
    }
}

/// Personal-only context used by the on-device overlay.
///
/// Unlike `AutosuggestContext`, this can carry a compact private ID for a
/// learned Bangla OOV token. Those IDs are never sent to the static LM or
/// scorer; they only index personal entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersonalAutosuggestContext {
    token_count: usize,
    context: PersonalContext,
    at_sentence_start: bool,
}

impl PersonalAutosuggestContext {
    pub fn new() -> Self {
        Self {
            token_count: 0,
            context: PersonalContext::empty(),
            at_sentence_start: true,
        }
    }

    pub fn clear(&mut self) {
        self.token_count = 0;
        self.context = PersonalContext::empty();
        self.at_sentence_start = true;
    }

    pub fn push_boundary(&mut self) {
        self.context = PersonalContext::empty();
        self.at_sentence_start = true;
    }

    pub fn token_count(self) -> usize {
        self.token_count
    }

    pub fn matched_token_count(self) -> usize {
        self.context.len as usize
    }

    pub fn push_word_token_id(&mut self, token_id: u32) {
        self.push_token_id(Some(token_id));
    }

    pub fn push_text(&mut self, text: &str) -> bool {
        self.push_text_token(text)
    }

    pub fn push_unknown_token(&mut self) {
        self.push_unknown();
    }

    fn is_sentence_start(self) -> bool {
        self.at_sentence_start && self.context.is_empty()
    }

    fn recent_context_ids(&self) -> &[u32] {
        self.context.ids()
    }

    fn push_token_id(&mut self, token_id: Option<u32>) {
        self.token_count += 1;
        self.at_sentence_start = false;
        match token_id {
            Some(id) if id > UNK_ID => self.context.push_id(id),
            _ => self.context = PersonalContext::empty(),
        }
    }

    fn push_text_token(&mut self, text: &str) -> bool {
        self.token_count += 1;
        self.at_sentence_start = false;
        let Some(id) = personal_text_context_id(text) else {
            self.context = PersonalContext::empty();
            return false;
        };
        self.context.push_id(id);
        true
    }

    fn push_unknown(&mut self) {
        self.push_token_id(None);
    }
}

impl Default for PersonalAutosuggestContext {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PersonalContext {
    ids: [u32; MAX_PERSONAL_CONTEXT_TOKENS],
    len: u8,
}

impl PersonalContext {
    fn empty() -> Self {
        Self {
            ids: [0; MAX_PERSONAL_CONTEXT_TOKENS],
            len: 0,
        }
    }

    fn sentence_start() -> Self {
        let mut context = Self::empty();
        context.ids[0] = BOS_ID;
        context.len = 1;
        context
    }

    fn from_suffix(ids: &[u32], len: usize) -> Self {
        let len = len.min(MAX_PERSONAL_CONTEXT_TOKENS).min(ids.len());
        let start = ids.len() - len;
        let mut context = Self::empty();
        context.len = len as u8;
        context.ids[..len].copy_from_slice(&ids[start..]);
        context
    }

    fn is_empty(self) -> bool {
        self.len == 0
    }

    fn ids(&self) -> &[u32] {
        &self.ids[..self.len as usize]
    }

    fn push_id(&mut self, id: u32) {
        let len = self.len as usize;
        if len < MAX_PERSONAL_CONTEXT_TOKENS {
            self.ids[len] = id;
            self.len += 1;
            return;
        }
        self.ids.copy_within(1..MAX_PERSONAL_CONTEXT_TOKENS, 0);
        self.ids[MAX_PERSONAL_CONTEXT_TOKENS - 1] = id;
    }
}

#[derive(Debug, Clone)]
struct PersonalUnigramCache {
    len: u8,
    items: [PersonalAutosuggestSuggestion; PERSONAL_UNIGRAM_CACHE_LIMIT],
}

impl PersonalUnigramCache {
    fn empty() -> Self {
        Self {
            len: 0,
            items: [EMPTY_PERSONAL_SUGGESTION; PERSONAL_UNIGRAM_CACHE_LIMIT],
        }
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn insert(&mut self, suggestion: PersonalAutosuggestSuggestion) {
        self.remove_token(suggestion.token_id);

        if self.len as usize >= PERSONAL_UNIGRAM_CACHE_LIMIT
            && self
                .last()
                .is_some_and(|last| !suggestion_precedes(suggestion, last))
        {
            return;
        }

        if self.len as usize >= PERSONAL_UNIGRAM_CACHE_LIMIT {
            self.len -= 1;
        }

        let len = self.len as usize;
        let insert_at = self.items[..len]
            .iter()
            .position(|existing| suggestion_precedes(suggestion, *existing))
            .unwrap_or(len);
        if insert_at < len {
            self.items.copy_within(insert_at..len, insert_at + 1);
        }
        self.items[insert_at] = suggestion;
        self.len += 1;
    }

    fn remove_token(&mut self, token_id: u32) {
        let len = self.len as usize;
        if let Some(position) = self.items[..len]
            .iter()
            .position(|suggestion| suggestion.token_id == token_id)
        {
            self.remove_at(position);
        }
    }

    fn collect(
        &self,
        limit: usize,
        min_count: u16,
        output: &mut Vec<PersonalAutosuggestSuggestion>,
    ) {
        for suggestion in self.items[..self.len as usize].iter().copied() {
            if output.len() >= limit {
                break;
            }
            if suggestion.count < min_count
                || output
                    .iter()
                    .any(|existing| existing.token_id == suggestion.token_id)
            {
                continue;
            }
            output.push(suggestion);
        }
    }

    fn last(&self) -> Option<PersonalAutosuggestSuggestion> {
        self.len
            .checked_sub(1)
            .map(|index| self.items[index as usize])
    }

    fn remove_at(&mut self, position: usize) {
        let len = self.len as usize;
        if position + 1 < len {
            self.items.copy_within(position + 1..len, position);
        }
        self.len -= 1;
        self.items[self.len as usize] = EMPTY_PERSONAL_SUGGESTION;
    }
}

#[derive(Debug, Clone)]
struct PersonalTextUnigramCache {
    len: u8,
    items: [PersonalAutosuggestTextSuggestion; PERSONAL_UNIGRAM_CACHE_LIMIT],
}

impl PersonalTextUnigramCache {
    fn empty() -> Self {
        Self {
            len: 0,
            items: [EMPTY_PERSONAL_TEXT_SUGGESTION; PERSONAL_UNIGRAM_CACHE_LIMIT],
        }
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn insert(
        &mut self,
        suggestion: PersonalAutosuggestTextSuggestion,
        entries: &[PersonalTextEntry],
    ) {
        self.remove_suggestion_text(suggestion, entries);

        if self.len as usize >= PERSONAL_UNIGRAM_CACHE_LIMIT
            && self
                .last()
                .is_some_and(|last| !text_suggestion_precedes(suggestion, last, entries))
        {
            return;
        }

        if self.len as usize >= PERSONAL_UNIGRAM_CACHE_LIMIT {
            self.len -= 1;
        }

        let len = self.len as usize;
        let insert_at = self.items[..len]
            .iter()
            .position(|existing| text_suggestion_precedes(suggestion, *existing, entries))
            .unwrap_or(len);
        if insert_at < len {
            self.items.copy_within(insert_at..len, insert_at + 1);
        }
        self.items[insert_at] = suggestion;
        self.len += 1;
    }

    fn remove_text(&mut self, text: &str, entries: &[PersonalTextEntry]) {
        let len = self.len as usize;
        if let Some(position) = self.items[..len].iter().position(|suggestion| {
            entries
                .get(suggestion.entry_index)
                .is_some_and(|entry| entry.text == text)
        }) {
            self.remove_at(position);
        }
    }

    fn remove_suggestion_text(
        &mut self,
        suggestion: PersonalAutosuggestTextSuggestion,
        entries: &[PersonalTextEntry],
    ) {
        if let Some(text) = entries
            .get(suggestion.entry_index)
            .map(|entry| entry.text.as_str())
        {
            self.remove_text(text, entries);
        }
    }

    fn collect(
        &self,
        limit: usize,
        min_count: u16,
        output: &mut Vec<PersonalAutosuggestTextSuggestion>,
        entries: &[PersonalTextEntry],
    ) {
        for suggestion in self.items[..self.len as usize].iter().copied() {
            if output.len() >= limit {
                break;
            }
            if suggestion.count < min_count
                || output
                    .iter()
                    .any(|existing| same_text_suggestion(*existing, suggestion, entries))
            {
                continue;
            }
            output.push(suggestion);
        }
    }

    fn last(&self) -> Option<PersonalAutosuggestTextSuggestion> {
        self.len
            .checked_sub(1)
            .map(|index| self.items[index as usize])
    }

    fn remove_at(&mut self, position: usize) {
        let len = self.len as usize;
        if position + 1 < len {
            self.items.copy_within(position + 1..len, position);
        }
        self.len -= 1;
        self.items[self.len as usize] = EMPTY_PERSONAL_TEXT_SUGGESTION;
    }
}

fn insert_suggestion_bounded(
    suggestion: PersonalAutosuggestSuggestion,
    limit: usize,
    output: &mut Vec<PersonalAutosuggestSuggestion>,
) {
    if output.len() >= limit
        && output
            .last()
            .is_some_and(|last| !suggestion_precedes(suggestion, *last))
    {
        return;
    }
    if output.len() >= limit {
        output.pop();
    }
    let insert_at = output
        .iter()
        .position(|existing| suggestion_precedes(suggestion, *existing))
        .unwrap_or(output.len());
    output.insert(insert_at, suggestion);
}

fn insert_text_suggestion_bounded(
    suggestion: PersonalAutosuggestTextSuggestion,
    limit: usize,
    output: &mut Vec<PersonalAutosuggestTextSuggestion>,
) {
    if output.len() >= limit
        && output
            .last()
            .is_some_and(|last| !text_suggestion_key_precedes(suggestion, *last))
    {
        return;
    }
    if output.len() >= limit {
        output.pop();
    }
    let insert_at = output
        .iter()
        .position(|existing| text_suggestion_key_precedes(suggestion, *existing))
        .unwrap_or(output.len());
    output.insert(insert_at, suggestion);
}

fn suggestion_precedes(
    left: PersonalAutosuggestSuggestion,
    right: PersonalAutosuggestSuggestion,
) -> bool {
    (
        left.context_len,
        left.count,
        left.last_seen,
        std::cmp::Reverse(left.token_id),
    ) > (
        right.context_len,
        right.count,
        right.last_seen,
        std::cmp::Reverse(right.token_id),
    )
}

fn text_suggestion_key_precedes(
    left: PersonalAutosuggestTextSuggestion,
    right: PersonalAutosuggestTextSuggestion,
) -> bool {
    (
        left.context_len,
        left.count,
        left.last_seen,
        std::cmp::Reverse(left.entry_index),
    ) > (
        right.context_len,
        right.count,
        right.last_seen,
        std::cmp::Reverse(right.entry_index),
    )
}

fn text_suggestion_precedes(
    left: PersonalAutosuggestTextSuggestion,
    right: PersonalAutosuggestTextSuggestion,
    entries: &[PersonalTextEntry],
) -> bool {
    text_suggestion_key_precedes(left, right)
        || (!text_suggestion_key_precedes(right, left)
            && personal_text_for_suggestion(left, entries)
                < personal_text_for_suggestion(right, entries))
}

fn personal_text_for_suggestion<'a>(
    suggestion: PersonalAutosuggestTextSuggestion,
    entries: &'a [PersonalTextEntry],
) -> &'a str {
    entries
        .get(suggestion.entry_index)
        .map_or("", |entry| entry.text.as_str())
}

fn same_text_suggestion(
    left: PersonalAutosuggestTextSuggestion,
    right: PersonalAutosuggestTextSuggestion,
    entries: &[PersonalTextEntry],
) -> bool {
    let left = entries
        .get(left.entry_index)
        .map(|entry| entry.text.as_str());
    let right = entries
        .get(right.entry_index)
        .map(|entry| entry.text.as_str());
    left.is_some() && left == right
}

fn entry_precedes(left: PersonalEntry, right: PersonalEntry) -> bool {
    entry_strength_key(&left) > entry_strength_key(&right)
}

fn entry_strength_key(entry: &PersonalEntry) -> (u16, u32, std::cmp::Reverse<u32>) {
    (
        entry.count,
        entry.last_seen,
        std::cmp::Reverse(entry.target_id),
    )
}

fn text_entry_precedes(left: &PersonalTextEntry, right: &PersonalTextEntry) -> bool {
    text_entry_strength_key(left) > text_entry_strength_key(right)
}

fn text_entry_strength_key(entry: &PersonalTextEntry) -> (u16, u32, std::cmp::Reverse<&str>) {
    (
        entry.count,
        entry.last_seen,
        std::cmp::Reverse(entry.text.as_str()),
    )
}

fn personal_text_key_cmp(
    entry: &PersonalTextEntry,
    context: PersonalContext,
    text: &str,
) -> std::cmp::Ordering {
    entry
        .context
        .cmp(&context)
        .then_with(|| entry.text.as_str().cmp(text))
}

fn text_entries_total_bytes(entries: &[PersonalTextEntry]) -> usize {
    entries.iter().map(|entry| entry.text.len()).sum()
}

fn is_personal_text_token(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= PERSONAL_TEXT_TOKEN_MAX_BYTES
        && text.chars().any(is_bangla_letter)
        && text.chars().all(is_personal_text_char)
}

fn is_bangla_letter(ch: char) -> bool {
    matches!(
        ch,
        '\u{0985}'..='\u{09B9}' | '\u{09CE}' | '\u{09DC}'..='\u{09DF}'
    )
}

fn is_personal_text_char(ch: char) -> bool {
    matches!(ch, '\u{0980}'..='\u{09FF}') && !matches!(ch, '।' | '॥' | '৳')
}

fn read_snapshot_u32(bytes: &[u8], offset: usize) -> Result<u32, PersonalAutosuggestError> {
    let end = offset
        .checked_add(4)
        .ok_or(PersonalAutosuggestError::InvalidLayout)?;
    let slice = bytes
        .get(offset..end)
        .ok_or(PersonalAutosuggestError::UnexpectedEof)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn write_snapshot_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn personal_snapshot_context_token_slots(version: u32) -> Result<usize, PersonalAutosuggestError> {
    match version {
        PERSONAL_VERSION_V1 => Ok(PERSONAL_V1_CONTEXT_TOKENS),
        PERSONAL_VERSION_V2 | PERSONAL_VERSION => Ok(MAX_PERSONAL_CONTEXT_TOKENS),
        other => Err(PersonalAutosuggestError::UnsupportedVersion(other)),
    }
}

fn personal_snapshot_entry_len(version: u32) -> Result<usize, PersonalAutosuggestError> {
    match version {
        PERSONAL_VERSION_V1 => Ok(PERSONAL_V1_ENTRY_LEN),
        PERSONAL_VERSION_V2 | PERSONAL_VERSION => Ok(PERSONAL_ENTRY_LEN),
        other => Err(PersonalAutosuggestError::UnsupportedVersion(other)),
    }
}

fn personal_snapshot_header_len(version: u32) -> Result<usize, PersonalAutosuggestError> {
    match version {
        PERSONAL_VERSION_V1 | PERSONAL_VERSION_V2 => Ok(PERSONAL_V1_V2_HEADER_LEN),
        PERSONAL_VERSION => Ok(PERSONAL_HEADER_LEN),
        other => Err(PersonalAutosuggestError::UnsupportedVersion(other)),
    }
}

fn reserve_to<T>(values: &mut Vec<T>, capacity: usize) {
    if values.capacity() < capacity {
        values.reserve_exact(capacity - values.capacity());
    }
}

fn scratch_heap_bytes<T>(configured_capacity: usize, current_capacity: usize) -> usize {
    configured_capacity
        .max(1)
        .max(current_capacity)
        .saturating_mul(mem::size_of::<T>())
}

fn validate_personal_context_token_id(
    token_id: u32,
    vocab_size: usize,
) -> Result<(), AutosuggestArtifactError> {
    if is_personal_text_context_id(token_id) {
        return Ok(());
    }
    if token_id == 0 || token_id == UNK_ID || token_id as usize >= vocab_size {
        return Err(AutosuggestArtifactError::InvalidTokenId(token_id));
    }
    Ok(())
}

fn validate_personal_target_token_id(
    token_id: u32,
    vocab_size: usize,
) -> Result<(), AutosuggestArtifactError> {
    if token_id <= UNK_ID || token_id as usize >= vocab_size {
        return Err(AutosuggestArtifactError::InvalidTokenId(token_id));
    }
    Ok(())
}

fn has_specific_personal_autosuggest_context(context: PersonalAutosuggestContext) -> bool {
    context.is_sentence_start() || !context.recent_context_ids().is_empty()
}

fn personal_context_from_static_context(context: AutosuggestContext) -> PersonalAutosuggestContext {
    let mut personal_context = PersonalAutosuggestContext::new();
    personal_context.token_count = context.token_count();
    personal_context.at_sentence_start = context.is_sentence_start();
    for token_id in context.recent_token_ids() {
        personal_context.context.push_id(*token_id);
    }
    personal_context
}

pub(crate) fn session_repetition_guard_pool_limit(limit: usize) -> usize {
    limit
        .max(1)
        .saturating_mul(4)
        .min(SESSION_REPETITION_GUARD_MAX_POOL)
        .max(limit.max(1))
}

fn push_personal_candidate_if_repetition_pass_matches<'a, D: AsRef<[u8]>>(
    lm: &'a AutosuggestLm<D>,
    context_ids: &[u32],
    suggestion: &PersonalAutosuggestSuggestion,
    admit_repeated: bool,
    output: &mut Vec<AutosuggestCandidate<'a>>,
) -> Result<(), AutosuggestArtifactError> {
    if !candidate_matches_repetition_pass(context_ids, suggestion.token_id, admit_repeated)
        || output
            .iter()
            .any(|existing| existing.token_id == suggestion.token_id)
    {
        return Ok(());
    }

    output.push(AutosuggestCandidate {
        text: lm.token_text(suggestion.token_id)?,
        token_id: suggestion.token_id,
        source: AutosuggestSource::Personal,
        count: u32::from(suggestion.count),
        score: suggestion.score,
    });
    Ok(())
}

fn push_model_candidate_if_repetition_pass_matches<'a>(
    context_ids: &[u32],
    candidate: &AutosuggestCandidate<'a>,
    admit_repeated: bool,
    output: &mut Vec<AutosuggestCandidate<'a>>,
) {
    if !candidate_matches_repetition_pass(context_ids, candidate.token_id, admit_repeated)
        || output
            .iter()
            .any(|existing| existing.token_id == candidate.token_id)
    {
        return;
    }
    output.push(candidate.clone());
}

fn push_personal_candidate_id_if_repetition_pass_matches(
    context_ids: &[u32],
    suggestion: &PersonalAutosuggestSuggestion,
    admit_repeated: bool,
    output: &mut Vec<AutosuggestCandidateId>,
) {
    if !candidate_matches_repetition_pass(context_ids, suggestion.token_id, admit_repeated)
        || output
            .iter()
            .any(|existing| existing.token_id == suggestion.token_id)
    {
        return;
    }
    output.push(AutosuggestCandidateId {
        token_id: suggestion.token_id,
        source: AutosuggestSource::Personal,
        count: u32::from(suggestion.count),
        score: suggestion.score,
    });
}

fn push_model_candidate_id_if_repetition_pass_matches(
    context_ids: &[u32],
    candidate: AutosuggestCandidateId,
    admit_repeated: bool,
    output: &mut Vec<AutosuggestCandidateId>,
) {
    if !candidate_matches_repetition_pass(context_ids, candidate.token_id, admit_repeated)
        || output
            .iter()
            .any(|existing| existing.token_id == candidate.token_id)
    {
        return;
    }
    output.push(candidate);
}

fn candidate_matches_repetition_pass(
    context_ids: &[u32],
    candidate_id: u32,
    admit_repeated: bool,
) -> bool {
    let repeats = candidate_repeats_recent_ngram(context_ids, candidate_id);
    repeats == admit_repeated
}

pub(crate) fn candidate_repeats_recent_ngram(context_ids: &[u32], candidate_id: u32) -> bool {
    let max_order = (context_ids.len() + 1)
        .min(MAX_PERSONAL_CONTEXT_TOKENS + 1)
        .min(4);
    for order in (SESSION_REPETITION_GUARD_MIN_NGRAM..=max_order).rev() {
        if candidate_repeats_recent_ngram_of_order(context_ids, candidate_id, order) {
            return true;
        }
    }
    false
}

fn candidate_repeats_recent_ngram_of_order(
    context_ids: &[u32],
    candidate_id: u32,
    order: usize,
) -> bool {
    if order < 2 || context_ids.len() + 1 < order {
        return false;
    }
    let prefix_len = order - 1;
    let suffix_prefix = &context_ids[context_ids.len() - prefix_len..];
    context_ids
        .windows(order)
        .any(|window| window[..prefix_len] == *suffix_prefix && window[prefix_len] == candidate_id)
}

fn personal_text_context_id(text: &str) -> Option<u32> {
    if !is_personal_text_token(text) {
        return None;
    }

    let mut hash = PERSONAL_TEXT_CONTEXT_HASH_OFFSET;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PERSONAL_TEXT_CONTEXT_HASH_PRIME);
    }
    let folded = ((hash >> 31) as u32 ^ hash as u32) & PERSONAL_TEXT_CONTEXT_ID_MASK;
    Some(PERSONAL_TEXT_CONTEXT_ID_MARKER | folded.max(1))
}

fn is_personal_text_context_id(token_id: u32) -> bool {
    token_id & PERSONAL_TEXT_CONTEXT_ID_MARKER != 0
}

fn validate_personal_model_fingerprint(
    snapshot_fingerprint: u32,
    model_fingerprint: u32,
) -> Result<(), AutosuggestArtifactError> {
    if snapshot_fingerprint != 0
        && model_fingerprint != 0
        && snapshot_fingerprint != model_fingerprint
    {
        return Err(AutosuggestArtifactError::ModelFingerprintMismatch {
            expected: model_fingerprint,
            actual: snapshot_fingerprint,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autosuggest::artifact::test_support::{build_fixture, Row};

    fn fixture() -> AutosuggestLm<Vec<u8>> {
        let tokens = ["<pad>", "<bos>", "<unk>", "আমি", "আজ", "ভাত", "খাই", "যাই"];
        AutosuggestLm::from_bytes(build_fixture(
            &tokens,
            &[(5, 100, 100), (6, 90, 90), (7, 80, 80)],
            &[
                Row {
                    context: vec![3],
                    candidates: vec![(7, 20, 20), (5, 10, 10)],
                },
                Row {
                    context: vec![3, 4],
                    candidates: vec![(7, 8, 8), (5, 6, 6)],
                },
            ],
        ))
        .expect("fixture should parse")
    }

    fn alternate_fixture() -> AutosuggestLm<Vec<u8>> {
        let tokens = ["<pad>", "<bos>", "<unk>", "আমি", "আজ", "দই", "খাই", "যাই"];
        AutosuggestLm::from_bytes(build_fixture(
            &tokens,
            &[(5, 100, 100), (6, 90, 90), (7, 80, 80)],
            &[Row {
                context: vec![3],
                candidates: vec![(7, 20, 20), (5, 10, 10)],
            }],
        ))
        .expect("alternate fixture should parse")
    }

    fn repetition_fixture() -> AutosuggestLm<Vec<u8>> {
        let tokens = [
            "<pad>",
            "<bos>",
            "<unk>",
            "তিনি",
            "মারা",
            "যান",
            "এবং",
            "এটি",
            "অন্য",
            "একটি",
            "তারা",
        ];
        AutosuggestLm::from_bytes(build_fixture(
            &tokens,
            &[(6, 100, 100), (8, 60, 60), (9, 50, 50)],
            &[
                Row {
                    context: vec![3, 4, 5],
                    candidates: vec![(6, 1000, 1000), (8, 500, 500)],
                },
                Row {
                    context: vec![6, 7],
                    candidates: vec![(9, 1000, 1000), (8, 500, 500)],
                },
            ],
        ))
        .expect("repetition fixture should parse")
    }

    #[test]
    fn personal_suggestion_can_lead_static_model_without_mutating_artifact() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 2,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(3));

        personal.observe_context_target(context, 6);
        personal.observe_context_target(context, 6);

        let result = personal
            .suggest_with_lm(&lm, context, AutosuggestOptions { max_candidates: 4 })
            .unwrap();
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| (candidate.text, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                ("খাই", AutosuggestSource::Personal),
                ("যাই", AutosuggestSource::Bigram),
                ("ভাত", AutosuggestSource::Bigram)
            ]
        );
    }

    #[test]
    fn session_commits_tokens_and_suggests_personal_candidates() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 2,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        for _ in 0..2 {
            session.clear_context();
            assert!(session.commit_token("আমি").unwrap());
            assert!(session.commit_token("খাই").unwrap());
        }

        session.clear_context();
        assert!(session.commit_token("আমি").unwrap());
        let metadata = session.suggest().unwrap();

        assert_eq!(metadata.context_token_count, 1);
        assert_eq!(metadata.matched_context_token_count, 1);
        assert_eq!(
            session
                .candidates()
                .iter()
                .map(|candidate| (candidate.text, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                ("খাই", AutosuggestSource::Personal),
                ("যাই", AutosuggestSource::Bigram),
                ("ভাত", AutosuggestSource::Bigram),
                ("আমি", AutosuggestSource::Personal)
            ]
        );
    }

    #[test]
    fn session_candidate_id_api_matches_text_suggestions() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 2,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        for _ in 0..2 {
            session.clear_context();
            assert!(session.commit_token("আমি").unwrap());
            assert!(session.commit_token("খাই").unwrap());
        }

        session.clear_context();
        assert!(session.commit_token("আমি").unwrap());
        let text_metadata = session.suggest().unwrap();
        let text_candidates = session
            .candidates()
            .iter()
            .map(|candidate| {
                (
                    candidate.token_id,
                    candidate.source,
                    candidate.count,
                    candidate.score,
                )
            })
            .collect::<Vec<_>>();

        let id_metadata = session.suggest_ids().unwrap();

        assert_eq!(id_metadata, text_metadata);
        assert_eq!(
            session
                .candidate_ids()
                .iter()
                .map(|candidate| (
                    candidate.token_id,
                    candidate.source,
                    candidate.count,
                    candidate.score
                ))
                .collect::<Vec<_>>(),
            text_candidates
        );
        assert_eq!(
            session
                .candidate_ids()
                .iter()
                .map(|candidate| lm.materialize_candidate(*candidate).unwrap().text)
                .collect::<Vec<_>>(),
            vec!["খাই", "যাই", "ভাত", "আমি"]
        );
    }

    #[test]
    fn session_demotes_candidate_that_repeats_recent_phrase() {
        let lm = repetition_fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 0,
                min_count: 2,
            },
            AutosuggestOptions { max_candidates: 1 },
        );

        for token in ["তিনি", "মারা", "যান", "এবং", "এটি", "তিনি", "মারা", "যান"]
        {
            assert!(session.commit_token(token).unwrap() || session.personal().is_empty());
        }
        session.suggest().unwrap();

        assert_eq!(
            session
                .candidates()
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["অন্য"]
        );
    }

    #[test]
    fn session_demotes_phrase_repeated_outside_model_context_window() {
        let lm = repetition_fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 0,
                min_count: 2,
            },
            AutosuggestOptions { max_candidates: 1 },
        );

        for token in ["এবং", "এটি", "একটি", "তারা", "এবং", "এটি"]
        {
            session.commit_token(token).unwrap();
        }
        session.suggest().unwrap();

        assert_eq!(
            session
                .candidates()
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["অন্য"]
        );
    }

    #[test]
    fn session_repetition_history_survives_oov_text_boundaries() {
        let lm = repetition_fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 0,
                min_count: 2,
            },
            AutosuggestOptions { max_candidates: 1 },
        );

        for token in ["এবং", "এটি", "একটি", "অচেনা", "তারা", "এবং", "এটি"]
        {
            session.commit_token(token).unwrap();
        }
        session.suggest().unwrap();

        assert_eq!(
            session
                .candidates()
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["অন্য"]
        );
    }

    #[test]
    fn session_keeps_top_candidate_when_it_does_not_repeat_recent_phrase() {
        let lm = repetition_fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 0,
                min_count: 2,
            },
            AutosuggestOptions { max_candidates: 1 },
        );

        for token in ["তিনি", "মারা", "যান"] {
            assert!(session.commit_token(token).unwrap() || session.personal().is_empty());
        }
        session.suggest().unwrap();

        assert_eq!(
            session
                .candidates()
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["এবং"]
        );
    }

    #[test]
    fn session_rerank_input_uses_merged_candidate_ids_without_text_materialization() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 2,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        for _ in 0..2 {
            session.clear_context();
            assert!(session.commit_token("আমি").unwrap());
            assert!(session.commit_token("খাই").unwrap());
        }

        session.clear_context();
        assert!(session.commit_token("আমি").unwrap());
        let mut scorer_context_ids = [99; 4];
        let metadata = session.rerank_input_into(&mut scorer_context_ids).unwrap();

        assert_eq!(scorer_context_ids, [PAD_ID, PAD_ID, PAD_ID, 3]);
        assert_eq!(metadata.context_token_count, 1);
        assert_eq!(metadata.matched_context_token_count, 1);
        assert_eq!(metadata.scorer_context_token_count, 1);
        assert_eq!(metadata.candidate_count, session.candidate_ids().len());
        assert_eq!(
            session
                .candidate_ids()
                .iter()
                .map(|candidate| (
                    lm.materialize_candidate(*candidate).unwrap().text,
                    candidate.source
                ))
                .collect::<Vec<_>>(),
            vec![
                ("খাই", AutosuggestSource::Personal),
                ("যাই", AutosuggestSource::Bigram),
                ("ভাত", AutosuggestSource::Bigram),
                ("আমি", AutosuggestSource::Personal),
            ]
        );
    }

    #[test]
    fn session_commits_resolved_token_ids_on_hot_path() {
        let lm = fixture();
        let ami = lm.token_id("আমি").unwrap();
        let khai = lm.token_id("খাই").unwrap();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 2,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        for _ in 0..2 {
            session.clear_context();
            assert!(session.commit_token_id(ami, false).unwrap());
            assert!(session.commit_token_id(khai, false).unwrap());
        }

        session.clear_context();
        assert!(session.commit_token_id(ami, false).unwrap());
        session.suggest().unwrap();

        assert_eq!(
            session.candidates().first().map(|candidate| candidate.text),
            Some("খাই")
        );
        assert_eq!(session.context().matched_token_count(), 1);
    }

    #[test]
    fn session_rejects_invalid_resolved_token_ids_without_mutating_state() {
        let lm = fixture();
        let invalid = lm.vocab_size() as u32;
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        assert_eq!(
            session.commit_token_id(Some(invalid), false).unwrap_err(),
            AutosuggestArtifactError::InvalidTokenId(invalid)
        );
        assert_eq!(session.context().token_count(), 0);
        assert!(session.personal().is_empty());
    }

    #[test]
    fn session_rejects_reserved_token_ids_without_mutating_state() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        for token_id in 0..=UNK_ID {
            assert_eq!(
                session.commit_token_id(Some(token_id), false).unwrap_err(),
                AutosuggestArtifactError::InvalidTokenId(token_id)
            );
            assert_eq!(session.context().token_count(), 0);
            assert!(session.personal().is_empty());
        }

        session.commit_unknown(false);
        assert_eq!(session.context().token_count(), 1);
        assert!(session.personal().is_empty());
    }

    #[test]
    fn session_unknown_commit_clears_recent_context_without_learning() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        assert!(session
            .commit_token_id(lm.token_id("আমি").unwrap(), false)
            .unwrap());
        session.commit_unknown(false);

        assert_eq!(session.context().token_count(), 2);
        assert_eq!(session.context().matched_token_count(), 0);
        assert_eq!(session.personal().len(), 2);
    }

    #[test]
    fn resolved_token_id_commit_learns_then_honors_sentence_boundary() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 32,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(lm.token_id("আমি").unwrap());

        assert!(personal.observe_resolved_token_id(&mut context, lm.token_id("আজ").unwrap(), true));

        assert_eq!(context.token_count(), 2);
        assert_eq!(context.matched_token_count(), 0);

        let mut query = AutosuggestContext::new();
        query.push_token_id(lm.token_id("আমি").unwrap());
        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(query, 3, &mut suggestions);
        assert_eq!(
            suggestions.first().map(|suggestion| suggestion.token_id),
            lm.token_id("আজ").unwrap()
        );
    }

    #[test]
    fn session_reuses_candidate_buffers_across_suggest_calls() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );
        session.commit_token("আমি").unwrap();
        session.commit_token("খাই").unwrap();
        session.clear_context();
        session.commit_token("আমি").unwrap();

        session.suggest().unwrap();
        let personal_ptr = session.personal_scratch.as_ptr();
        let model_ptr = session.model_scratch.as_ptr();
        let candidates_ptr = session.candidates.as_ptr();
        session.suggest().unwrap();

        assert_eq!(session.personal_scratch.as_ptr(), personal_ptr);
        assert_eq!(session.model_scratch.as_ptr(), model_ptr);
        assert_eq!(session.candidates.as_ptr(), candidates_ptr);
    }

    #[test]
    fn session_heap_limit_is_conservative_after_candidate_buffers_grow() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 8 },
        );
        let grown_limit = session.heap_limit_bytes();
        session.set_options(AutosuggestOptions { max_candidates: 2 });

        assert!(session.heap_limit_bytes() >= grown_limit);
        assert!(session.heap_limit_bytes() >= session.estimated_heap_bytes());
    }

    #[test]
    fn session_snapshot_limit_tracks_personal_store_cap() {
        let lm = fixture();
        let session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 8 },
        );

        assert_eq!(
            session.personal_snapshot_limit_bytes(),
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            }
            .compact_snapshot_limit_bytes()
        );
        assert!(session.personal_snapshot_limit_bytes() >= session.personal_snapshot_len());
    }

    #[test]
    fn session_can_replace_personal_snapshot_without_resetting_context() {
        let lm = fixture();
        let mut trainer = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );
        trainer.commit_token("আমি").unwrap();
        trainer.commit_token("খাই").unwrap();
        let snapshot = trainer.personal().to_compact_bytes();

        let loaded =
            PersonalAutosuggest::from_compact_bytes(trainer.personal().config(), &snapshot)
                .unwrap();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            trainer.personal().config(),
            AutosuggestOptions { max_candidates: 4 },
        );
        session.commit_token("আমি").unwrap();
        session.try_replace_personal(loaded).unwrap();
        session.suggest().unwrap();

        assert_eq!(
            session.candidates().first().map(|candidate| candidate.text),
            Some("খাই")
        );
        assert_eq!(session.context().matched_token_count(), 1);
    }

    #[test]
    fn session_imports_personal_snapshot_bytes_without_resetting_context() {
        let lm = fixture();
        let mut trainer = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );
        trainer.commit_token("আমি").unwrap();
        trainer.commit_token("খাই").unwrap();
        trainer.clear_context();
        trainer.commit_token("আজ").unwrap();
        trainer.commit_token("যাব").unwrap();
        let mut bytes = Vec::new();
        trainer.write_personal_snapshot_into(&mut bytes);

        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 1,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );
        session.commit_token("আমি").unwrap();
        let original_context = session.context();

        session.import_personal_snapshot(&bytes).unwrap();

        assert_eq!(session.context(), original_context);
        assert_eq!(session.personal().config().max_entries, 1);
        assert_eq!(
            session.personal().model_fingerprint(),
            lm.vocab_fingerprint()
        );
        assert_eq!(session.personal().len(), 1);
    }

    #[test]
    fn session_rejects_personal_snapshot_bytes_without_mutating_state() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );
        session.commit_token("আমি").unwrap();
        session.suggest().unwrap();
        let original_context = session.context();
        let original_personal_len = session.personal().len();
        let original_candidates = session.candidates().to_vec();

        assert_eq!(
            session.import_personal_snapshot(b"bad").unwrap_err(),
            PersonalAutosuggestSnapshotError::Snapshot(PersonalAutosuggestError::UnexpectedEof)
        );
        assert_eq!(session.context(), original_context);
        assert_eq!(session.personal().len(), original_personal_len);
        assert_eq!(session.candidates(), original_candidates.as_slice());
    }

    #[test]
    fn session_rejects_snapshot_bytes_from_different_model_fingerprint() {
        let lm = fixture();
        let alternate = alternate_fixture();
        let mut trainer = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );
        trainer.commit_token("আমি").unwrap();
        trainer.commit_token("খাই").unwrap();
        let mut bytes = Vec::new();
        trainer.write_personal_snapshot_into(&mut bytes);

        let mut session = AutosuggestSession::with_personal_config(
            &alternate,
            trainer.personal().config(),
            AutosuggestOptions { max_candidates: 4 },
        );

        assert_eq!(
            session.import_personal_snapshot(&bytes).unwrap_err(),
            PersonalAutosuggestSnapshotError::Model(
                AutosuggestArtifactError::ModelFingerprintMismatch {
                    expected: alternate.vocab_fingerprint(),
                    actual: lm.vocab_fingerprint(),
                }
            )
        );
        assert!(session.personal().is_empty());
    }

    #[test]
    fn session_rejects_foreign_personal_snapshot_without_mutating_state() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );
        session.commit_token("আমি").unwrap();
        session.suggest().unwrap();
        let original_context = session.context();
        let original_personal_len = session.personal().len();
        let original_candidates = session.candidates().to_vec();

        let invalid_token_id = lm.vocab_size() as u32;
        let mut foreign = PersonalAutosuggest::new(session.personal().config());
        foreign.observe_context_ids_target(&[], invalid_token_id);

        assert_eq!(
            session.try_replace_personal(foreign).unwrap_err(),
            AutosuggestArtifactError::InvalidTokenId(invalid_token_id)
        );
        assert_eq!(session.context(), original_context);
        assert_eq!(session.personal().len(), original_personal_len);
        assert_eq!(session.candidates(), original_candidates.as_slice());
    }

    #[test]
    fn session_rejects_snapshot_from_different_model_fingerprint() {
        let lm = fixture();
        let alternate = alternate_fixture();
        assert_eq!(lm.vocab_size(), alternate.vocab_size());
        assert_ne!(lm.vocab_fingerprint(), alternate.vocab_fingerprint());

        let mut trainer = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );
        trainer.commit_token("আমি").unwrap();
        trainer.commit_token("খাই").unwrap();
        let mut bytes = Vec::new();
        trainer.write_personal_snapshot_into(&mut bytes);
        let loaded =
            PersonalAutosuggest::from_compact_bytes(trainer.personal().config(), &bytes).unwrap();

        let mut session = AutosuggestSession::with_personal_config(
            &alternate,
            trainer.personal().config(),
            AutosuggestOptions { max_candidates: 4 },
        );

        assert_eq!(
            session.try_replace_personal(loaded).unwrap_err(),
            AutosuggestArtifactError::ModelFingerprintMismatch {
                expected: alternate.vocab_fingerprint(),
                actual: lm.vocab_fingerprint(),
            }
        );
        assert!(session.personal().is_empty());
    }

    #[test]
    fn session_accepts_legacy_zero_fingerprint_snapshot_and_stamps_model() {
        let lm = fixture();
        let mut trainer = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );
        trainer.commit_token("আমি").unwrap();
        trainer.commit_token("খাই").unwrap();
        let mut bytes = Vec::new();
        trainer.write_personal_snapshot_into(&mut bytes);
        bytes[28..32].copy_from_slice(&0_u32.to_le_bytes());
        let loaded =
            PersonalAutosuggest::from_compact_bytes(trainer.personal().config(), &bytes).unwrap();

        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            trainer.personal().config(),
            AutosuggestOptions { max_candidates: 4 },
        );
        session.try_replace_personal(loaded).unwrap();

        assert_eq!(
            session.personal().model_fingerprint(),
            lm.vocab_fingerprint()
        );
    }

    #[test]
    fn caller_owned_buffers_are_reused_for_hot_path_merge() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(3));
        personal.observe_context_target(context, 6);

        let mut personal_scratch = Vec::with_capacity(4);
        let mut model_scratch = Vec::with_capacity(4);
        let mut output = Vec::with_capacity(4);
        let personal_ptr = personal_scratch.as_ptr();
        let model_ptr = model_scratch.as_ptr();
        let output_ptr = output.as_ptr();

        personal
            .suggest_with_lm_into(
                &lm,
                context,
                AutosuggestOptions { max_candidates: 4 },
                &mut personal_scratch,
                &mut model_scratch,
                &mut output,
            )
            .unwrap();
        personal
            .suggest_with_lm_into(
                &lm,
                context,
                AutosuggestOptions { max_candidates: 4 },
                &mut personal_scratch,
                &mut model_scratch,
                &mut output,
            )
            .unwrap();

        assert_eq!(personal_scratch.as_ptr(), personal_ptr);
        assert_eq!(model_scratch.as_ptr(), model_ptr);
        assert_eq!(output.as_ptr(), output_ptr);
        assert_eq!(output.first().map(|candidate| candidate.text), Some("খাই"));
    }

    #[test]
    fn higher_order_personal_context_backfills_lower_order_context() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(3));
        context.push_token_id(Some(4));

        personal.observe_context_target(context, 6);

        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(context, 3, &mut suggestions);
        assert_eq!(
            suggestions
                .iter()
                .map(|suggestion| (suggestion.token_id, suggestion.context_len))
                .collect::<Vec<_>>(),
            vec![(6, 2)]
        );

        let mut shorter = AutosuggestContext::new();
        shorter.push_token_id(Some(4));
        personal.suggest_token_ids_into(shorter, 3, &mut suggestions);
        assert_eq!(
            suggestions
                .iter()
                .map(|suggestion| (suggestion.token_id, suggestion.context_len))
                .collect::<Vec<_>>(),
            vec![(6, 1)]
        );
    }

    #[test]
    fn personal_context_uses_static_context_width() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(3));
        context.push_token_id(Some(4));
        context.push_token_id(Some(5));

        personal.observe_context_target(context, 6);

        assert!(personal
            .entries
            .iter()
            .all(|entry| entry.context.len as usize <= MAX_PERSONAL_CONTEXT_TOKENS));
        assert!(personal
            .entries
            .iter()
            .any(|entry| entry.context.ids() == [3, 4, 5]));
        assert!(personal
            .entries
            .iter()
            .any(|entry| entry.context.ids() == [4, 5]));

        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(context, 3, &mut suggestions);
        assert_eq!(
            suggestions
                .iter()
                .map(|suggestion| (suggestion.token_id, suggestion.context_len))
                .collect::<Vec<_>>(),
            vec![(6, 3)]
        );
    }

    #[test]
    fn personal_sentence_start_context_is_distinct_from_unknown_fallback() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 2,
        });
        let start = AutosuggestContext::new();
        let mut unknown_after_typing = AutosuggestContext::new();
        unknown_after_typing.push_unknown();

        for _ in 0..2 {
            personal.observe_context_target(start, 4);
        }
        for _ in 0..2 {
            personal.observe_context_target(unknown_after_typing, 6);
        }

        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(start, 4, &mut suggestions);
        assert_eq!(
            suggestions
                .iter()
                .map(|suggestion| (suggestion.token_id, suggestion.context_len))
                .collect::<Vec<_>>(),
            vec![(4, 1), (6, 0)]
        );

        personal.suggest_token_ids_into(unknown_after_typing, 4, &mut suggestions);
        assert!(suggestions
            .iter()
            .all(|suggestion| suggestion.context_len == 0));
        assert_eq!(
            suggestions
                .iter()
                .map(|suggestion| suggestion.token_id)
                .collect::<Vec<_>>(),
            vec![6, 4]
        );
    }

    #[test]
    fn compact_snapshot_accepts_bos_context_but_rejects_pad_or_unknown_context() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 1,
        });

        personal.observe_context_target(AutosuggestContext::new(), 6);
        assert!(personal
            .entries
            .iter()
            .any(|entry| entry.context == PersonalContext::sentence_start()));
        personal.validate_token_ids(lm.vocab_size()).unwrap();

        let bytes = personal.to_compact_bytes();
        let loaded = PersonalAutosuggest::from_compact_bytes(personal.config(), &bytes).unwrap();
        loaded.validate_token_ids(lm.vocab_size()).unwrap();

        let bos_entry_index = loaded
            .entries
            .iter()
            .position(|entry| entry.context == PersonalContext::sentence_start())
            .expect("sentence-start context should be persisted");
        let context_id_offset = PERSONAL_HEADER_LEN + bos_entry_index * PERSONAL_ENTRY_LEN + 4;

        let mut pad_context = bytes.clone();
        pad_context[context_id_offset..context_id_offset + 4].copy_from_slice(&0_u32.to_le_bytes());
        let loaded =
            PersonalAutosuggest::from_compact_bytes(personal.config(), &pad_context).unwrap();
        assert_eq!(
            loaded.validate_token_ids(lm.vocab_size()).unwrap_err(),
            AutosuggestArtifactError::InvalidTokenId(0)
        );

        let mut unknown_context = bytes;
        unknown_context[context_id_offset..context_id_offset + 4]
            .copy_from_slice(&UNK_ID.to_le_bytes());
        let loaded =
            PersonalAutosuggest::from_compact_bytes(personal.config(), &unknown_context).unwrap();
        assert_eq!(
            loaded.validate_token_ids(lm.vocab_size()).unwrap_err(),
            AutosuggestArtifactError::InvalidTokenId(UNK_ID)
        );
    }

    #[test]
    fn empty_context_personal_unigram_cache_is_bounded_and_ranked() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 64,
            min_count: 1,
        });
        for target_id in 5..30 {
            personal.observe_context_ids_target(&[], target_id);
        }
        personal.observe_context_ids_target(&[], 20);
        personal.observe_context_ids_target(&[], 20);

        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(AutosuggestContext::new(), 5, &mut suggestions);

        assert_eq!(
            personal.unigram_cache.len as usize,
            PERSONAL_UNIGRAM_CACHE_LIMIT
        );
        assert_eq!(suggestions.len(), 5);
        assert_eq!(
            suggestions.first().map(|suggestion| suggestion.token_id),
            Some(20)
        );
        assert_eq!(suggestions, personal.unigram_cache.items[..5].to_vec());
    }

    #[test]
    fn empty_context_large_limit_can_scan_beyond_unigram_cache() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 64,
            min_count: 1,
        });
        for target_id in 5..30 {
            personal.observe_context_ids_target(&[], target_id);
        }

        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(AutosuggestContext::new(), 32, &mut suggestions);

        assert_eq!(
            personal.unigram_cache.len as usize,
            PERSONAL_UNIGRAM_CACHE_LIMIT
        );
        assert_eq!(suggestions.len(), 25);
        assert_eq!(
            suggestions
                .iter()
                .map(|suggestion| suggestion.token_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            suggestions.len()
        );
    }

    #[test]
    fn minimum_count_blocks_one_off_personal_noise_until_repeated() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 2,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(3));
        let mut suggestions = Vec::new();

        personal.observe_context_target(context, 6);
        personal.suggest_token_ids_into(context, 3, &mut suggestions);
        assert!(suggestions.is_empty());

        personal.observe_context_target(context, 6);
        personal.suggest_token_ids_into(context, 3, &mut suggestions);
        assert_eq!(
            suggestions.first().map(|suggestion| suggestion.token_id),
            Some(6)
        );
    }

    #[test]
    fn bounded_personal_store_evicts_weakest_old_entries() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 6,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(3));

        personal.observe_context_target(context, 5);
        personal.observe_context_target(context, 6);
        personal.observe_context_target(context, 7);
        assert_eq!(personal.len(), 6);

        personal.observe_context_target(context, 6);
        personal.observe_context_target(context, 8);
        assert_eq!(personal.len(), 6);

        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(context, 5, &mut suggestions);
        assert!(!suggestions
            .iter()
            .any(|suggestion| suggestion.token_id == 5));
        assert_eq!(
            suggestions.first().map(|suggestion| suggestion.token_id),
            Some(6)
        );
    }

    #[test]
    fn full_personal_store_rejects_weaker_new_singletons() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 2,
            min_count: 1,
        });

        for _ in 0..2 {
            personal.observe_context_ids_target(&[], 5);
            personal.observe_context_ids_target(&[], 6);
        }
        personal.observe_context_ids_target(&[], 7);

        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(AutosuggestContext::new(), 4, &mut suggestions);

        assert_eq!(personal.len(), 2);
        assert_eq!(
            suggestions
                .iter()
                .map(|suggestion| suggestion.token_id)
                .collect::<Vec<_>>(),
            vec![6, 5]
        );
        assert!(!suggestions
            .iter()
            .any(|suggestion| suggestion.token_id == 7));
    }

    #[test]
    fn full_personal_store_caches_weakest_entry_after_rejecting_noise() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 2,
            min_count: 1,
        });

        for _ in 0..2 {
            personal.observe_context_ids_target(&[], 5);
            personal.observe_context_ids_target(&[], 6);
        }
        assert_eq!(personal.weakest_index, None);

        personal.observe_context_ids_target(&[], 7);
        let cached = personal.weakest_index;

        personal.observe_context_ids_target(&[], 8);

        assert_eq!(personal.len(), 2);
        assert!(cached.is_some());
        assert_eq!(personal.weakest_index, cached);
    }

    #[test]
    fn full_personal_store_invalidates_weakest_cache_after_replacement() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 2,
            min_count: 1,
        });

        personal.observe_context_ids_target(&[], 5);
        personal.observe_context_ids_target(&[], 6);
        assert_eq!(personal.weakest_entry_index(), Some(0));

        personal.observe_context_ids_target(&[], 7);

        assert_eq!(personal.len(), 2);
        assert_eq!(personal.weakest_index, None);
        assert!(!personal.entries.iter().any(|entry| entry.target_id == 5));
        assert!(personal.entries.iter().any(|entry| entry.target_id == 7));
    }

    #[test]
    fn unigram_cache_tracks_evicted_empty_context_entries() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 4,
            min_count: 1,
        });
        for target_id in 5..9 {
            personal.observe_context_ids_target(&[], target_id);
        }
        personal.observe_context_ids_target(&[], 5);
        personal.observe_context_ids_target(&[], 9);

        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(AutosuggestContext::new(), 8, &mut suggestions);

        assert_eq!(
            suggestions.first().map(|suggestion| suggestion.token_id),
            Some(5)
        );
        assert!(!suggestions
            .iter()
            .any(|suggestion| suggestion.token_id == 6));
        assert!(suggestions
            .iter()
            .any(|suggestion| suggestion.token_id == 9));
    }

    #[test]
    fn decay_removes_stale_singletons() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(3));

        personal.observe_context_target(context, 6);
        personal.decay_counts();

        assert!(personal.is_empty());
        assert_eq!(personal.unigram_cache.len, 0);
    }

    #[test]
    fn personal_entries_remain_key_sorted_after_insert_and_eviction() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 4,
            min_count: 1,
        });
        let mut first = AutosuggestContext::new();
        first.push_token_id(Some(4));
        let mut second = AutosuggestContext::new();
        second.push_token_id(Some(3));

        personal.observe_context_target(first, 7);
        personal.observe_context_target(second, 6);
        personal.observe_context_target(second, 5);
        personal.observe_context_target(first, 6);
        personal.observe_context_target(first, 5);

        assert_eq!(personal.len(), 4);
        assert!(personal.entries.windows(2).all(
            |pair| (pair[0].context, pair[0].target_id) < (pair[1].context, pair[1].target_id)
        ));

        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(second, 4, &mut suggestions);
        assert!(suggestions.iter().all(|suggestion| suggestion.count > 0));
    }

    #[test]
    fn compact_snapshot_round_trips_personal_entries() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(3));
        context.push_token_id(Some(4));
        context.push_token_id(Some(5));

        personal.observe_context_target(context, 6);
        personal.observe_context_target(context, 6);
        personal.observe_context_target(context, 7);

        let bytes = personal.to_compact_bytes();
        assert_eq!(bytes.len(), personal.compact_snapshot_len());
        assert_eq!(
            bytes.len(),
            PERSONAL_HEADER_LEN + personal.len() * PERSONAL_ENTRY_LEN
        );

        let loaded = PersonalAutosuggest::from_compact_bytes(personal.config(), &bytes).unwrap();
        assert_eq!(loaded.config(), personal.config());
        assert_eq!(loaded.model_fingerprint(), 0);
        assert_eq!(loaded.entries, personal.entries);
        assert_eq!(loaded.unigram_cache.items[0].token_id, 6);

        let mut suggestions = Vec::new();
        loaded.suggest_token_ids_into(context, 3, &mut suggestions);
        assert_eq!(
            suggestions.first().map(|suggestion| suggestion.token_id),
            Some(6)
        );
    }

    #[test]
    fn compact_snapshot_reads_legacy_two_token_entries() {
        let config = PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 1,
        };
        let mut bytes = Vec::new();
        bytes.extend_from_slice(PERSONAL_MAGIC);
        write_snapshot_u32(&mut bytes, PERSONAL_VERSION_V1);
        write_snapshot_u32(&mut bytes, 7);
        write_snapshot_u32(&mut bytes, 1);
        write_snapshot_u32(&mut bytes, 0);
        write_snapshot_u32(&mut bytes, 2);
        write_snapshot_u32(&mut bytes, 3);
        write_snapshot_u32(&mut bytes, 4);
        write_snapshot_u32(&mut bytes, 6);
        write_snapshot_u32(&mut bytes, 2);
        write_snapshot_u32(&mut bytes, 7);
        assert_eq!(
            bytes.len(),
            PERSONAL_V1_V2_HEADER_LEN + PERSONAL_V1_ENTRY_LEN
        );

        let loaded = PersonalAutosuggest::from_compact_bytes(config, &bytes).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.entries[0].context.ids(), &[3, 4]);
        assert_eq!(loaded.entries[0].target_id, 6);

        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(3));
        context.push_token_id(Some(4));
        let mut suggestions = Vec::new();
        loaded.suggest_token_ids_into(context, 3, &mut suggestions);
        assert_eq!(
            suggestions
                .iter()
                .map(|suggestion| (suggestion.token_id, suggestion.context_len))
                .collect::<Vec<_>>(),
            vec![(6, 2)]
        );
    }

    #[test]
    fn compact_snapshot_layout_can_be_rejected_against_current_vocab() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 1,
        });
        personal.observe_context_ids_target(&[], 6);
        let mut bytes = personal.to_compact_bytes();
        let invalid_token_id = lm.vocab_size() as u32;
        let target_offset = PERSONAL_HEADER_LEN + 4 + MAX_PERSONAL_CONTEXT_TOKENS * 4;
        bytes[target_offset..target_offset + 4].copy_from_slice(&invalid_token_id.to_le_bytes());

        let loaded = PersonalAutosuggest::from_compact_bytes(personal.config(), &bytes).unwrap();
        assert_eq!(
            loaded.validate_token_ids(lm.vocab_size()).unwrap_err(),
            AutosuggestArtifactError::InvalidTokenId(invalid_token_id)
        );
    }

    #[test]
    fn compact_snapshot_rejects_foreign_context_token_against_current_vocab() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 1,
        });
        personal.observe_context_ids_target(&[lm.vocab_size() as u32], 6);

        assert_eq!(
            personal.validate_token_ids(lm.vocab_size()).unwrap_err(),
            AutosuggestArtifactError::InvalidTokenId(lm.vocab_size() as u32)
        );
    }

    #[test]
    fn compact_snapshot_writer_reuses_caller_buffer() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(3));

        personal.observe_context_target(context, 6);
        personal.observe_context_target(context, 7);

        let mut bytes = Vec::with_capacity(personal.compact_snapshot_len() + 16);
        personal.write_compact_bytes_into(&mut bytes);
        let expected = personal.to_compact_bytes();
        let ptr = bytes.as_ptr();

        bytes.extend_from_slice(b"stale tail");
        personal.write_compact_bytes_into(&mut bytes);

        assert_eq!(bytes, expected);
        assert_eq!(bytes.as_ptr(), ptr);
    }

    #[test]
    fn session_writes_personal_snapshot_into_reused_buffer() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );
        session.commit_token("আমি").unwrap();
        session.commit_token("খাই").unwrap();

        let mut bytes = Vec::with_capacity(session.personal_snapshot_len());
        session.write_personal_snapshot_into(&mut bytes);
        let ptr = bytes.as_ptr();

        session.write_personal_snapshot_into(&mut bytes);
        let loaded =
            PersonalAutosuggest::from_compact_bytes(session.personal().config(), &bytes).unwrap();

        assert_eq!(bytes.as_ptr(), ptr);
        assert_eq!(loaded.model_fingerprint(), lm.vocab_fingerprint());
        assert_eq!(loaded.entries, session.personal().entries);
    }

    #[test]
    fn compact_snapshot_respects_smaller_runtime_cap() {
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 16,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(3));
        for target_id in 5..12 {
            personal.observe_context_target(context, target_id);
        }
        personal.observe_context_target(context, 8);

        let loaded = PersonalAutosuggest::from_compact_bytes(
            PersonalAutosuggestConfig {
                max_entries: 4,
                min_count: 1,
            },
            &personal.to_compact_bytes(),
        )
        .unwrap();

        assert_eq!(loaded.len(), 4);
        assert!(loaded.entries.iter().any(|entry| entry.target_id == 8));
        assert!(loaded.entries.windows(2).all(
            |pair| (pair[0].context, pair[0].target_id) < (pair[1].context, pair[1].target_id)
        ));
    }

    #[test]
    fn compact_snapshot_rejects_invalid_inputs() {
        assert_eq!(
            PersonalAutosuggest::from_compact_bytes(PersonalAutosuggestConfig::default(), b"bad")
                .unwrap_err(),
            PersonalAutosuggestError::UnexpectedEof
        );

        let mut bytes = PersonalAutosuggest::default().to_compact_bytes();
        bytes[0] = b'X';
        assert_eq!(
            PersonalAutosuggest::from_compact_bytes(PersonalAutosuggestConfig::default(), &bytes)
                .unwrap_err(),
            PersonalAutosuggestError::InvalidMagic
        );
    }

    #[test]
    fn default_personal_store_has_explicit_small_memory_bound() {
        let personal = PersonalAutosuggest::default();
        assert_eq!(personal.len(), 0);
        assert_eq!(personal.estimated_heap_bytes(), 0);
        assert_eq!(
            personal.heap_limit_bytes(),
            PersonalAutosuggestConfig::default().heap_limit_bytes()
        );
        assert_eq!(
            personal.heap_limit_bytes(),
            DEFAULT_PERSONAL_AUTOSUGGEST_ENTRIES * mem::size_of::<PersonalEntry>()
                + DEFAULT_PERSONAL_TEXT_AUTOSUGGEST_ENTRIES * mem::size_of::<PersonalTextEntry>()
                + PERSONAL_TEXT_TOTAL_MAX_BYTES
        );
        assert_eq!(personal.compact_snapshot_len(), PERSONAL_HEADER_LEN);
        assert_eq!(
            personal.compact_snapshot_limit_bytes(),
            PersonalAutosuggestConfig::default().compact_snapshot_limit_bytes()
        );
    }

    #[test]
    fn first_personal_learning_reserves_small_block_not_full_cap() {
        let mut personal = PersonalAutosuggest::default();

        personal.observe_context_ids_target(&[], 5);

        assert_eq!(personal.len(), 1);
        assert!(personal.entries.capacity() <= PERSONAL_INITIAL_ENTRY_CAPACITY);
        assert!(
            personal.estimated_heap_bytes()
                < DEFAULT_PERSONAL_AUTOSUGGEST_ENTRIES * mem::size_of::<PersonalEntry>()
        );
    }

    #[test]
    fn observe_committed_token_learns_before_advancing_context() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 32,
            min_count: 2,
        });

        for _ in 0..2 {
            let mut context = AutosuggestContext::new();
            assert!(personal
                .observe_committed_token(&lm, &mut context, "আমি")
                .unwrap());
            assert!(personal
                .observe_committed_token(&lm, &mut context, "খাই")
                .unwrap());
        }

        let mut context = AutosuggestContext::new();
        context.push_token_id(lm.token_id("আমি").unwrap());
        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(context, 3, &mut suggestions);
        assert_eq!(
            suggestions.first().map(|suggestion| suggestion.token_id),
            lm.token_id("খাই").unwrap()
        );
    }

    #[test]
    fn observe_committed_token_learns_unknown_text_words_without_static_context() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 32,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(lm.token_id("আমি").unwrap());

        assert!(personal
            .observe_committed_token(&lm, &mut context, "অচেনা")
            .unwrap());
        assert_eq!(personal.len(), 0);
        assert_eq!(personal.text_len(), 2);
        assert_eq!(context.token_count(), 2);
        assert_eq!(context.matched_token_count(), 0);

        let mut query = AutosuggestContext::new();
        query.push_token_id(lm.token_id("আমি").unwrap());
        let mut suggestions = Vec::new();
        personal.suggest_text_into(query, 3, &mut suggestions);
        assert_eq!(
            suggestions
                .first()
                .and_then(|suggestion| personal.text_suggestion_text(*suggestion)),
            Some("অচেনা")
        );
    }

    #[test]
    fn session_exposes_personal_text_suggestions_without_token_ids() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        assert!(session.commit_token("আমি").unwrap());
        assert!(session.commit_token("অচেনা").unwrap());
        session.clear_context();
        assert!(session.commit_token("আমি").unwrap());

        let metadata = session.suggest_personal_text();
        assert_eq!(metadata.context_token_count, 1);
        assert_eq!(metadata.matched_context_token_count, 1);
        assert_eq!(
            session
                .personal_text_suggestions()
                .first()
                .and_then(|suggestion| session.personal_text_suggestion_text(*suggestion)),
            Some("অচেনা")
        );
    }

    #[test]
    fn session_uses_personal_oov_word_as_context_for_known_target() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        assert!(session.commit_token("অচেনা").unwrap());
        assert!(session.commit_token("খাই").unwrap());
        session.clear_context();
        assert!(session.commit_token("অচেনা").unwrap());
        session.suggest_ids().unwrap();

        assert_eq!(
            session
                .candidate_ids()
                .first()
                .map(|candidate| (candidate.token_id, candidate.source)),
            Some((
                lm.token_id("খাই").unwrap().unwrap(),
                AutosuggestSource::Personal
            ))
        );
    }

    #[test]
    fn session_uses_personal_oov_word_as_context_for_text_target() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        assert!(session.commit_token("অচেনা").unwrap());
        assert!(session.commit_token("ঝুম্পালিকা").unwrap());
        session.clear_context();
        assert!(session.commit_token("অচেনা").unwrap());

        session.suggest_personal_text();
        assert_eq!(
            session
                .personal_text_suggestions()
                .first()
                .and_then(|suggestion| session.personal_text_suggestion_text(*suggestion)),
            Some("ঝুম্পালিকা")
        );
    }

    #[test]
    fn personal_oov_context_round_trips_through_snapshot() {
        let lm = fixture();
        let mut trainer = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        assert!(trainer.commit_token("অচেনা").unwrap());
        assert!(trainer.commit_token("খাই").unwrap());
        let mut snapshot = Vec::new();
        trainer.write_personal_snapshot_into(&mut snapshot);

        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );
        session.import_personal_snapshot(&snapshot).unwrap();
        assert!(session.commit_token("অচেনা").unwrap());
        session.suggest_ids().unwrap();

        assert_eq!(
            session
                .candidate_ids()
                .first()
                .map(|candidate| (candidate.token_id, candidate.source)),
            Some((
                lm.token_id("খাই").unwrap().unwrap(),
                AutosuggestSource::Personal
            ))
        );
    }

    #[test]
    fn scorer_input_does_not_receive_private_oov_context_ids() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 1,
            },
            AutosuggestOptions { max_candidates: 4 },
        );

        assert!(session.commit_token("অচেনা").unwrap());
        let mut scorer_context_ids = [99; 4];
        let metadata = session.rerank_input_into(&mut scorer_context_ids).unwrap();

        assert_eq!(metadata.scorer_context_token_count, 0);
        assert_eq!(metadata.matched_context_token_count, 0);
        assert_eq!(scorer_context_ids, [PAD_ID; 4]);
    }

    #[test]
    fn compact_snapshot_round_trips_personal_text_entries() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 32,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(lm.token_id("আমি").unwrap());

        assert!(personal
            .observe_committed_token(&lm, &mut context, "অচেনা")
            .unwrap());

        let mut bytes = Vec::new();
        personal
            .write_compact_bytes_with_model_fingerprint_into(&mut bytes, lm.vocab_fingerprint());
        let loaded = PersonalAutosuggest::from_compact_bytes_for_model(
            personal.config(),
            &bytes,
            lm.vocab_size(),
            lm.vocab_fingerprint(),
        )
        .unwrap();

        let mut query = AutosuggestContext::new();
        query.push_token_id(lm.token_id("আমি").unwrap());
        let mut suggestions = Vec::new();
        loaded.suggest_text_into(query, 3, &mut suggestions);
        assert_eq!(
            suggestions
                .first()
                .and_then(|suggestion| loaded.text_suggestion_text(*suggestion)),
            Some("অচেনা")
        );
    }

    #[test]
    fn observe_committed_token_rejects_non_bangla_personal_text() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 32,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();

        assert!(!personal
            .observe_committed_token(&lm, &mut context, "hello")
            .unwrap());
        assert_eq!(personal.text_len(), 0);
        assert_eq!(context.token_count(), 1);
        assert_eq!(context.matched_token_count(), 0);
    }

    #[test]
    fn observe_committed_token_learns_then_honors_sentence_boundary() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 32,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(lm.token_id("আমি").unwrap());

        assert!(personal
            .observe_committed_token(&lm, &mut context, "আজ।")
            .unwrap());
        assert_eq!(context.token_count(), 2);
        assert_eq!(context.matched_token_count(), 0);

        let mut query = AutosuggestContext::new();
        query.push_token_id(lm.token_id("আমি").unwrap());
        let mut suggestions = Vec::new();
        personal.suggest_token_ids_into(query, 3, &mut suggestions);
        assert_eq!(
            suggestions.first().map(|suggestion| suggestion.token_id),
            lm.token_id("আজ").unwrap()
        );
    }

    #[test]
    fn commit_strength_weights_are_ordered_and_committed_is_one() {
        // `Committed` must stay 1 so the default observe path is unchanged; the
        // stronger classes must be strictly heavier and above the default
        // suggestion threshold in a single event.
        assert_eq!(CommitStrength::Committed.weight(), 1);
        assert!(
            CommitStrength::CorrectionRejected.weight() >= DEFAULT_PERSONAL_AUTOSUGGEST_MIN_COUNT
        );
        assert!(
            CommitStrength::ManuallyAdded.weight() >= CommitStrength::CorrectionRejected.weight()
        );
    }

    #[test]
    fn observe_committed_with_strength_sets_post_decay_membership_weight() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 64,
            min_count: 2,
        });
        let mut context = AutosuggestContext::new();

        // An out-of-vocabulary name — stored on the personal *text* path.
        let word = "রকিব";
        assert_eq!(personal.committed_text_weight(word), 0);
        assert!(!personal.is_text_established(word, 2));

        personal
            .observe_committed_token_with_strength(
                &lm,
                &mut context,
                word,
                CommitStrength::ManuallyAdded,
            )
            .expect("commit should succeed");
        assert_eq!(
            personal.committed_text_weight(word),
            CommitStrength::ManuallyAdded.weight()
        );
        assert!(personal.is_text_established(word, 2));

        // A following ordinary commit adds exactly 1 on top — proof the transient
        // commit weight was restored to the default after the strong commit.
        let mut next_context = AutosuggestContext::new();
        personal
            .observe_committed_token_with_strength(
                &lm,
                &mut next_context,
                word,
                CommitStrength::Committed,
            )
            .expect("commit should succeed");
        assert_eq!(
            personal.committed_text_weight(word),
            CommitStrength::ManuallyAdded.weight() + 1
        );

        // Decay halves the evidence in place, so membership is post-decay.
        personal.decay_counts();
        assert_eq!(
            personal.committed_text_weight(word),
            (CommitStrength::ManuallyAdded.weight() + 1) / 2
        );
    }

    #[test]
    fn session_membership_gates_on_established_weight() {
        let lm = fixture();
        let mut session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
                max_entries: 64,
                min_count: 2,
            },
            AutosuggestOptions { max_candidates: 5 },
        );

        let name = "নাসির";
        assert_eq!(session.established_weight(name), 0);
        assert!(!session.is_word_established(name, 2));

        // One ordinary commit is below the protection threshold...
        session.commit_token(name).expect("commit should succeed");
        assert_eq!(session.established_weight(name), 1);
        assert!(!session.is_word_established(name, 2));

        // ...but a rejected correction establishes it in a single event.
        let kept = "সৌম";
        session
            .commit_token_with_strength(kept, CommitStrength::CorrectionRejected)
            .expect("commit should succeed");
        assert_eq!(
            session.established_weight(kept),
            CommitStrength::CorrectionRejected.weight()
        );
        assert!(session.is_word_established(kept, 2));
    }
}
