use serde::Serialize;
#[cfg(not(target_arch = "wasm32"))]
use std::error::Error;
#[cfg(not(target_arch = "wasm32"))]
use std::fmt;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use super::artifact::{
    parse_layout, read_i32, read_u32, token_slice, AutosuggestArtifactError, Layout,
    BIGRAM_ROW_LEN, CANDIDATE_RECORD_LEN, COUNT_CANDIDATE_RECORD_LEN, FOURGRAM_ROW_LEN,
    ID_TOKEN_RECORD_LEN, TOKEN_INDEX_RECORD_LEN, TRIGRAM_ROW_LEN, VERSION_V2,
};

pub const DEFAULT_AUTOSUGGEST_CANDIDATES: usize = 5;
pub const AUTOSUGGEST_PAD_ID: u32 = 0;
pub const AUTOSUGGEST_BOS_ID: u32 = 1;
pub const AUTOSUGGEST_UNK_ID: u32 = 2;
pub const AUTOSUGGEST_PAD_I32: i32 = 0;
pub const AUTOSUGGEST_BOS_I32: i32 = 1;
pub const AUTOSUGGEST_UNK_I32: i32 = 2;
pub const MAX_AUTOSUGGEST_CONTEXT_TOKENS: usize = 3;
pub const MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS: usize = 16;
const V1_CONTEXT_TOKENS: usize = 2;
pub const AUTOSUGGEST_ARTIFACT_KIND: &str = "obadh-autosuggest-ngram";
pub(crate) const PAD_ID: u32 = AUTOSUGGEST_PAD_ID;
pub(crate) const BOS_ID: u32 = AUTOSUGGEST_BOS_ID;
pub(crate) const UNK_ID: u32 = AUTOSUGGEST_UNK_ID;

#[derive(Debug, Clone, Copy)]
pub struct AutosuggestOptions {
    pub max_candidates: usize,
}

impl Default for AutosuggestOptions {
    fn default() -> Self {
        Self {
            max_candidates: DEFAULT_AUTOSUGGEST_CANDIDATES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AutosuggestSource {
    Personal,
    Fourgram,
    Trigram,
    Bigram,
    Unigram,
}

impl AutosuggestSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Personal => "personal",
            Self::Fourgram => "fourgram",
            Self::Trigram => "trigram",
            Self::Bigram => "bigram",
            Self::Unigram => "unigram",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AutosuggestCandidate<'a> {
    pub text: &'a str,
    pub token_id: u32,
    pub source: AutosuggestSource,
    pub count: u32,
    pub score: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AutosuggestCandidateId {
    pub token_id: u32,
    pub source: AutosuggestSource,
    pub count: u32,
    pub score: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AutosuggestRerankInputMetadata {
    pub context_token_count: usize,
    pub matched_context_token_count: usize,
    pub scorer_context_token_count: usize,
    pub candidate_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AutosuggestContextPriorOptions {
    pub max_prior_candidates: usize,
}

impl Default for AutosuggestContextPriorOptions {
    fn default() -> Self {
        Self {
            max_prior_candidates: 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AutosuggestContextPriorMetadata {
    pub context_token_count: usize,
    pub matched_context_token_count: usize,
    pub prior_candidate_count: usize,
    pub matched_candidate_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AutosuggestCandidatePrior {
    pub candidate_index: usize,
    pub token_id: u32,
    pub prior_rank: usize,
    pub source: AutosuggestSource,
    pub count: u32,
    pub score: i32,
}

pub fn scorer_candidate_ids_for_candidates_into(
    candidates: &[AutosuggestCandidateId],
    output: &mut [u32],
) -> usize {
    output.fill(PAD_ID);
    let len = candidates.len().min(output.len());
    for (slot, candidate) in output.iter_mut().zip(candidates.iter()).take(len) {
        *slot = candidate.token_id;
    }
    len
}

pub fn scorer_candidate_i32s_for_candidates_into(
    candidates: &[AutosuggestCandidateId],
    output: &mut [i32],
) -> Result<usize, AutosuggestArtifactError> {
    output.fill(AUTOSUGGEST_PAD_I32);
    let len = candidates.len().min(output.len());
    for (slot, candidate) in output.iter_mut().zip(candidates.iter()).take(len) {
        *slot = scorer_i32_from_token_id(candidate.token_id)?;
    }
    Ok(len)
}

/// Incremental next-word context for keyboard integrations.
///
/// The n-gram artifact consumes only the newest two or three known Bengali token
/// IDs, but a platform scorer can use a wider fixed context. Keeping this state
/// outside the LM avoids rescanning committed text on every keystroke. Unknown
/// or special token IDs clear the recent context. Sentence-start state is
/// tracked separately so the LM can use its learned `<bos>` row without
/// treating an unknown in-sentence token as a boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AutosuggestContext {
    token_count: usize,
    ids: [u32; MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS],
    id_len: usize,
    at_sentence_start: bool,
}

impl AutosuggestContext {
    pub fn new() -> Self {
        Self {
            token_count: 0,
            ids: [0; MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS],
            id_len: 0,
            at_sentence_start: true,
        }
    }

    pub fn clear(&mut self) {
        self.token_count = 0;
        self.id_len = 0;
        self.at_sentence_start = true;
    }

    pub fn clear_recent(&mut self) {
        self.id_len = 0;
        self.at_sentence_start = false;
    }

    pub fn push_boundary(&mut self) {
        self.id_len = 0;
        self.at_sentence_start = true;
    }

    pub fn token_count(self) -> usize {
        self.token_count
    }

    pub fn matched_token_count(self) -> usize {
        self.id_len
    }

    pub fn recent_token_ids(&self) -> &[u32] {
        &self.ids[..self.id_len]
    }

    pub fn is_sentence_start(self) -> bool {
        self.at_sentence_start && self.id_len == 0
    }

    pub fn push_token_id(&mut self, token_id: Option<u32>) {
        self.token_count += 1;
        self.at_sentence_start = false;
        match token_id {
            Some(id) if id > UNK_ID => {
                push_recent_context_id(&mut self.ids, &mut self.id_len, id);
            }
            _ => self.id_len = 0,
        }
    }

    pub fn push_unknown(&mut self) {
        self.push_token_id(None);
    }
}

impl Default for AutosuggestContext {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AutosuggestResult<'a> {
    pub context_token_count: usize,
    pub matched_context_token_count: usize,
    pub candidates: Vec<AutosuggestCandidate<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AutosuggestMetadata {
    pub context_token_count: usize,
    pub matched_context_token_count: usize,
}

/// Cheap static identity and sizing metadata for an autosuggest LM.
///
/// This is safe for startup and hot-path-adjacent diagnostics: it is derived
/// from the already-validated artifact header and computed vocabulary identity.
/// The whole-artifact fingerprint is intentionally exposed separately because
/// it scans the full model bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AutosuggestModelInfo {
    pub artifact_kind: &'static str,
    pub version: u32,
    pub artifact_bytes: usize,
    pub vocab_size: usize,
    pub vocab_fingerprint: u32,
    pub unigram_count: usize,
    pub bigram_rows: usize,
    pub trigram_rows: usize,
    pub fourgram_rows: usize,
    pub candidate_record_len: usize,
}

#[derive(Debug, Clone)]
pub struct AutosuggestLm<D: AsRef<[u8]> = Vec<u8>> {
    bytes: D,
    layout: Layout,
    score_mode: ScoreMode,
    vocab_fingerprint: u32,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub enum AutosuggestLoadError {
    Io(std::io::Error),
    Artifact(AutosuggestArtifactError),
}

#[cfg(not(target_arch = "wasm32"))]
impl fmt::Display for AutosuggestLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "failed to load autosuggest artifact: {error}"),
            Self::Artifact(error) => error.fmt(f),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Error for AutosuggestLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Artifact(error) => Some(error),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<std::io::Error> for AutosuggestLoadError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<AutosuggestArtifactError> for AutosuggestLoadError {
    fn from(error: AutosuggestArtifactError) -> Self {
        Self::Artifact(error)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AutosuggestLm<memmap2::Mmap> {
    /// Memory-map an autosuggest artifact for native integrations.
    ///
    /// This avoids copying the shipped model into Rust heap memory. The
    /// returned LM owns the read-only OS mapping, so token text borrowed from
    /// candidates remains valid for the LM lifetime.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, AutosuggestLoadError> {
        let file = std::fs::File::open(path)?;
        Self::from_file(&file)
    }

    /// Memory-map an already-open autosuggest artifact file.
    pub fn from_file(file: &std::fs::File) -> Result<Self, AutosuggestLoadError> {
        // The mapping is read-only, and the returned Mmap owns the OS mapping.
        let mmap = unsafe { memmap2::MmapOptions::new().map(file)? };
        Ok(Self::from_bytes(mmap)?)
    }
}

impl<D: AsRef<[u8]>> AutosuggestLm<D> {
    pub fn from_bytes(bytes: D) -> Result<Self, AutosuggestArtifactError> {
        let layout = parse_layout(bytes.as_ref())?;
        let score_mode = detect_score_mode(bytes.as_ref(), layout)?;
        let computed_fingerprint = compute_vocab_fingerprint(bytes.as_ref(), layout);
        if layout.header.vocab_fingerprint != 0
            && layout.header.vocab_fingerprint != computed_fingerprint
        {
            return Err(AutosuggestArtifactError::ModelFingerprintMismatch {
                expected: layout.header.vocab_fingerprint,
                actual: computed_fingerprint,
            });
        }
        let vocab_fingerprint = layout.header.vocab_fingerprint.max(computed_fingerprint);
        Ok(Self {
            bytes,
            layout,
            score_mode,
            vocab_fingerprint,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.layout.header.vocab_size as usize
    }

    pub fn unigram_count(&self) -> usize {
        self.layout.header.unigram_count as usize
    }

    pub fn bigram_row_count(&self) -> usize {
        self.layout.header.bigram_row_count as usize
    }

    pub fn trigram_row_count(&self) -> usize {
        self.layout.header.trigram_row_count as usize
    }

    pub fn fourgram_row_count(&self) -> usize {
        self.layout.header.fourgram_row_count as usize
    }

    pub fn candidate_record_len(&self) -> usize {
        self.layout.header.candidate_record_len as usize
    }

    fn model_context_token_limit(&self) -> usize {
        if self.layout.header.version >= VERSION_V2 && self.fourgram_row_count() > 0 {
            MAX_AUTOSUGGEST_CONTEXT_TOKENS
        } else {
            V1_CONTEXT_TOKENS
        }
    }

    pub fn artifact_bytes(&self) -> usize {
        self.layout.sections.end
    }

    pub fn vocab_fingerprint(&self) -> u32 {
        self.vocab_fingerprint
    }

    pub fn model_info(&self) -> AutosuggestModelInfo {
        AutosuggestModelInfo {
            artifact_kind: AUTOSUGGEST_ARTIFACT_KIND,
            version: self.layout.header.version,
            artifact_bytes: self.artifact_bytes(),
            vocab_size: self.vocab_size(),
            vocab_fingerprint: self.vocab_fingerprint(),
            unigram_count: self.unigram_count(),
            bigram_rows: self.bigram_row_count(),
            trigram_rows: self.trigram_row_count(),
            fourgram_rows: self.fourgram_row_count(),
            candidate_record_len: self.candidate_record_len(),
        }
    }

    /// Compute a stable whole-artifact fingerprint.
    ///
    /// This scans the entire artifact and is intended for load-time integrity
    /// checks, manifest validation, and tooling. Keep it off the typing hot
    /// path; native keyboard integrations that only need the checked header and
    /// vocab identity can skip this call.
    pub fn artifact_fingerprint(&self) -> u64 {
        compute_artifact_fingerprint(self.bytes.as_ref())
    }

    pub fn token_id(&self, token: &str) -> Result<Option<u32>, AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let target = token.as_bytes();
        let mut low = 0_usize;
        let mut high = self.layout.header.token_index_count as usize;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.layout.sections.token_index + mid * TOKEN_INDEX_RECORD_LEN;
            let token_offset = read_u32(bytes, offset)?;
            let token_len = read_u32(bytes, offset + 4)?;
            let id = read_u32(bytes, offset + 8)?;
            let token = self.token_bytes(token_offset, token_len)?;
            match token.cmp(target) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => return Ok(Some(id)),
                std::cmp::Ordering::Greater => high = mid,
            }
        }

        Ok(None)
    }

    pub fn token_text(&self, token_id: u32) -> Result<&str, AutosuggestArtifactError> {
        if token_id >= self.layout.header.vocab_size {
            return Err(AutosuggestArtifactError::InvalidTokenId(token_id));
        }
        let offset = self.layout.sections.id_tokens + token_id as usize * ID_TOKEN_RECORD_LEN;
        let token_offset = read_u32(self.bytes.as_ref(), offset)?;
        let token_len = read_u32(self.bytes.as_ref(), offset + 4)?;
        token_slice(self.bytes.as_ref(), self.layout, token_offset, token_len)
    }

    /// Returns true only for vocabulary IDs that represent commit-worthy words.
    ///
    /// IDs `0..=2` are reserved artifact control tokens (`<pad>`, `<bos>`,
    /// `<unk>`). Use `None`/`commit_unknown` for unknown committed text instead
    /// of passing those reserved IDs through the hot path.
    pub fn is_word_token_id(&self, token_id: u32) -> bool {
        token_id > UNK_ID && token_id < self.layout.header.vocab_size
    }

    pub fn validate_word_token_id(&self, token_id: u32) -> Result<(), AutosuggestArtifactError> {
        if !self.is_word_token_id(token_id) {
            return Err(AutosuggestArtifactError::InvalidTokenId(token_id));
        }
        Ok(())
    }

    pub fn suggest_for_text(
        &self,
        context: &str,
        options: AutosuggestOptions,
    ) -> Result<AutosuggestResult<'_>, AutosuggestArtifactError> {
        let mut candidates = Vec::with_capacity(options.max_candidates.max(1));
        let metadata = self.suggest_for_text_into(context, options, &mut candidates)?;
        Ok(AutosuggestResult {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            candidates,
        })
    }

    pub fn suggest_for_text_into<'a>(
        &'a self,
        context: &str,
        options: AutosuggestOptions,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        let mut autosuggest_context = AutosuggestContext::new();
        self.push_context_text(&mut autosuggest_context, context)?;

        self.suggest_for_context_into(autosuggest_context, options, output)
    }

    pub fn suggest_for_tokens(
        &self,
        context_tokens: &[&str],
        options: AutosuggestOptions,
    ) -> Result<AutosuggestResult<'_>, AutosuggestArtifactError> {
        let mut candidates = Vec::with_capacity(options.max_candidates.max(1));
        let metadata = self.suggest_for_tokens_into(context_tokens, options, &mut candidates)?;
        Ok(AutosuggestResult {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            candidates,
        })
    }

    pub fn suggest_for_tokens_into<'a>(
        &'a self,
        context_tokens: &[&str],
        options: AutosuggestOptions,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        let mut context = AutosuggestContext::new();

        for token in context_tokens {
            self.push_context_token(&mut context, token)?;
        }
        self.suggest_for_context_into(context, options, output)
    }

    pub fn push_context_token(
        &self,
        context: &mut AutosuggestContext,
        raw_token: &str,
    ) -> Result<(), AutosuggestArtifactError> {
        let token = analyze_context_token(raw_token);
        if let Some(token) = token.text {
            context.push_token_id(self.token_id(token)?);
        }
        if token.boundary_after {
            context.push_boundary();
        }
        Ok(())
    }

    pub fn push_context_text(
        &self,
        context: &mut AutosuggestContext,
        text: &str,
    ) -> Result<(), AutosuggestArtifactError> {
        let mut token_start = None;

        for (index, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if let Some(start) = token_start.take() {
                    self.push_context_token(context, &text[start..index])?;
                }
                if is_editor_boundary(ch) {
                    context.push_boundary();
                }
            } else if token_start.is_none() {
                token_start = Some(index);
            }
        }

        if let Some(start) = token_start {
            self.push_context_token(context, &text[start..])?;
        }

        Ok(())
    }

    pub fn suggest_for_context(
        &self,
        context: AutosuggestContext,
        options: AutosuggestOptions,
    ) -> Result<AutosuggestResult<'_>, AutosuggestArtifactError> {
        let mut candidates = Vec::with_capacity(options.max_candidates.max(1));
        let metadata = self.suggest_for_context_into(context, options, &mut candidates)?;
        Ok(AutosuggestResult {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            candidates,
        })
    }

    pub fn suggest_for_context_into<'a>(
        &'a self,
        context: AutosuggestContext,
        options: AutosuggestOptions,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        output.clear();
        let mut model_context_ids = [0; MAX_AUTOSUGGEST_CONTEXT_TOKENS];
        let model_context_len = copy_model_context_ids(
            context,
            self.model_context_token_limit(),
            &mut model_context_ids,
        );
        let model_context = &model_context_ids[..model_context_len];
        for token_id in model_context {
            if *token_id >= self.layout.header.vocab_size {
                return Err(AutosuggestArtifactError::InvalidTokenId(*token_id));
            }
        }
        let matched_context_token_count = if context.is_sentence_start() {
            0
        } else {
            model_context_len
        };
        self.suggest_for_token_ids_into(
            context.token_count(),
            matched_context_token_count,
            model_context,
            options,
            output,
        )
    }

    pub fn suggest_ids_for_context_into(
        &self,
        context: AutosuggestContext,
        options: AutosuggestOptions,
        output: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        output.clear();
        let mut model_context_ids = [0; MAX_AUTOSUGGEST_CONTEXT_TOKENS];
        let model_context_len = copy_model_context_ids(
            context,
            self.model_context_token_limit(),
            &mut model_context_ids,
        );
        let model_context = &model_context_ids[..model_context_len];
        for token_id in model_context {
            if *token_id >= self.layout.header.vocab_size {
                return Err(AutosuggestArtifactError::InvalidTokenId(*token_id));
            }
        }
        let matched_context_token_count = if context.is_sentence_start() {
            0
        } else {
            model_context_len
        };
        self.suggest_ids_for_token_ids_into(
            context.token_count(),
            matched_context_token_count,
            model_context,
            options,
            output,
        )
    }

    /// Fill a scorer context buffer with left-padded token IDs.
    ///
    /// The newest known context IDs are right-aligned. At a true sentence start
    /// the buffer contains only the right-aligned `<bos>` ID. Unknown in-sentence
    /// tokens clear the context rather than becoming `<unk>`, matching the
    /// n-gram runtime and the training pipeline.
    pub fn scorer_context_ids_for_context_into(
        &self,
        context: AutosuggestContext,
        output: &mut [u32],
    ) -> Result<usize, AutosuggestArtifactError> {
        output.fill(PAD_ID);
        if output.is_empty() {
            return Ok(0);
        }

        if context.is_sentence_start() {
            let last = output.len() - 1;
            output[last] = BOS_ID;
            return Ok(1);
        }

        let ids = context.recent_token_ids();
        let len = ids.len().min(output.len());
        let ids = &ids[ids.len().saturating_sub(len)..];
        for token_id in ids {
            if *token_id >= self.layout.header.vocab_size {
                output.fill(PAD_ID);
                return Err(AutosuggestArtifactError::InvalidTokenId(*token_id));
            }
        }
        let start = output.len() - len;
        output[start..].copy_from_slice(ids);
        Ok(len)
    }

    /// Fill a Core ML-compatible int32 scorer context buffer.
    ///
    /// Core ML exports use fixed int32 tensors. This mirrors
    /// `scorer_context_ids_for_context_into` without forcing platform clients to
    /// allocate a temporary u32 buffer and convert it on each suggestion.
    pub fn scorer_context_i32s_for_context_into(
        &self,
        context: AutosuggestContext,
        output: &mut [i32],
    ) -> Result<usize, AutosuggestArtifactError> {
        output.fill(AUTOSUGGEST_PAD_I32);
        if output.is_empty() {
            return Ok(0);
        }

        if context.is_sentence_start() {
            let last = output.len() - 1;
            output[last] = AUTOSUGGEST_BOS_I32;
            return Ok(1);
        }

        let ids = context.recent_token_ids();
        let len = ids.len().min(output.len());
        let ids = &ids[ids.len().saturating_sub(len)..];
        for token_id in ids {
            if *token_id >= self.layout.header.vocab_size {
                output.fill(AUTOSUGGEST_PAD_I32);
                return Err(AutosuggestArtifactError::InvalidTokenId(*token_id));
            }
        }
        let start = output.len() - len;
        for (slot, token_id) in output[start..].iter_mut().zip(ids.iter()) {
            *slot = scorer_i32_from_token_id(*token_id)?;
        }
        Ok(len)
    }

    /// Build fixed-shape inputs for a bounded candidate scorer.
    ///
    /// The returned candidate IDs are generated by the compact n-gram artifact.
    /// Platform runtimes can pass `scorer_context_ids` and the candidates'
    /// `token_id`s to Core ML/ONNX, then materialize text only for the few
    /// visible results.
    pub fn rerank_input_for_context_into(
        &self,
        context: AutosuggestContext,
        options: AutosuggestOptions,
        scorer_context_ids: &mut [u32],
        candidates: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestArtifactError> {
        candidates.clear();
        let scorer_context_token_count =
            self.scorer_context_ids_for_context_into(context, scorer_context_ids)?;
        let metadata = self.suggest_ids_for_context_into(context, options, candidates)?;

        Ok(AutosuggestRerankInputMetadata {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            scorer_context_token_count,
            candidate_count: candidates.len(),
        })
    }

    /// Find next-word context priors for an already-bounded candidate list.
    ///
    /// This is the intended bridge between current-word autocorrection and the
    /// next-word LM: generate a small context prior pool, then annotate only the
    /// candidate texts that are already present in another decoder's candidate
    /// set. The method intentionally uses a linear scan over both small lists to
    /// avoid a per-keystroke hash table allocation.
    pub fn context_priors_for_candidate_texts_into(
        &self,
        context: AutosuggestContext,
        candidate_texts: &[&str],
        options: AutosuggestContextPriorOptions,
        prior_scratch: &mut Vec<AutosuggestCandidateId>,
        output: &mut Vec<AutosuggestCandidatePrior>,
    ) -> Result<AutosuggestContextPriorMetadata, AutosuggestArtifactError> {
        output.clear();
        prior_scratch.clear();

        let pool_size = options.max_prior_candidates.max(1);
        let metadata = self.suggest_ids_for_context_into(
            context,
            AutosuggestOptions {
                max_candidates: pool_size,
            },
            prior_scratch,
        )?;

        for (candidate_index, text) in candidate_texts.iter().enumerate() {
            let Some(token_id) = self.token_id(text)? else {
                continue;
            };
            if let Some((prior_index, prior)) = prior_scratch
                .iter()
                .enumerate()
                .find(|(_, prior)| prior.token_id == token_id)
            {
                output.push(AutosuggestCandidatePrior {
                    candidate_index,
                    token_id,
                    prior_rank: prior_index + 1,
                    source: prior.source,
                    count: prior.count,
                    score: prior.score,
                });
            }
        }

        Ok(AutosuggestContextPriorMetadata {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            prior_candidate_count: prior_scratch.len(),
            matched_candidate_count: output.len(),
        })
    }

    pub fn materialize_candidate(
        &self,
        candidate: AutosuggestCandidateId,
    ) -> Result<AutosuggestCandidate<'_>, AutosuggestArtifactError> {
        Ok(AutosuggestCandidate {
            text: self.token_text(candidate.token_id)?,
            token_id: candidate.token_id,
            source: candidate.source,
            count: candidate.count,
            score: candidate.score,
        })
    }

    fn suggest_for_token_ids_into<'a>(
        &'a self,
        context_token_count: usize,
        matched_context_token_count: usize,
        context_ids: &[u32],
        options: AutosuggestOptions,
        candidates: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        let limit = options.max_candidates.max(1);
        if self.score_mode == ScoreMode::BackoffOrder {
            return self.suggest_for_token_ids_backoff_into(
                context_token_count,
                matched_context_token_count,
                context_ids,
                limit,
                candidates,
            );
        }

        if let [.., prefix1, prefix2, prefix3] = context_ids {
            if let Some(row) = self.find_fourgram_row(*prefix1, *prefix2, *prefix3)? {
                self.merge_candidates(
                    row.0,
                    row.1,
                    AutosuggestSource::Fourgram,
                    limit,
                    candidates,
                )?;
            }
        }

        if let [.., prefix1, prefix2] = context_ids {
            if let Some(row) = self.find_trigram_row(*prefix1, *prefix2)? {
                self.merge_candidates(row.0, row.1, AutosuggestSource::Trigram, limit, candidates)?;
            }
        }

        if let Some(prefix) = context_ids.last().copied() {
            if let Some(row) = self.find_bigram_row(prefix)? {
                self.merge_candidates(row.0, row.1, AutosuggestSource::Bigram, limit, candidates)?;
            }
        }

        self.merge_unigrams(limit, candidates)?;

        Ok(AutosuggestMetadata {
            context_token_count,
            matched_context_token_count,
        })
    }

    fn suggest_ids_for_token_ids_into(
        &self,
        context_token_count: usize,
        matched_context_token_count: usize,
        context_ids: &[u32],
        options: AutosuggestOptions,
        candidates: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        let limit = options.max_candidates.max(1);
        if self.score_mode == ScoreMode::BackoffOrder {
            return self.suggest_ids_for_token_ids_backoff_into(
                context_token_count,
                matched_context_token_count,
                context_ids,
                limit,
                candidates,
            );
        }

        if let [.., prefix1, prefix2, prefix3] = context_ids {
            if let Some(row) = self.find_fourgram_row(*prefix1, *prefix2, *prefix3)? {
                self.merge_candidate_ids(
                    row.0,
                    row.1,
                    AutosuggestSource::Fourgram,
                    limit,
                    candidates,
                )?;
            }
        }

        if let [.., prefix1, prefix2] = context_ids {
            if let Some(row) = self.find_trigram_row(*prefix1, *prefix2)? {
                self.merge_candidate_ids(
                    row.0,
                    row.1,
                    AutosuggestSource::Trigram,
                    limit,
                    candidates,
                )?;
            }
        }

        if let Some(prefix) = context_ids.last().copied() {
            if let Some(row) = self.find_bigram_row(prefix)? {
                self.merge_candidate_ids(
                    row.0,
                    row.1,
                    AutosuggestSource::Bigram,
                    limit,
                    candidates,
                )?;
            }
        }

        self.merge_unigram_ids(limit, candidates)?;

        Ok(AutosuggestMetadata {
            context_token_count,
            matched_context_token_count,
        })
    }

    fn suggest_for_token_ids_backoff_into<'a>(
        &'a self,
        context_token_count: usize,
        matched_context_token_count: usize,
        context_ids: &[u32],
        limit: usize,
        candidates: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        if let [.., prefix1, prefix2, prefix3] = context_ids {
            if let Some(row) = self.find_fourgram_row(*prefix1, *prefix2, *prefix3)? {
                self.append_candidates(
                    row.0,
                    row.1,
                    AutosuggestSource::Fourgram,
                    limit,
                    candidates,
                )?;
            }
        }

        if let [.., prefix1, prefix2] = context_ids {
            if candidates.len() < limit {
                if let Some(row) = self.find_trigram_row(*prefix1, *prefix2)? {
                    self.append_candidates(
                        row.0,
                        row.1,
                        AutosuggestSource::Trigram,
                        limit,
                        candidates,
                    )?;
                }
            }
        }

        if candidates.len() < limit {
            if let Some(prefix) = context_ids.last().copied() {
                if let Some(row) = self.find_bigram_row(prefix)? {
                    self.append_candidates(
                        row.0,
                        row.1,
                        AutosuggestSource::Bigram,
                        limit,
                        candidates,
                    )?;
                }
            }
        }

        if candidates.len() < limit {
            self.append_unigrams(limit, candidates)?;
        }

        Ok(AutosuggestMetadata {
            context_token_count,
            matched_context_token_count,
        })
    }

    fn suggest_ids_for_token_ids_backoff_into(
        &self,
        context_token_count: usize,
        matched_context_token_count: usize,
        context_ids: &[u32],
        limit: usize,
        candidates: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        if let [.., prefix1, prefix2, prefix3] = context_ids {
            if let Some(row) = self.find_fourgram_row(*prefix1, *prefix2, *prefix3)? {
                self.append_candidate_ids(
                    row.0,
                    row.1,
                    AutosuggestSource::Fourgram,
                    limit,
                    candidates,
                )?;
            }
        }

        if let [.., prefix1, prefix2] = context_ids {
            if candidates.len() < limit {
                if let Some(row) = self.find_trigram_row(*prefix1, *prefix2)? {
                    self.append_candidate_ids(
                        row.0,
                        row.1,
                        AutosuggestSource::Trigram,
                        limit,
                        candidates,
                    )?;
                }
            }
        }

        if candidates.len() < limit {
            if let Some(prefix) = context_ids.last().copied() {
                if let Some(row) = self.find_bigram_row(prefix)? {
                    self.append_candidate_ids(
                        row.0,
                        row.1,
                        AutosuggestSource::Bigram,
                        limit,
                        candidates,
                    )?;
                }
            }
        }

        if candidates.len() < limit {
            self.append_unigram_ids(limit, candidates)?;
        }

        Ok(AutosuggestMetadata {
            context_token_count,
            matched_context_token_count,
        })
    }

    fn find_bigram_row(&self, prefix: u32) -> Result<Option<(u32, u32)>, AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let mut low = 0_usize;
        let mut high = self.layout.header.bigram_row_count as usize;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.layout.sections.bigram_rows + mid * BIGRAM_ROW_LEN;
            let row_prefix = read_u32(bytes, offset)?;
            match row_prefix.cmp(&prefix) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => {
                    let start = read_u32(bytes, offset + 4)?;
                    let len = read_u32(bytes, offset + 8)?;
                    return Ok(Some((start, len)));
                }
                std::cmp::Ordering::Greater => high = mid,
            }
        }

        Ok(None)
    }

    fn find_fourgram_row(
        &self,
        prefix1: u32,
        prefix2: u32,
        prefix3: u32,
    ) -> Result<Option<(u32, u32)>, AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let mut low = 0_usize;
        let mut high = self.layout.header.fourgram_row_count as usize;
        let target = (prefix1, prefix2, prefix3);

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.layout.sections.fourgram_rows + mid * FOURGRAM_ROW_LEN;
            let row = (
                read_u32(bytes, offset)?,
                read_u32(bytes, offset + 4)?,
                read_u32(bytes, offset + 8)?,
            );
            match row.cmp(&target) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => {
                    let start = read_u32(bytes, offset + 12)?;
                    let len = read_u32(bytes, offset + 16)?;
                    return Ok(Some((start, len)));
                }
                std::cmp::Ordering::Greater => high = mid,
            }
        }

        Ok(None)
    }

    fn find_trigram_row(
        &self,
        prefix1: u32,
        prefix2: u32,
    ) -> Result<Option<(u32, u32)>, AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let mut low = 0_usize;
        let mut high = self.layout.header.trigram_row_count as usize;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.layout.sections.trigram_rows + mid * TRIGRAM_ROW_LEN;
            let row_prefix1 = read_u32(bytes, offset)?;
            let row_prefix2 = read_u32(bytes, offset + 4)?;
            match (row_prefix1, row_prefix2).cmp(&(prefix1, prefix2)) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => {
                    let start = read_u32(bytes, offset + 8)?;
                    let len = read_u32(bytes, offset + 12)?;
                    return Ok(Some((start, len)));
                }
                std::cmp::Ordering::Greater => high = mid,
            }
        }

        Ok(None)
    }

    fn merge_unigrams<'a>(
        &'a self,
        limit: usize,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<(), AutosuggestArtifactError> {
        let len = self.layout.header.unigram_count as usize;
        let record_len = self.candidate_record_len();
        for index in 0..len {
            let offset = self.layout.sections.unigrams + index * record_len;
            if self.merge_candidate_at(offset, AutosuggestSource::Unigram, limit, output)?
                == MergeStatus::Stop
            {
                break;
            }
        }
        Ok(())
    }

    fn merge_unigram_ids(
        &self,
        limit: usize,
        output: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<(), AutosuggestArtifactError> {
        let len = self.layout.header.unigram_count as usize;
        let record_len = self.candidate_record_len();
        for index in 0..len {
            let offset = self.layout.sections.unigrams + index * record_len;
            if self.merge_candidate_id_at(offset, AutosuggestSource::Unigram, limit, output)?
                == MergeStatus::Stop
            {
                break;
            }
        }
        Ok(())
    }

    fn append_unigrams<'a>(
        &'a self,
        limit: usize,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<(), AutosuggestArtifactError> {
        let len = self.layout.header.unigram_count as usize;
        let record_len = self.candidate_record_len();
        for index in 0..len {
            if output.len() >= limit {
                break;
            }
            let offset = self.layout.sections.unigrams + index * record_len;
            self.append_candidate_at(offset, AutosuggestSource::Unigram, output)?;
        }
        Ok(())
    }

    fn append_unigram_ids(
        &self,
        limit: usize,
        output: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<(), AutosuggestArtifactError> {
        let len = self.layout.header.unigram_count as usize;
        let record_len = self.candidate_record_len();
        for index in 0..len {
            if output.len() >= limit {
                break;
            }
            let offset = self.layout.sections.unigrams + index * record_len;
            self.append_candidate_id_at(offset, AutosuggestSource::Unigram, output)?;
        }
        Ok(())
    }

    fn merge_candidates<'a>(
        &'a self,
        start: u32,
        len: u32,
        source: AutosuggestSource,
        limit: usize,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<(), AutosuggestArtifactError> {
        let start = start as usize;
        let len = len as usize;
        let end = start
            .checked_add(len)
            .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
        if end > self.layout.header.candidate_count as usize {
            return Err(AutosuggestArtifactError::InvalidSectionLayout);
        }

        let record_len = self.candidate_record_len();
        for index in start..end {
            let offset = self.layout.sections.candidates + index * record_len;
            if self.merge_candidate_at(offset, source, limit, output)? == MergeStatus::Stop {
                break;
            }
        }

        Ok(())
    }

    fn merge_candidate_ids(
        &self,
        start: u32,
        len: u32,
        source: AutosuggestSource,
        limit: usize,
        output: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<(), AutosuggestArtifactError> {
        let start = start as usize;
        let len = len as usize;
        let end = start
            .checked_add(len)
            .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
        if end > self.layout.header.candidate_count as usize {
            return Err(AutosuggestArtifactError::InvalidSectionLayout);
        }

        let record_len = self.candidate_record_len();
        for index in start..end {
            let offset = self.layout.sections.candidates + index * record_len;
            if self.merge_candidate_id_at(offset, source, limit, output)? == MergeStatus::Stop {
                break;
            }
        }

        Ok(())
    }

    fn append_candidates<'a>(
        &'a self,
        start: u32,
        len: u32,
        source: AutosuggestSource,
        limit: usize,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<(), AutosuggestArtifactError> {
        let start = start as usize;
        let len = len as usize;
        let end = start
            .checked_add(len)
            .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
        if end > self.layout.header.candidate_count as usize {
            return Err(AutosuggestArtifactError::InvalidSectionLayout);
        }

        let record_len = self.candidate_record_len();
        for index in start..end {
            if output.len() >= limit {
                break;
            }
            let offset = self.layout.sections.candidates + index * record_len;
            self.append_candidate_at(offset, source, output)?;
        }

        Ok(())
    }

    fn append_candidate_ids(
        &self,
        start: u32,
        len: u32,
        source: AutosuggestSource,
        limit: usize,
        output: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<(), AutosuggestArtifactError> {
        let start = start as usize;
        let len = len as usize;
        let end = start
            .checked_add(len)
            .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
        if end > self.layout.header.candidate_count as usize {
            return Err(AutosuggestArtifactError::InvalidSectionLayout);
        }

        let record_len = self.candidate_record_len();
        for index in start..end {
            if output.len() >= limit {
                break;
            }
            let offset = self.layout.sections.candidates + index * record_len;
            self.append_candidate_id_at(offset, source, output)?;
        }

        Ok(())
    }

    fn append_candidate_at<'a>(
        &'a self,
        offset: usize,
        source: AutosuggestSource,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<(), AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let record = read_candidate_record(bytes, self.layout, offset)?;
        let token_id = record.token_id;
        if token_id <= UNK_ID
            || output
                .iter()
                .any(|candidate| candidate.token_id == token_id)
        {
            return Ok(());
        }
        let text = self.token_text(token_id)?;
        output.push(AutosuggestCandidate {
            text,
            token_id,
            source,
            count: record.count,
            score: record.score,
        });
        Ok(())
    }

    fn append_candidate_id_at(
        &self,
        offset: usize,
        source: AutosuggestSource,
        output: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<(), AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let record = read_candidate_record(bytes, self.layout, offset)?;
        let token_id = record.token_id;
        if token_id <= UNK_ID
            || output
                .iter()
                .any(|candidate| candidate.token_id == token_id)
        {
            return Ok(());
        }
        output.push(AutosuggestCandidateId {
            token_id,
            source,
            count: record.count,
            score: record.score,
        });
        Ok(())
    }

    fn merge_candidate_at<'a>(
        &'a self,
        offset: usize,
        source: AutosuggestSource,
        limit: usize,
        output: &mut Vec<AutosuggestCandidate<'a>>,
    ) -> Result<MergeStatus, AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let record = read_candidate_record(bytes, self.layout, offset)?;
        let token_id = record.token_id;
        if token_id <= UNK_ID {
            return Ok(MergeStatus::Continue);
        }

        if let Some(position) = output
            .iter()
            .position(|candidate| candidate.token_id == token_id)
        {
            if candidate_precedes(
                record.score,
                source,
                record.count,
                token_id,
                &output[position],
            ) {
                output.remove(position);
            } else if output.len() >= limit
                && output.last().is_some_and(|last| {
                    !candidate_precedes(record.score, source, record.count, token_id, last)
                })
            {
                return Ok(MergeStatus::Stop);
            } else {
                return Ok(MergeStatus::Continue);
            }
        } else if output.len() >= limit
            && output.last().is_some_and(|last| {
                !candidate_precedes(record.score, source, record.count, token_id, last)
            })
        {
            return Ok(MergeStatus::Stop);
        } else if output.len() >= limit {
            output.pop();
        }

        let text = self.token_text(token_id)?;
        let candidate = AutosuggestCandidate {
            text,
            token_id,
            source,
            count: record.count,
            score: record.score,
        };
        let insert_at = output
            .iter()
            .position(|existing| {
                candidate_precedes(record.score, source, record.count, token_id, existing)
            })
            .unwrap_or(output.len());
        output.insert(insert_at, candidate);
        Ok(MergeStatus::Accepted)
    }

    fn merge_candidate_id_at(
        &self,
        offset: usize,
        source: AutosuggestSource,
        limit: usize,
        output: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<MergeStatus, AutosuggestArtifactError> {
        let bytes = self.bytes.as_ref();
        let record = read_candidate_record(bytes, self.layout, offset)?;
        let token_id = record.token_id;
        if token_id <= UNK_ID {
            return Ok(MergeStatus::Continue);
        }

        if let Some(position) = output
            .iter()
            .position(|candidate| candidate.token_id == token_id)
        {
            if candidate_id_precedes(
                record.score,
                source,
                record.count,
                token_id,
                &output[position],
            ) {
                output.remove(position);
            } else if output.len() >= limit
                && output.last().is_some_and(|last| {
                    !candidate_id_precedes(record.score, source, record.count, token_id, last)
                })
            {
                return Ok(MergeStatus::Stop);
            } else {
                return Ok(MergeStatus::Continue);
            }
        } else if output.len() >= limit
            && output.last().is_some_and(|last| {
                !candidate_id_precedes(record.score, source, record.count, token_id, last)
            })
        {
            return Ok(MergeStatus::Stop);
        } else if output.len() >= limit {
            output.pop();
        }

        let candidate = AutosuggestCandidateId {
            token_id,
            source,
            count: record.count,
            score: record.score,
        };
        let insert_at = output
            .iter()
            .position(|existing| {
                candidate_id_precedes(record.score, source, record.count, token_id, existing)
            })
            .unwrap_or(output.len());
        output.insert(insert_at, candidate);
        Ok(MergeStatus::Accepted)
    }

    fn token_bytes(&self, offset: u32, len: u32) -> Result<&[u8], AutosuggestArtifactError> {
        let start = self
            .layout
            .sections
            .token_bytes
            .checked_add(offset as usize)
            .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
        let end = start
            .checked_add(len as usize)
            .ok_or(AutosuggestArtifactError::InvalidSectionLayout)?;
        self.bytes
            .as_ref()
            .get(start..end)
            .ok_or(AutosuggestArtifactError::UnexpectedEof)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScoreMode {
    BackoffOrder,
    InterpolatedScore,
}

#[derive(Debug, Clone, Copy)]
struct CandidateRecord {
    token_id: u32,
    count: u32,
    score: i32,
}

fn read_candidate_record(
    bytes: &[u8],
    layout: Layout,
    offset: usize,
) -> Result<CandidateRecord, AutosuggestArtifactError> {
    let token_id = read_u32(bytes, offset)?;
    let count = read_u32(bytes, offset + 4)?;
    let score = match layout.header.candidate_record_len as usize {
        COUNT_CANDIDATE_RECORD_LEN => count.min(i32::MAX as u32) as i32,
        CANDIDATE_RECORD_LEN => read_i32(bytes, offset + 8)?,
        _ => return Err(AutosuggestArtifactError::InvalidSectionLayout),
    };
    Ok(CandidateRecord {
        token_id,
        count,
        score,
    })
}

fn detect_score_mode(bytes: &[u8], layout: Layout) -> Result<ScoreMode, AutosuggestArtifactError> {
    if layout.header.candidate_record_len as usize == COUNT_CANDIDATE_RECORD_LEN {
        return Ok(ScoreMode::BackoffOrder);
    }

    let sample_len = (layout.header.unigram_count as usize).min(8);
    if sample_len == 0 {
        return Ok(ScoreMode::InterpolatedScore);
    }

    let record_len = layout.header.candidate_record_len as usize;
    for index in 0..sample_len {
        let offset = layout.sections.unigrams + index * record_len;
        let record = read_candidate_record(bytes, layout, offset)?;
        if record.score < 0 || record.score as u32 != record.count {
            return Ok(ScoreMode::InterpolatedScore);
        }
    }

    Ok(ScoreMode::BackoffOrder)
}

fn compute_vocab_fingerprint(bytes: &[u8], layout: Layout) -> u32 {
    const OFFSET: u32 = 0x811c_9dc5;
    const PRIME: u32 = 0x0100_0193;

    let mut hash = OFFSET;
    for value in [
        layout.header.vocab_size,
        layout.header.token_index_count,
        layout.header.token_bytes_len,
    ] {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u32::from(byte)).wrapping_mul(PRIME);
        }
    }

    for byte in &bytes[layout.sections.id_tokens..layout.sections.token_index] {
        hash = (hash ^ u32::from(*byte)).wrapping_mul(PRIME);
    }
    for byte in &bytes[layout.sections.token_bytes..layout.sections.end] {
        hash = (hash ^ u32::from(*byte)).wrapping_mul(PRIME);
    }

    if hash == 0 {
        1
    } else {
        hash
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeStatus {
    Accepted,
    Continue,
    Stop,
}

fn candidate_precedes(
    score: i32,
    source: AutosuggestSource,
    count: u32,
    token_id: u32,
    other: &AutosuggestCandidate<'_>,
) -> bool {
    candidate_order_key(score, source, count, token_id)
        > candidate_order_key(other.score, other.source, other.count, other.token_id)
}

fn candidate_id_precedes(
    score: i32,
    source: AutosuggestSource,
    count: u32,
    token_id: u32,
    other: &AutosuggestCandidateId,
) -> bool {
    candidate_order_key(score, source, count, token_id)
        > candidate_order_key(other.score, other.source, other.count, other.token_id)
}

fn candidate_order_key(
    score: i32,
    source: AutosuggestSource,
    count: u32,
    token_id: u32,
) -> (i32, u8, u32, std::cmp::Reverse<u32>) {
    (
        score,
        source_priority(source),
        count,
        std::cmp::Reverse(token_id),
    )
}

fn source_priority(source: AutosuggestSource) -> u8 {
    match source {
        AutosuggestSource::Personal => 5,
        AutosuggestSource::Fourgram => 4,
        AutosuggestSource::Trigram => 3,
        AutosuggestSource::Bigram => 2,
        AutosuggestSource::Unigram => 1,
    }
}

fn scorer_i32_from_token_id(token_id: u32) -> Result<i32, AutosuggestArtifactError> {
    i32::try_from(token_id).map_err(|_| AutosuggestArtifactError::InvalidTokenId(token_id))
}

fn compute_artifact_fingerprint(bytes: &[u8]) -> u64 {
    // Canonical FNV-1a shared with the autocorrect artifacts, so a fingerprint is
    // comparable across every artifact type. See `crate::fingerprint`.
    crate::fingerprint::artifact_fingerprint(bytes)
}

fn copy_model_context_ids(
    context: AutosuggestContext,
    max_context_tokens: usize,
    output: &mut [u32; MAX_AUTOSUGGEST_CONTEXT_TOKENS],
) -> usize {
    if context.is_sentence_start() {
        output[0] = BOS_ID;
        return 1;
    }
    let ids = context.recent_token_ids();
    let max_context_tokens = max_context_tokens.min(output.len());
    let start = ids.len().saturating_sub(max_context_tokens);
    let ids = &ids[start..];
    output[..ids.len()].copy_from_slice(ids);
    ids.len()
}

fn push_recent_context_id(ids: &mut [u32], len: &mut usize, id: u32) {
    if *len < ids.len() {
        ids[*len] = id;
        *len += 1;
    } else {
        ids.copy_within(1.., 0);
        ids[ids.len() - 1] = id;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextToken<'a> {
    pub(crate) text: Option<&'a str>,
    pub(crate) boundary_after: bool,
}

pub(crate) fn analyze_context_token(raw: &str) -> ContextToken<'_> {
    let token = raw.trim_matches(is_context_punctuation);
    ContextToken {
        text: if token.is_empty() { None } else { Some(token) },
        boundary_after: has_trailing_sentence_boundary(raw),
    }
}

fn has_trailing_sentence_boundary(raw: &str) -> bool {
    let mut found_boundary = false;
    for ch in raw.chars().rev() {
        if is_context_punctuation(ch) {
            found_boundary |= is_sentence_boundary(ch);
            continue;
        }
        return found_boundary;
    }
    found_boundary
}

fn is_context_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation() || matches!(ch, '।' | '॥' | '…' | '“' | '”' | '‘' | '’' | '—' | '–')
}

fn is_sentence_boundary(ch: char) -> bool {
    matches!(ch, '।' | '॥' | '.' | '!' | '?' | '…')
}

fn is_editor_boundary(ch: char) -> bool {
    matches!(ch, '\n' | '\r' | '\u{2028}' | '\u{2029}')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autosuggest::artifact::{
        test_support::{build_count_fixture, build_fixture, Row},
        COUNT_CANDIDATE_RECORD_LEN, VERSION, VERSION_V2, VERSION_V3,
    };

    fn fixture() -> AutosuggestLm<Vec<u8>> {
        let tokens = [
            "<pad>",
            "<bos>",
            "<unk>",
            "আমি",
            "আজ",
            "সকালে",
            "স্কুলে",
            "যাব",
            "খাব",
        ];
        AutosuggestLm::from_bytes(build_fixture(
            &tokens,
            &[(5, 100, -800), (7, 90, -900), (8, 80, -950)],
            &[
                Row {
                    context: vec![4],
                    candidates: vec![(5, 40, -700), (7, 12, -850)],
                },
                Row {
                    context: vec![3, 4],
                    candidates: vec![(7, 9, -500), (5, 7, -600)],
                },
            ],
        ))
        .expect("fixture should parse")
    }

    fn sentence_start_fixture() -> AutosuggestLm<Vec<u8>> {
        let tokens = ["<pad>", "<bos>", "<unk>", "শুরু", "সাধারণ", "অন্য"];
        AutosuggestLm::from_bytes(build_fixture(
            &tokens,
            &[(4, 100, 100), (5, 90, 90)],
            &[Row {
                context: vec![BOS_ID],
                candidates: vec![(3, 30, 30)],
            }],
        ))
        .expect("sentence-start fixture should parse")
    }

    #[test]
    fn token_lookup_uses_sorted_index() {
        let lm = fixture();
        assert_eq!(lm.token_id("আজ").unwrap(), Some(4));
        assert_eq!(lm.token_id("নেই").unwrap(), None);
        assert_eq!(lm.token_text(7).unwrap(), "যাব");
        assert_ne!(lm.vocab_fingerprint(), 0);
        assert_ne!(lm.artifact_fingerprint(), 0);
    }

    #[test]
    fn model_info_reports_validated_header_identity() {
        let lm = fixture();
        let info = lm.model_info();

        assert_eq!(info.artifact_kind, AUTOSUGGEST_ARTIFACT_KIND);
        assert_eq!(info.version, VERSION);
        assert_eq!(info.artifact_bytes, lm.artifact_bytes());
        assert_eq!(info.vocab_size, 9);
        assert_eq!(info.vocab_fingerprint, lm.vocab_fingerprint());
        assert_eq!(info.unigram_count, 3);
        assert_eq!(info.bigram_rows, 1);
        assert_eq!(info.trigram_rows, 1);
        assert_eq!(info.fourgram_rows, 0);
        assert_eq!(info.candidate_record_len, CANDIDATE_RECORD_LEN);
    }

    #[test]
    fn count_only_artifact_derives_score_from_count() {
        let tokens = ["<pad>", "<bos>", "<unk>", "আমি", "আজ", "যাব", "না"];
        let lm = AutosuggestLm::from_bytes(build_count_fixture(
            &tokens,
            &[(5, 100, 0), (6, 50, 0)],
            &[Row {
                context: vec![3, 4],
                candidates: vec![(5, 9, 0), (6, 7, 0)],
            }],
        ))
        .unwrap();

        let result = lm
            .suggest_for_text("আমি আজ", AutosuggestOptions { max_candidates: 3 })
            .unwrap();

        assert_eq!(lm.model_info().version, VERSION_V3);
        assert_eq!(
            lm.model_info().candidate_record_len,
            COUNT_CANDIDATE_RECORD_LEN
        );
        assert_eq!(result.matched_context_token_count, V1_CONTEXT_TOKENS);
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| (candidate.text, candidate.count, candidate.score))
                .collect::<Vec<_>>(),
            vec![("যাব", 9, 9), ("না", 7, 7)]
        );
    }

    #[test]
    fn word_token_validation_rejects_reserved_artifact_ids() {
        let lm = fixture();

        for token_id in 0..=UNK_ID {
            assert!(!lm.is_word_token_id(token_id));
            assert_eq!(
                lm.validate_word_token_id(token_id).unwrap_err(),
                AutosuggestArtifactError::InvalidTokenId(token_id)
            );
        }

        assert!(lm.is_word_token_id(3));
        assert!(lm.validate_word_token_id(3).is_ok());
        assert!(!lm.is_word_token_id(lm.vocab_size() as u32));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn mmap_backed_lm_loads_from_path_without_heap_copying_artifact() {
        let tokens = ["<pad>", "<bos>", "<unk>", "আমি", "আজ"];
        let bytes = build_fixture(
            &tokens,
            &[(3, 20, 20), (4, 10, 10)],
            &[Row {
                context: vec![BOS_ID],
                candidates: vec![(4, 7, 7)],
            }],
        );
        let path = std::env::temp_dir().join(format!(
            "obadh-autosuggest-lm-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, &bytes).unwrap();

        let lm = AutosuggestLm::from_path(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(lm.artifact_bytes(), bytes.len());
        assert_eq!(lm.token_id("আমি").unwrap(), Some(3));
        let result = lm
            .suggest_for_context(
                AutosuggestContext::new(),
                AutosuggestOptions { max_candidates: 2 },
            )
            .unwrap();
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| (candidate.text, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                ("আজ", AutosuggestSource::Bigram),
                ("আমি", AutosuggestSource::Unigram)
            ]
        );
    }

    #[test]
    fn artifact_fingerprint_covers_ngram_body_not_only_vocab() {
        let tokens = ["<pad>", "<bos>", "<unk>", "আমি", "আজ"];
        let bytes = build_fixture(
            &tokens,
            &[(3, 20, 20), (4, 10, 10)],
            &[Row {
                context: vec![BOS_ID],
                candidates: vec![(4, 7, 7)],
            }],
        );
        let mut changed = bytes.clone();
        let layout = parse_layout(&changed).unwrap();
        changed[layout.sections.candidates + 4] ^= 1;

        let original = AutosuggestLm::from_bytes(bytes).unwrap();
        let changed = AutosuggestLm::from_bytes(changed).unwrap();

        assert_eq!(original.vocab_fingerprint(), changed.vocab_fingerprint());
        assert_ne!(
            original.artifact_fingerprint(),
            changed.artifact_fingerprint()
        );
    }

    #[test]
    fn zero_header_fingerprint_uses_computed_vocab_fingerprint() {
        let tokens = ["<pad>", "<bos>", "<unk>", "আমি"];
        let bytes = build_fixture(&tokens, &[(3, 10, 10)], &[]);
        assert_eq!(u32::from_le_bytes(bytes[48..52].try_into().unwrap()), 0);

        let lm = AutosuggestLm::from_bytes(bytes).unwrap();

        assert_ne!(lm.vocab_fingerprint(), 0);
    }

    #[test]
    fn nonzero_header_fingerprint_must_match_computed_vocab_fingerprint() {
        let tokens = ["<pad>", "<bos>", "<unk>", "আমি"];
        let mut bytes = build_fixture(&tokens, &[(3, 10, 10)], &[]);
        let wrong = 0x1234_5678_u32;
        bytes[48..52].copy_from_slice(&wrong.to_le_bytes());

        let error = AutosuggestLm::from_bytes(bytes).unwrap_err();

        assert!(matches!(
            error,
            AutosuggestArtifactError::ModelFingerprintMismatch {
                expected: 0x1234_5678,
                actual
            } if actual != 0x1234_5678
        ));
    }

    #[test]
    fn trigram_candidates_backoff_without_duplicates() {
        let lm = fixture();
        let result = lm
            .suggest_for_text("আমি আজ", AutosuggestOptions { max_candidates: 4 })
            .unwrap();
        let rows = result
            .candidates
            .iter()
            .map(|candidate| (candidate.text, candidate.source))
            .collect::<Vec<_>>();
        assert_eq!(
            rows,
            vec![
                ("যাব", AutosuggestSource::Trigram),
                ("সকালে", AutosuggestSource::Trigram),
                ("খাব", AutosuggestSource::Unigram)
            ]
        );
        assert_eq!(result.matched_context_token_count, 2);
    }

    #[test]
    fn sentence_start_context_uses_bos_bigram_without_reporting_matched_word() {
        let lm = sentence_start_fixture();

        let result = lm
            .suggest_for_context(
                AutosuggestContext::new(),
                AutosuggestOptions { max_candidates: 2 },
            )
            .unwrap();

        assert_eq!(result.context_token_count, 0);
        assert_eq!(result.matched_context_token_count, 0);
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| (candidate.text, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                ("শুরু", AutosuggestSource::Bigram),
                ("সাধারণ", AutosuggestSource::Unigram)
            ]
        );
    }

    #[test]
    fn unknown_in_sentence_context_does_not_use_bos_bigram() {
        let lm = sentence_start_fixture();
        let mut context = AutosuggestContext::new();
        context.push_unknown();

        let result = lm
            .suggest_for_context(context, AutosuggestOptions { max_candidates: 2 })
            .unwrap();

        assert_eq!(result.context_token_count, 1);
        assert_eq!(result.matched_context_token_count, 0);
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| (candidate.text, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                ("সাধারণ", AutosuggestSource::Unigram),
                ("অন্য", AutosuggestSource::Unigram)
            ]
        );
    }

    #[test]
    fn sentence_boundary_restores_bos_bigram_context() {
        let lm = sentence_start_fixture();

        let result = lm
            .suggest_for_text("সাধারণ।", AutosuggestOptions { max_candidates: 2 })
            .unwrap();

        assert_eq!(result.context_token_count, 1);
        assert_eq!(result.matched_context_token_count, 0);
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["শুরু", "সাধারণ"]
        );
    }

    #[test]
    fn candidate_id_api_matches_text_api_and_materializes_lazily() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        lm.push_context_token(&mut context, "আমি").unwrap();
        lm.push_context_token(&mut context, "আজ").unwrap();
        let mut ids = Vec::with_capacity(4);

        let metadata = lm
            .suggest_ids_for_context_into(
                context,
                AutosuggestOptions { max_candidates: 4 },
                &mut ids,
            )
            .unwrap();
        let text = lm
            .suggest_for_context(context, AutosuggestOptions { max_candidates: 4 })
            .unwrap();

        assert_eq!(metadata.context_token_count, text.context_token_count);
        assert_eq!(
            metadata.matched_context_token_count,
            text.matched_context_token_count
        );
        assert_eq!(
            ids.iter()
                .map(|candidate| (
                    candidate.token_id,
                    candidate.source,
                    candidate.count,
                    candidate.score
                ))
                .collect::<Vec<_>>(),
            text.candidates
                .iter()
                .map(|candidate| (
                    candidate.token_id,
                    candidate.source,
                    candidate.count,
                    candidate.score
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ids.into_iter()
                .map(|candidate| lm.materialize_candidate(candidate).unwrap().text)
                .collect::<Vec<_>>(),
            vec!["যাব", "সকালে", "খাব"]
        );
    }

    #[test]
    fn rerank_input_left_pads_wider_context_without_changing_ngram_suffix() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        lm.push_context_token(&mut context, "সকালে").unwrap();
        lm.push_context_token(&mut context, "স্কুলে").unwrap();
        lm.push_context_token(&mut context, "খাব").unwrap();
        lm.push_context_token(&mut context, "আমি").unwrap();
        lm.push_context_token(&mut context, "আজ").unwrap();
        let mut scorer_context_ids = [99; 6];
        let mut candidates = Vec::with_capacity(8);

        let metadata = lm
            .rerank_input_for_context_into(
                context,
                AutosuggestOptions { max_candidates: 4 },
                &mut scorer_context_ids,
                &mut candidates,
            )
            .unwrap();

        assert_eq!(
            scorer_context_ids,
            [
                PAD_ID,
                lm.token_id("সকালে").unwrap().unwrap(),
                lm.token_id("স্কুলে").unwrap().unwrap(),
                lm.token_id("খাব").unwrap().unwrap(),
                lm.token_id("আমি").unwrap().unwrap(),
                lm.token_id("আজ").unwrap().unwrap(),
            ]
        );
        assert_eq!(metadata.context_token_count, 5);
        assert_eq!(metadata.matched_context_token_count, V1_CONTEXT_TOKENS);
        assert_eq!(metadata.scorer_context_token_count, 5);
        assert_eq!(metadata.candidate_count, candidates.len());
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| lm.materialize_candidate(*candidate).unwrap().text)
                .collect::<Vec<_>>(),
            vec!["যাব", "সকালে", "খাব"]
        );
    }

    #[test]
    fn scorer_candidate_ids_are_ordered_and_padded_for_fixed_model_input() {
        let lm = fixture();
        let mut candidates = Vec::new();
        lm.suggest_ids_for_context_into(
            {
                let mut context = AutosuggestContext::new();
                lm.push_context_token(&mut context, "আমি").unwrap();
                lm.push_context_token(&mut context, "আজ").unwrap();
                context
            },
            AutosuggestOptions { max_candidates: 4 },
            &mut candidates,
        )
        .unwrap();
        let mut scorer_candidate_ids = [99; 5];

        let copied =
            scorer_candidate_ids_for_candidates_into(&candidates, &mut scorer_candidate_ids);

        assert_eq!(copied, 3);
        assert_eq!(scorer_candidate_ids, [7, 5, 8, PAD_ID, PAD_ID]);
    }

    #[test]
    fn scorer_candidate_i32s_are_ordered_and_padded_for_coreml_input() {
        let lm = fixture();
        let mut candidates = Vec::new();
        lm.suggest_ids_for_context_into(
            {
                let mut context = AutosuggestContext::new();
                lm.push_context_token(&mut context, "আমি").unwrap();
                lm.push_context_token(&mut context, "আজ").unwrap();
                context
            },
            AutosuggestOptions { max_candidates: 4 },
            &mut candidates,
        )
        .unwrap();
        let mut scorer_candidate_ids = [99; 5];

        let copied =
            scorer_candidate_i32s_for_candidates_into(&candidates, &mut scorer_candidate_ids)
                .unwrap();

        assert_eq!(copied, 3);
        assert_eq!(
            scorer_candidate_ids,
            [7, 5, 8, AUTOSUGGEST_PAD_I32, AUTOSUGGEST_PAD_I32]
        );
    }

    #[test]
    fn scorer_candidate_ids_truncate_to_caller_buffer() {
        let candidates = [
            AutosuggestCandidateId {
                token_id: 10,
                source: AutosuggestSource::Fourgram,
                count: 1,
                score: 1,
            },
            AutosuggestCandidateId {
                token_id: 20,
                source: AutosuggestSource::Trigram,
                count: 1,
                score: 1,
            },
            AutosuggestCandidateId {
                token_id: 30,
                source: AutosuggestSource::Bigram,
                count: 1,
                score: 1,
            },
        ];
        let mut scorer_candidate_ids = [99; 2];

        let copied =
            scorer_candidate_ids_for_candidates_into(&candidates, &mut scorer_candidate_ids);

        assert_eq!(copied, 2);
        assert_eq!(scorer_candidate_ids, [10, 20]);
    }

    #[test]
    fn scorer_candidate_i32s_reject_unrepresentable_token_ids() {
        let candidates = [AutosuggestCandidateId {
            token_id: i32::MAX as u32 + 1,
            source: AutosuggestSource::Unigram,
            count: 1,
            score: 1,
        }];
        let mut scorer_candidate_ids = [99; 1];

        let error =
            scorer_candidate_i32s_for_candidates_into(&candidates, &mut scorer_candidate_ids)
                .unwrap_err();

        assert_eq!(
            error,
            AutosuggestArtifactError::InvalidTokenId(i32::MAX as u32 + 1)
        );
    }

    #[test]
    fn scorer_context_uses_bos_only_at_true_sentence_start() {
        let lm = sentence_start_fixture();
        let mut start_context = [99; 4];
        let mut start_candidates = Vec::new();

        let start_metadata = lm
            .rerank_input_for_context_into(
                AutosuggestContext::new(),
                AutosuggestOptions { max_candidates: 2 },
                &mut start_context,
                &mut start_candidates,
            )
            .unwrap();

        assert_eq!(start_context, [PAD_ID, PAD_ID, PAD_ID, BOS_ID]);
        assert_eq!(start_metadata.context_token_count, 0);
        assert_eq!(start_metadata.matched_context_token_count, 0);
        assert_eq!(start_metadata.scorer_context_token_count, 1);
        assert_eq!(
            start_candidates
                .iter()
                .map(|candidate| lm.materialize_candidate(*candidate).unwrap().text)
                .collect::<Vec<_>>(),
            vec!["শুরু", "সাধারণ"]
        );

        let mut unknown = AutosuggestContext::new();
        unknown.push_unknown();
        let mut unknown_context = [99; 4];
        let mut unknown_candidates = Vec::new();
        let unknown_metadata = lm
            .rerank_input_for_context_into(
                unknown,
                AutosuggestOptions { max_candidates: 2 },
                &mut unknown_context,
                &mut unknown_candidates,
            )
            .unwrap();

        assert_eq!(unknown_context, [PAD_ID; 4]);
        assert_eq!(unknown_metadata.context_token_count, 1);
        assert_eq!(unknown_metadata.matched_context_token_count, 0);
        assert_eq!(unknown_metadata.scorer_context_token_count, 0);
        assert_eq!(
            unknown_candidates
                .iter()
                .map(|candidate| lm.materialize_candidate(*candidate).unwrap().text)
                .collect::<Vec<_>>(),
            vec!["সাধারণ", "অন্য"]
        );
    }

    #[test]
    fn context_priors_annotate_bounded_candidate_texts() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        lm.push_context_token(&mut context, "আমি").unwrap();
        lm.push_context_token(&mut context, "আজ").unwrap();
        let candidate_texts = ["খাব", "যাব", "নেই"];
        let mut prior_scratch = Vec::new();
        let mut priors = Vec::new();

        let metadata = lm
            .context_priors_for_candidate_texts_into(
                context,
                &candidate_texts,
                AutosuggestContextPriorOptions {
                    max_prior_candidates: 3,
                },
                &mut prior_scratch,
                &mut priors,
            )
            .unwrap();

        assert_eq!(metadata.context_token_count, 2);
        assert_eq!(metadata.matched_context_token_count, 2);
        assert_eq!(metadata.prior_candidate_count, 3);
        assert_eq!(metadata.matched_candidate_count, 2);
        assert_eq!(
            priors
                .iter()
                .map(|prior| (
                    prior.candidate_index,
                    prior.token_id,
                    prior.prior_rank,
                    prior.source
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    0,
                    lm.token_id("খাব").unwrap().unwrap(),
                    3,
                    AutosuggestSource::Unigram
                ),
                (
                    1,
                    lm.token_id("যাব").unwrap().unwrap(),
                    1,
                    AutosuggestSource::Trigram
                ),
            ]
        );
    }

    #[test]
    fn scorer_context_i32s_match_u32_context_for_coreml_input() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        lm.push_context_token(&mut context, "আমি").unwrap();
        lm.push_context_token(&mut context, "আজ").unwrap();
        let mut scorer_context_u32 = [99; 5];
        let mut scorer_context_i32 = [99; 5];

        let u32_len = lm
            .scorer_context_ids_for_context_into(context, &mut scorer_context_u32)
            .unwrap();
        let i32_len = lm
            .scorer_context_i32s_for_context_into(context, &mut scorer_context_i32)
            .unwrap();

        assert_eq!(u32_len, i32_len);
        assert_eq!(scorer_context_u32, [0, 0, 0, 3, 4]);
        assert_eq!(scorer_context_i32, [0, 0, 0, 3, 4]);
    }

    #[test]
    fn scorer_context_rejects_invalid_ids_before_model_handoff() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(99));
        lm.push_context_token(&mut context, "খাব").unwrap();
        lm.push_context_token(&mut context, "আমি").unwrap();
        lm.push_context_token(&mut context, "আজ").unwrap();
        let mut scorer_context_ids = [42; 4];
        let mut candidates = Vec::new();

        let error = lm
            .rerank_input_for_context_into(
                context,
                AutosuggestOptions { max_candidates: 4 },
                &mut scorer_context_ids,
                &mut candidates,
            )
            .unwrap_err();

        assert_eq!(error, AutosuggestArtifactError::InvalidTokenId(99));
        assert_eq!(scorer_context_ids, [PAD_ID; 4]);
        assert!(candidates.is_empty());
    }

    #[test]
    fn v1_artifact_uses_latest_two_token_suffix_from_wider_context_ring() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        lm.push_context_token(&mut context, "খাব").unwrap();
        lm.push_context_token(&mut context, "আমি").unwrap();
        lm.push_context_token(&mut context, "আজ").unwrap();

        assert_eq!(
            context.recent_token_ids(),
            &[
                lm.token_id("খাব").unwrap().unwrap(),
                lm.token_id("আমি").unwrap().unwrap(),
                lm.token_id("আজ").unwrap().unwrap(),
            ]
        );

        let result = lm
            .suggest_for_context(context, AutosuggestOptions { max_candidates: 3 })
            .unwrap();

        assert_eq!(result.context_token_count, 3);
        assert_eq!(result.matched_context_token_count, V1_CONTEXT_TOKENS);
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| (candidate.text, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                ("যাব", AutosuggestSource::Trigram),
                ("সকালে", AutosuggestSource::Trigram),
                ("খাব", AutosuggestSource::Unigram),
            ]
        );
    }

    #[test]
    fn v2_artifact_uses_three_token_context_when_available() {
        let tokens = [
            "<pad>",
            "<bos>",
            "<unk>",
            "আমি",
            "আজ",
            "এখানে",
            "যাব",
            "থাকব",
            "খাব",
        ];
        let lm = AutosuggestLm::from_bytes(build_fixture(
            &tokens,
            &[(6, 1000, -700), (7, 900, -800), (8, 800, -900)],
            &[
                Row {
                    context: vec![5],
                    candidates: vec![(8, 10, -300)],
                },
                Row {
                    context: vec![4, 5],
                    candidates: vec![(7, 20, -200)],
                },
                Row {
                    context: vec![3, 4, 5],
                    candidates: vec![(6, 30, -100)],
                },
            ],
        ))
        .unwrap();

        let result = lm
            .suggest_for_text("আমি আজ এখানে", AutosuggestOptions { max_candidates: 4 })
            .unwrap();

        assert_eq!(lm.model_info().version, VERSION_V2);
        assert_eq!(lm.model_info().fourgram_rows, 1);
        assert_eq!(result.context_token_count, 3);
        assert_eq!(result.matched_context_token_count, 3);
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| (candidate.text, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                ("যাব", AutosuggestSource::Fourgram),
                ("থাকব", AutosuggestSource::Trigram),
                ("খাব", AutosuggestSource::Bigram),
            ]
        );
    }

    #[test]
    fn lower_order_candidates_can_outrank_weak_specific_context_by_score() {
        let tokens = ["<pad>", "<bos>", "<unk>", "আমি", "আজ", "সকালে", "যাব", "না"];
        let lm = AutosuggestLm::from_bytes(build_fixture(
            &tokens,
            &[(5, 1000, -400), (6, 900, -450), (7, 800, -500)],
            &[
                Row {
                    context: vec![4],
                    candidates: vec![(6, 20, -300), (7, 10, -700)],
                },
                Row {
                    context: vec![3, 4],
                    candidates: vec![(7, 3, -900)],
                },
            ],
        ))
        .unwrap();

        let result = lm
            .suggest_for_text("আমি আজ", AutosuggestOptions { max_candidates: 3 })
            .unwrap();
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| (candidate.text, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                ("যাব", AutosuggestSource::Bigram),
                ("সকালে", AutosuggestSource::Unigram),
                ("না", AutosuggestSource::Unigram)
            ]
        );
    }

    #[test]
    fn legacy_count_scored_artifacts_keep_backoff_order() {
        let tokens = ["<pad>", "<bos>", "<unk>", "আমি", "আজ", "সকালে", "যাব", "খাব"];
        let lm = AutosuggestLm::from_bytes(build_fixture(
            &tokens,
            &[(5, 100, 100), (6, 90, 90), (7, 80, 80)],
            &[Row {
                context: vec![3, 4],
                candidates: vec![(6, 9, 9), (5, 7, 7)],
            }],
        ))
        .unwrap();

        let result = lm
            .suggest_for_text("আমি আজ", AutosuggestOptions { max_candidates: 3 })
            .unwrap();
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| (candidate.text, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                ("যাব", AutosuggestSource::Trigram),
                ("সকালে", AutosuggestSource::Trigram),
                ("খাব", AutosuggestSource::Unigram)
            ]
        );
    }

    #[test]
    fn unknown_context_falls_back_to_unigram() {
        let lm = fixture();
        let result = lm
            .suggest_for_text("আমি অচেনা", AutosuggestOptions { max_candidates: 2 })
            .unwrap();
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["সকালে", "যাব"]
        );
        assert_eq!(result.matched_context_token_count, 0);
    }

    #[test]
    fn incremental_context_api_matches_text_api() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        lm.push_context_token(&mut context, "আমি").unwrap();
        lm.push_context_token(&mut context, "“আজ,”").unwrap();

        assert_eq!(context.token_count(), 2);
        assert_eq!(context.matched_token_count(), 2);
        assert_eq!(context.recent_token_ids(), &[3, 4]);

        let from_context = lm
            .suggest_for_context(context, AutosuggestOptions { max_candidates: 4 })
            .unwrap();
        let from_text = lm
            .suggest_for_text("আমি আজ", AutosuggestOptions { max_candidates: 4 })
            .unwrap();
        assert_eq!(
            from_context
                .candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            from_text
                .candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn sentence_boundary_resets_recent_context_after_committed_word() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        lm.push_context_token(&mut context, "আমি").unwrap();
        lm.push_context_token(&mut context, "আজ।").unwrap();

        assert_eq!(context.token_count(), 2);
        assert_eq!(context.matched_token_count(), 0);
        assert!(context.recent_token_ids().is_empty());

        let result = lm
            .suggest_for_context(context, AutosuggestOptions { max_candidates: 2 })
            .unwrap();
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["সকালে", "যাব"]
        );
    }

    #[test]
    fn punctuation_only_sentence_boundary_resets_recent_context_without_counting_word() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        lm.push_context_token(&mut context, "আমি").unwrap();
        lm.push_context_token(&mut context, "আজ").unwrap();
        lm.push_context_token(&mut context, "।").unwrap();

        assert_eq!(context.token_count(), 2);
        assert_eq!(context.matched_token_count(), 0);
        assert!(context.recent_token_ids().is_empty());
    }

    #[test]
    fn text_api_resets_context_after_sentence_boundary() {
        let lm = fixture();
        let result = lm
            .suggest_for_text("আমি আজ।", AutosuggestOptions { max_candidates: 2 })
            .unwrap();
        assert_eq!(result.context_token_count, 2);
        assert_eq!(result.matched_context_token_count, 0);
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["সকালে", "যাব"]
        );
    }

    #[test]
    fn text_api_keeps_context_across_regular_spaces() {
        let lm = fixture();
        let result = lm
            .suggest_for_text("আমি   আজ", AutosuggestOptions { max_candidates: 4 })
            .unwrap();
        assert_eq!(result.context_token_count, 2);
        assert_eq!(result.matched_context_token_count, 2);
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["যাব", "সকালে", "খাব"]
        );
    }

    #[test]
    fn text_api_resets_context_after_editor_boundary() {
        let lm = fixture();
        let result = lm
            .suggest_for_text("আমি আজ\n", AutosuggestOptions { max_candidates: 2 })
            .unwrap();
        assert_eq!(result.context_token_count, 2);
        assert_eq!(result.matched_context_token_count, 0);
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["সকালে", "যাব"]
        );
    }

    #[test]
    fn explicit_context_boundary_resets_recent_ids_without_counting_word() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        lm.push_context_token(&mut context, "আমি").unwrap();
        lm.push_context_token(&mut context, "আজ").unwrap();
        context.push_boundary();

        assert_eq!(context.token_count(), 2);
        assert_eq!(context.matched_token_count(), 0);
        assert!(context.recent_token_ids().is_empty());
    }

    #[test]
    fn incremental_context_resets_on_unknown_token_ids() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        context.push_token_id(lm.token_id("আমি").unwrap());
        context.push_token_id(lm.token_id("অচেনা").unwrap());

        assert_eq!(context.token_count(), 2);
        assert_eq!(context.matched_token_count(), 0);

        let result = lm
            .suggest_for_context(context, AutosuggestOptions { max_candidates: 2 })
            .unwrap();
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["সকালে", "যাব"]
        );
    }

    #[test]
    fn incremental_context_rejects_invalid_token_ids() {
        let lm = fixture();
        let mut context = AutosuggestContext::new();
        context.push_token_id(Some(99));
        let mut candidates = Vec::with_capacity(4);

        let error = lm
            .suggest_for_context_into(
                context,
                AutosuggestOptions { max_candidates: 4 },
                &mut candidates,
            )
            .unwrap_err();
        assert_eq!(error, AutosuggestArtifactError::InvalidTokenId(99));
        assert!(candidates.is_empty());
    }

    #[test]
    fn into_api_reuses_caller_candidate_buffer() {
        let lm = fixture();
        let mut candidates = Vec::with_capacity(4);
        let initial_capacity = candidates.capacity();

        let metadata = lm
            .suggest_for_text_into(
                "আমি আজ",
                AutosuggestOptions { max_candidates: 4 },
                &mut candidates,
            )
            .unwrap();
        assert_eq!(metadata.matched_context_token_count, 2);
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates.capacity(), initial_capacity);

        let metadata = lm
            .suggest_for_text_into(
                "আমি অচেনা",
                AutosuggestOptions { max_candidates: 2 },
                &mut candidates,
            )
            .unwrap();
        assert_eq!(metadata.matched_context_token_count, 0);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["সকালে", "যাব"]
        );
        assert_eq!(candidates.capacity(), initial_capacity);
    }
}
