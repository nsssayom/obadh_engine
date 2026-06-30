use super::lm::{
    analyze_context_token, AutosuggestCandidate, AutosuggestContext, AutosuggestLm,
    AutosuggestMetadata, AutosuggestOptions, AutosuggestResult, AutosuggestSource,
    MAX_AUTOSUGGEST_CONTEXT_TOKENS,
};
use crate::autosuggest::AutosuggestArtifactError;
use std::error::Error;
use std::fmt;
use std::mem;

const UNK_ID: u32 = 2;
const PERSONAL_MAGIC: &[u8; 16] = b"OBPERSUGLM_V1\0\0\0";
const PERSONAL_VERSION: u32 = 1;
const PERSONAL_HEADER_LEN: usize = 32;
const PERSONAL_ENTRY_LEN: usize = 24;
const PERSONAL_INITIAL_ENTRY_CAPACITY: usize = 16;
const PERSONAL_UNIGRAM_CACHE_LIMIT: usize = 16;

pub const DEFAULT_PERSONAL_AUTOSUGGEST_ENTRIES: usize = 4096;
pub const DEFAULT_PERSONAL_AUTOSUGGEST_MIN_COUNT: u16 = 2;

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

#[derive(Debug)]
pub struct AutosuggestSession<'lm, D: AsRef<[u8]>> {
    lm: &'lm AutosuggestLm<D>,
    personal: PersonalAutosuggest,
    context: AutosuggestContext,
    options: AutosuggestOptions,
    personal_scratch: Vec<PersonalAutosuggestSuggestion>,
    model_scratch: Vec<AutosuggestCandidate<'lm>>,
    candidates: Vec<AutosuggestCandidate<'lm>>,
}

impl<'lm, D: AsRef<[u8]>> AutosuggestSession<'lm, D> {
    pub fn new(
        lm: &'lm AutosuggestLm<D>,
        personal: PersonalAutosuggest,
        options: AutosuggestOptions,
    ) -> Self {
        let capacity = options.max_candidates.max(1);
        Self {
            lm,
            personal,
            context: AutosuggestContext::new(),
            options,
            personal_scratch: Vec::with_capacity(capacity),
            model_scratch: Vec::with_capacity(capacity),
            candidates: Vec::with_capacity(capacity),
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
        self.candidates.clear();
    }

    pub fn push_boundary(&mut self) {
        self.context.push_boundary();
        self.candidates.clear();
    }

    pub fn personal(&self) -> &PersonalAutosuggest {
        &self.personal
    }

    pub fn personal_mut(&mut self) -> &mut PersonalAutosuggest {
        &mut self.personal
    }

    pub fn replace_personal(&mut self, personal: PersonalAutosuggest) {
        self.personal = personal;
        self.candidates.clear();
    }

    pub fn options(&self) -> AutosuggestOptions {
        self.options
    }

    pub fn set_options(&mut self, options: AutosuggestOptions) {
        self.options = options;
        self.ensure_candidate_capacity();
    }

    pub fn candidates(&self) -> &[AutosuggestCandidate<'lm>] {
        &self.candidates
    }

    pub fn commit_token(&mut self, raw_token: &str) -> Result<bool, AutosuggestArtifactError> {
        let learned =
            self.personal
                .observe_committed_token(self.lm, &mut self.context, raw_token)?;
        self.candidates.clear();
        Ok(learned)
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
        let learned =
            self.personal
                .observe_resolved_token_id(&mut self.context, token_id, boundary_after);
        self.candidates.clear();
        Ok(learned)
    }

    /// Commit a token that is not represented in the autosuggest vocabulary.
    pub fn commit_unknown(&mut self, boundary_after: bool) {
        self.personal
            .observe_resolved_token_id(&mut self.context, None, boundary_after);
        self.candidates.clear();
    }

    pub fn suggest(&mut self) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
        self.ensure_candidate_capacity();
        self.personal.suggest_with_lm_into(
            self.lm,
            self.context,
            self.options,
            &mut self.personal_scratch,
            &mut self.model_scratch,
            &mut self.candidates,
        )
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
                self.model_scratch
                    .capacity()
                    .saturating_mul(mem::size_of::<AutosuggestCandidate<'lm>>()),
            )
            .saturating_add(
                self.candidates
                    .capacity()
                    .saturating_mul(mem::size_of::<AutosuggestCandidate<'lm>>()),
            )
    }

    pub fn personal_snapshot_len(&self) -> usize {
        self.personal.compact_snapshot_len()
    }

    pub fn write_personal_snapshot_into(&self, output: &mut Vec<u8>) {
        self.personal.write_compact_bytes_into(output);
    }

    fn ensure_candidate_capacity(&mut self) {
        let capacity = self.options.max_candidates.max(1);
        reserve_to(&mut self.personal_scratch, capacity);
        reserve_to(&mut self.model_scratch, capacity);
        reserve_to(&mut self.candidates, capacity);
    }

    fn validate_token_id(&self, token_id: u32) -> Result<(), AutosuggestArtifactError> {
        if token_id >= self.lm.vocab_size() as u32 {
            return Err(AutosuggestArtifactError::InvalidTokenId(token_id));
        }
        Ok(())
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

#[derive(Debug, Clone)]
pub struct PersonalAutosuggest {
    config: PersonalAutosuggestConfig,
    entries: Vec<PersonalEntry>,
    unigram_cache: PersonalUnigramCache,
    tick: u32,
}

impl PersonalAutosuggest {
    pub fn new(config: PersonalAutosuggestConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            unigram_cache: PersonalUnigramCache::empty(),
            tick: 0,
        }
    }

    pub fn config(&self) -> PersonalAutosuggestConfig {
        self.config
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn estimated_heap_bytes(&self) -> usize {
        self.entries
            .capacity()
            .saturating_mul(std::mem::size_of::<PersonalEntry>())
    }

    pub fn compact_snapshot_len(&self) -> usize {
        PERSONAL_HEADER_LEN + self.entries.len() * PERSONAL_ENTRY_LEN
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.unigram_cache.clear();
        self.tick = 0;
    }

    pub fn decay_counts(&mut self) {
        for entry in &mut self.entries {
            entry.count /= 2;
        }
        self.entries.retain(|entry| entry.count > 0);
        self.rebuild_unigram_cache();
    }

    pub fn from_compact_bytes(
        config: PersonalAutosuggestConfig,
        bytes: &[u8],
    ) -> Result<Self, PersonalAutosuggestError> {
        if bytes.len() < PERSONAL_HEADER_LEN {
            return Err(PersonalAutosuggestError::UnexpectedEof);
        }
        if &bytes[..PERSONAL_MAGIC.len()] != PERSONAL_MAGIC {
            return Err(PersonalAutosuggestError::InvalidMagic);
        }
        let version = read_snapshot_u32(bytes, 16)?;
        if version != PERSONAL_VERSION {
            return Err(PersonalAutosuggestError::UnsupportedVersion(version));
        }
        let tick = read_snapshot_u32(bytes, 20)?;
        let entry_count = read_snapshot_u32(bytes, 24)? as usize;
        let expected_len = PERSONAL_HEADER_LEN
            .checked_add(
                entry_count
                    .checked_mul(PERSONAL_ENTRY_LEN)
                    .ok_or(PersonalAutosuggestError::InvalidLayout)?,
            )
            .ok_or(PersonalAutosuggestError::InvalidLayout)?;
        if expected_len != bytes.len() {
            return Err(PersonalAutosuggestError::InvalidLayout);
        }

        let mut entries = Vec::with_capacity(entry_count.min(config.max_entries));
        let mut max_seen = tick;
        for index in 0..entry_count {
            let offset = PERSONAL_HEADER_LEN + index * PERSONAL_ENTRY_LEN;
            let context_len = read_snapshot_u32(bytes, offset)?;
            if context_len as usize > MAX_AUTOSUGGEST_CONTEXT_TOKENS {
                return Err(PersonalAutosuggestError::InvalidLayout);
            }
            let mut context = PersonalContext::empty();
            context.len = context_len as u8;
            if context_len > 0 {
                context.ids[0] = read_snapshot_u32(bytes, offset + 4)?;
            }
            if context_len > 1 {
                context.ids[1] = read_snapshot_u32(bytes, offset + 8)?;
            }
            let target_id = read_snapshot_u32(bytes, offset + 12)?;
            let count = read_snapshot_u32(bytes, offset + 16)?;
            let last_seen = read_snapshot_u32(bytes, offset + 20)?;
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

        let mut personal = Self {
            config,
            entries,
            unigram_cache: PersonalUnigramCache::empty(),
            tick: max_seen,
        };
        personal.rebuild_unigram_cache();
        Ok(personal)
    }

    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.compact_snapshot_len());
        self.write_compact_bytes_into(&mut bytes);
        bytes
    }

    pub fn write_compact_bytes_into(&self, bytes: &mut Vec<u8>) {
        bytes.clear();
        reserve_to(bytes, self.compact_snapshot_len());
        bytes.extend_from_slice(PERSONAL_MAGIC);
        write_snapshot_u32(bytes, PERSONAL_VERSION);
        write_snapshot_u32(bytes, self.tick);
        write_snapshot_u32(bytes, self.entries.len() as u32);
        write_snapshot_u32(bytes, 0);
        for entry in &self.entries {
            write_snapshot_u32(bytes, u32::from(entry.context.len));
            write_snapshot_u32(bytes, entry.context.ids[0]);
            write_snapshot_u32(bytes, entry.context.ids[1]);
            write_snapshot_u32(bytes, entry.target_id);
            write_snapshot_u32(bytes, u32::from(entry.count));
            write_snapshot_u32(bytes, entry.last_seen);
        }
    }

    pub fn observe_context_target(&mut self, context: AutosuggestContext, target_id: u32) {
        self.observe_context_ids_target(context.recent_token_ids(), target_id);
    }

    pub fn observe_context_ids_target(&mut self, context_ids: &[u32], target_id: u32) {
        if self.config.max_entries == 0 || target_id <= UNK_ID {
            return;
        }

        self.tick = self.tick.wrapping_add(1);
        self.observe_key(PersonalContext::empty(), target_id);

        let usable = context_ids.len().min(MAX_AUTOSUGGEST_CONTEXT_TOKENS);
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
        let token = analyze_context_token(raw_token);
        let token_id = match token.text {
            Some(text) => lm.token_id(text)?,
            None => None,
        };

        Ok(if token.text.is_some() {
            self.observe_resolved_token_id(context, token_id, token.boundary_after)
        } else {
            if token.boundary_after {
                context.push_boundary();
            }
            false
        })
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
        let learned = match token_id {
            Some(id) if id > UNK_ID => {
                self.observe_context_target(*context, id);
                context.push_token_id(Some(id));
                true
            }
            _ => {
                context.push_unknown();
                false
            }
        };

        if boundary_after {
            context.push_boundary();
        }

        learned
    }

    pub fn suggest_token_ids_into(
        &self,
        context: AutosuggestContext,
        limit: usize,
        output: &mut Vec<PersonalAutosuggestSuggestion>,
    ) {
        output.clear();
        let limit = limit.max(1);
        let context_ids = context.recent_token_ids();
        let usable = context_ids.len().min(MAX_AUTOSUGGEST_CONTEXT_TOKENS);

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
        output.clear();
        let limit = options.max_candidates.max(1);
        self.suggest_token_ids_into(context, limit, personal_scratch);
        let has_context = !context.recent_token_ids().is_empty();

        for suggestion in personal_scratch.iter() {
            if output.len() >= limit {
                break;
            }
            if has_context && suggestion.context_len == 0 {
                continue;
            }
            output.push(AutosuggestCandidate {
                text: lm.token_text(suggestion.token_id)?,
                token_id: suggestion.token_id,
                source: AutosuggestSource::Personal,
                count: u32::from(suggestion.count),
                score: suggestion.score,
            });
        }

        let metadata = lm.suggest_for_context_into(context, options, model_scratch)?;
        for candidate in model_scratch.iter() {
            if output.len() >= limit {
                break;
            }
            if output
                .iter()
                .any(|existing| existing.token_id == candidate.token_id)
            {
                continue;
            }
            output.push(candidate.clone());
        }

        if has_context {
            for suggestion in personal_scratch.iter() {
                if output.len() >= limit {
                    break;
                }
                if suggestion.context_len > 0
                    || output
                        .iter()
                        .any(|existing| existing.token_id == suggestion.token_id)
                {
                    continue;
                }
                output.push(AutosuggestCandidate {
                    text: lm.token_text(suggestion.token_id)?,
                    token_id: suggestion.token_id,
                    source: AutosuggestSource::Personal,
                    count: u32::from(suggestion.count),
                    score: suggestion.score,
                });
            }
        }

        Ok(metadata)
    }

    fn observe_key(&mut self, context: PersonalContext, target_id: u32) {
        let mut changed_unigram = None;
        let mut removed_unigram = None;

        match self.find_entry(context, target_id) {
            Ok(index) => {
                let entry = &mut self.entries[index];
                entry.count = entry.count.saturating_add(1);
                entry.last_seen = self.tick;
                if entry.context.is_empty() {
                    changed_unigram = Some(*entry);
                }
            }
            Err(index) => {
                let entry = PersonalEntry {
                    context,
                    target_id,
                    count: 1,
                    last_seen: self.tick,
                };
                if self.entries.len() < self.config.max_entries {
                    self.ensure_entry_capacity_for_insert();
                    self.entries.insert(index, entry);
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

    fn find_entry(&self, context: PersonalContext, target_id: u32) -> Result<usize, usize> {
        self.entries
            .binary_search_by_key(&(context, target_id), |entry| {
                (entry.context, entry.target_id)
            })
    }

    fn insert_entry_sorted(&mut self, entry: PersonalEntry) {
        let index = self
            .find_entry(entry.context, entry.target_id)
            .expect_err("personal autosuggest replacement must be a new key");
        self.entries.insert(index, entry);
    }

    fn weakest_entry_index(&self) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry_strength_key(entry))
            .map(|(index, _)| index)
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

    fn rebuild_unigram_cache(&mut self) {
        self.unigram_cache.clear();
        let empty = PersonalContext::empty();
        let start = self.entries.partition_point(|entry| entry.context < empty);
        let end = start + self.entries[start..].partition_point(|entry| entry.context == empty);
        for entry in &self.entries[start..end] {
            self.unigram_cache.insert(entry.suggestion());
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PersonalContext {
    ids: [u32; MAX_AUTOSUGGEST_CONTEXT_TOKENS],
    len: u8,
}

impl PersonalContext {
    fn empty() -> Self {
        Self {
            ids: [0; MAX_AUTOSUGGEST_CONTEXT_TOKENS],
            len: 0,
        }
    }

    fn from_suffix(ids: &[u32], len: usize) -> Self {
        let len = len.min(MAX_AUTOSUGGEST_CONTEXT_TOKENS).min(ids.len());
        let start = ids.len() - len;
        let mut context = Self::empty();
        context.len = len as u8;
        context.ids[..len].copy_from_slice(&ids[start..]);
        context
    }

    fn is_empty(self) -> bool {
        self.len == 0
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

fn reserve_to<T>(values: &mut Vec<T>, capacity: usize) {
    if values.capacity() < capacity {
        values.reserve_exact(capacity - values.capacity());
    }
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
        assert_eq!(session.personal().len(), 1);
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
        session.replace_personal(loaded);
        session.suggest().unwrap();

        assert_eq!(
            session.candidates().first().map(|candidate| candidate.text),
            Some("খাই")
        );
        assert_eq!(session.context().matched_token_count(), 1);
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

        personal.observe_context_target(context, 6);
        personal.observe_context_target(context, 6);
        personal.observe_context_target(context, 7);

        let bytes = personal.to_compact_bytes();
        assert_eq!(bytes.len(), personal.compact_snapshot_len());

        let loaded = PersonalAutosuggest::from_compact_bytes(personal.config(), &bytes).unwrap();
        assert_eq!(loaded.config(), personal.config());
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
        assert_eq!(personal.compact_snapshot_len(), PERSONAL_HEADER_LEN);
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
    fn observe_committed_token_does_not_learn_unknown_words() {
        let lm = fixture();
        let mut personal = PersonalAutosuggest::new(PersonalAutosuggestConfig {
            max_entries: 32,
            min_count: 1,
        });
        let mut context = AutosuggestContext::new();
        context.push_token_id(lm.token_id("আমি").unwrap());

        assert!(!personal
            .observe_committed_token(&lm, &mut context, "অচেনা")
            .unwrap());
        assert_eq!(personal.len(), 0);
        assert_eq!(context.token_count(), 2);
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
}
