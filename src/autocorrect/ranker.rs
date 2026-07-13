use std::collections::HashMap;

use super::bangla::bangla_units;
use super::edit::EditCost;
use super::lexicon::{Lexicon, LexiconEntry};

pub const AUTOCORRECT_FEATURE_DIM: usize = 9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrectionRequest {
    pub current: String,
    pub left_context: Vec<String>,
    pub roman_input: Option<String>,
    pub obadh_output: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionSource {
    NoChange,
    LexiconEdit,
    PrefixCompletion,
    PhoneticSkeleton,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrectionCandidate {
    pub text: String,
    pub source: CorrectionSource,
    pub edit_cost: EditCost,
    pub frequency: u32,
    pub score: i32,
    pub features: CandidateFeatures,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateFeatures {
    pub source_id: u8,
    pub edit_cost: u16,
    pub input_unit_len: u16,
    pub candidate_unit_len: u16,
    pub unit_len_delta: i16,
    pub frequency_log2: u8,
    pub input_known: bool,
    pub candidate_known: bool,
    pub obadh_baseline: bool,
}

/// The result of [`AutocorrectEngine::decide`] over the in-memory [`Lexicon`]:
/// the ranked candidates and a conservative replacement suggestion.
///
/// This is the heap path, for offline and corpus use. The mmap runtime a
/// keyboard uses exposes ranked candidates with provenance instead
/// (`FstLexicon::suggest`), leaving auto-insert to the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutocorrectDecision {
    /// The baseline that was analyzed — the Bengali the user's keystrokes
    /// produced. Also the head of [`candidates`](Self::candidates) as the
    /// `NoChange` "keep" candidate.
    pub input: String,
    /// Ranked candidates, best first, starting with the keep candidate.
    pub candidates: Vec<CorrectionCandidate>,
    /// A conservative replacement over the keep candidate: `Some` only for a
    /// [`LexiconEdit`](CorrectionSource::LexiconEdit) (never a prefix or skeleton
    /// guess) that beats the keep candidate by
    /// [`AutocorrectConfig::autocorrect_margin`]. `None` otherwise.
    pub replacement: Option<CorrectionCandidate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutocorrectConfig {
    pub max_edit_cost: u16,
    pub roman_input_max_edit_cost: u16,
    pub max_candidates: usize,
    pub unknown_keep_score: i32,
    pub known_keep_bonus: i32,
    pub edit_cost_penalty: i32,
    pub skeleton_edit_cost_penalty: i32,
    /// How far a correction must outscore the keep candidate before
    /// [`decide`](AutocorrectEngine::decide) will set
    /// [`AutocorrectDecision::replacement`]. Higher is more conservative.
    pub autocorrect_margin: i32,
    /// Whether [`decide`](AutocorrectEngine::decide) may set `replacement` when
    /// the request carries a `roman_input`. Default `false`. See
    /// [`decide`](AutocorrectEngine::decide).
    pub auto_replace_roman_input: bool,
    pub search_known_input: bool,
    pub max_prefix_candidates: usize,
    pub max_skeleton_candidates: usize,
    pub max_skeleton_edit_cost: u16,
}

#[derive(Debug, Clone)]
pub struct AutocorrectEngine {
    lexicon: Lexicon,
    config: AutocorrectConfig,
}

#[derive(Debug, Clone)]
struct RequestAnalysis {
    frequency: u32,
    unit_len: u16,
}

impl CorrectionRequest {
    pub fn new(current: impl Into<String>) -> Self {
        Self {
            current: current.into(),
            left_context: Vec::new(),
            roman_input: None,
            obadh_output: None,
        }
    }

    pub fn with_left_context(
        mut self,
        context: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.left_context = context.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_roman_input(mut self, roman_input: impl Into<String>) -> Self {
        self.roman_input = Some(roman_input.into());
        self
    }

    pub fn with_obadh_output(mut self, obadh_output: impl Into<String>) -> Self {
        self.obadh_output = Some(obadh_output.into());
        self
    }
}

impl CandidateFeatures {
    pub fn as_i16_array(self) -> [i16; AUTOCORRECT_FEATURE_DIM] {
        [
            self.source_id as i16,
            clamped_i16(self.edit_cost),
            clamped_i16(self.input_unit_len),
            clamped_i16(self.candidate_unit_len),
            self.unit_len_delta,
            self.frequency_log2 as i16,
            self.input_known as i16,
            self.candidate_known as i16,
            self.obadh_baseline as i16,
        ]
    }
}

impl Default for AutocorrectConfig {
    fn default() -> Self {
        Self {
            max_edit_cost: 4,
            roman_input_max_edit_cost: 0,
            max_candidates: 8,
            unknown_keep_score: 500,
            known_keep_bonus: 900,
            edit_cost_penalty: 160,
            skeleton_edit_cost_penalty: 90,
            autocorrect_margin: 180,
            auto_replace_roman_input: false,
            search_known_input: false,
            max_prefix_candidates: 8,
            max_skeleton_candidates: 12,
            max_skeleton_edit_cost: 1,
        }
    }
}

impl AutocorrectEngine {
    pub fn new(lexicon: Lexicon) -> Self {
        Self::with_config(lexicon, AutocorrectConfig::default())
    }

    pub fn from_entries(entries: impl IntoIterator<Item = LexiconEntry>) -> Self {
        Self::new(Lexicon::new(entries))
    }

    pub fn with_config(lexicon: Lexicon, config: AutocorrectConfig) -> Self {
        Self { lexicon, config }
    }

    pub fn lexicon(&self) -> &Lexicon {
        &self.lexicon
    }

    pub fn config(&self) -> AutocorrectConfig {
        self.config
    }

    /// Ranked candidates for `input`, best first. Convenience wrapper over
    /// [`decide`](Self::decide) that returns only the candidates.
    pub fn suggest(&self, input: &str) -> Vec<CorrectionCandidate> {
        self.decide(CorrectionRequest::new(input)).candidates
    }

    /// Rank candidates for a request over the in-memory (heap) [`Lexicon`].
    ///
    /// The returned [`AutocorrectDecision`] carries the baseline
    /// ([`input`](AutocorrectDecision::input)), the ranked
    /// [`candidates`](AutocorrectDecision::candidates), and a conservative
    /// [`replacement`](AutocorrectDecision::replacement) (a confident lexicon
    /// edit only, never a prefix or skeleton guess). If the request carries a
    /// `roman_input`, `replacement` is suppressed unless
    /// [`AutocorrectConfig::auto_replace_roman_input`] is set.
    ///
    /// This heap path is a convenience for offline/corpus use. It is **not** the
    /// mmap `FstLexicon` runtime path a keyboard uses, and its `replacement` is
    /// not an auto-insert recommendation: the runtime exposes ranked candidates
    /// with per-candidate provenance (`FstLexicon::suggest` /
    /// `obadh_autocorrect_suggest_detailed`) and leaves the auto-insert decision
    /// to the client, which has the frequency data and product policy to make it.
    ///
    /// ```
    /// use obadh_engine::{AutocorrectEngine, CorrectionRequest, LexiconEntry};
    ///
    /// let engine = AutocorrectEngine::from_entries([LexiconEntry::new("বাংলা", 10_000)]);
    /// let decision = engine.decide(CorrectionRequest::new("বামলা"));
    ///
    /// // `replacement`, when set, is one of the ranked candidates.
    /// if let Some(replacement) = &decision.replacement {
    ///     assert!(decision.candidates.contains(replacement));
    /// }
    /// // `suggest` is the same ranking without the replacement field.
    /// assert_eq!(engine.suggest("বামলা"), decision.candidates);
    /// ```
    pub fn decide(&self, request: CorrectionRequest) -> AutocorrectDecision {
        if request.current.is_empty() {
            return AutocorrectDecision {
                input: request.current,
                candidates: Vec::new(),
                replacement: None,
            };
        }

        let analysis = RequestAnalysis {
            frequency: self.lexicon.frequency(&request.current).unwrap_or(0),
            unit_len: unit_len(&request.current),
        };
        let keep = self.keep_candidate(&request, &analysis);

        if analysis.frequency > 0 && !self.config.search_known_input {
            return AutocorrectDecision {
                input: request.current,
                candidates: vec![keep],
                replacement: None,
            };
        }

        let mut candidates = vec![keep.clone()];
        candidates.extend(self.lexicon_candidates(&request, &analysis));
        candidates.extend(self.prefix_candidates(&request, &analysis));
        candidates.extend(self.skeleton_candidates(&request, &analysis));
        candidates = deduplicated_candidates(candidates);
        candidates.truncate(self.config.max_candidates.max(1));

        let replacement = candidates
            .iter()
            .find(|candidate| candidate.source != CorrectionSource::NoChange)
            .filter(|candidate| self.can_auto_replace(&request, candidate))
            .filter(|candidate| candidate.score >= keep.score + self.config.autocorrect_margin)
            .cloned();

        AutocorrectDecision {
            input: request.current,
            candidates,
            replacement,
        }
    }

    fn keep_candidate(
        &self,
        request: &CorrectionRequest,
        analysis: &RequestAnalysis,
    ) -> CorrectionCandidate {
        let score = if analysis.frequency > 0 {
            self.config.unknown_keep_score
                + self.config.known_keep_bonus
                + frequency_score(analysis.frequency)
        } else {
            self.config.unknown_keep_score
        };
        CorrectionCandidate {
            text: request.current.clone(),
            source: CorrectionSource::NoChange,
            edit_cost: EditCost(0),
            frequency: analysis.frequency,
            score,
            features: candidate_features(
                request,
                &request.current,
                CorrectionSource::NoChange,
                EditCost(0),
                analysis.frequency,
                analysis.unit_len,
                analysis.unit_len,
            ),
        }
    }

    fn lexicon_candidates(
        &self,
        request: &CorrectionRequest,
        analysis: &RequestAnalysis,
    ) -> Vec<CorrectionCandidate> {
        let mut candidates = Vec::new();
        let max_edit_cost = self.lexicon_edit_cost_limit(request);
        if max_edit_cost == 0 {
            return candidates;
        }

        for matched in self
            .lexicon
            .find_within_edit_cost(&request.current, max_edit_cost)
        {
            let entry = matched.entry;
            if entry.word == request.current {
                continue;
            }
            let score = 1000 - (matched.edit_cost.0 as i32 * self.config.edit_cost_penalty)
                + frequency_score(entry.frequency);
            candidates.push(CorrectionCandidate {
                text: entry.word.clone(),
                source: CorrectionSource::LexiconEdit,
                edit_cost: matched.edit_cost,
                frequency: entry.frequency,
                score,
                features: candidate_features(
                    request,
                    &entry.word,
                    CorrectionSource::LexiconEdit,
                    matched.edit_cost,
                    entry.frequency,
                    analysis.unit_len,
                    matched.unit_len,
                )
                .with_input_known(analysis.frequency > 0)
                .with_candidate_known(true),
            });
        }
        candidates
    }

    fn lexicon_edit_cost_limit(&self, request: &CorrectionRequest) -> u16 {
        if request.roman_input.is_some() {
            self.config
                .max_edit_cost
                .min(self.config.roman_input_max_edit_cost)
        } else {
            self.config.max_edit_cost
        }
    }

    fn skeleton_candidates(
        &self,
        request: &CorrectionRequest,
        analysis: &RequestAnalysis,
    ) -> Vec<CorrectionCandidate> {
        let mut candidates = Vec::new();
        for matched in self.lexicon.find_by_fuzzy_phonetic_skeleton(
            &request.current,
            self.config.max_skeleton_edit_cost,
            self.config.max_skeleton_candidates,
        ) {
            let entry = matched.entry;
            if entry.word == request.current {
                continue;
            }
            let score = 840 - (matched.edit_cost.0 as i32 * self.config.skeleton_edit_cost_penalty)
                + frequency_score(entry.frequency);
            candidates.push(CorrectionCandidate {
                text: entry.word.clone(),
                source: CorrectionSource::PhoneticSkeleton,
                edit_cost: matched.edit_cost,
                frequency: entry.frequency,
                score,
                features: candidate_features(
                    request,
                    &entry.word,
                    CorrectionSource::PhoneticSkeleton,
                    matched.edit_cost,
                    entry.frequency,
                    analysis.unit_len,
                    matched.unit_len,
                )
                .with_input_known(analysis.frequency > 0)
                .with_candidate_known(true),
            });
        }
        candidates
    }

    fn prefix_candidates(
        &self,
        request: &CorrectionRequest,
        analysis: &RequestAnalysis,
    ) -> Vec<CorrectionCandidate> {
        if self.config.max_prefix_candidates == 0 {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        for matched in self
            .lexicon
            .find_by_prefix(&request.current, self.config.max_prefix_candidates)
        {
            let entry = matched.entry;
            if entry.word == request.current {
                continue;
            }
            let completion_units = matched.unit_len.saturating_sub(analysis.unit_len);
            let score = 760 + frequency_score(entry.frequency) - i32::from(completion_units) * 12;
            candidates.push(CorrectionCandidate {
                text: entry.word.clone(),
                source: CorrectionSource::PrefixCompletion,
                edit_cost: matched.edit_cost,
                frequency: entry.frequency,
                score,
                features: candidate_features(
                    request,
                    &entry.word,
                    CorrectionSource::PrefixCompletion,
                    matched.edit_cost,
                    entry.frequency,
                    analysis.unit_len,
                    matched.unit_len,
                )
                .with_input_known(analysis.frequency > 0)
                .with_candidate_known(true),
            });
        }
        candidates
    }

    fn can_auto_replace(
        &self,
        request: &CorrectionRequest,
        candidate: &CorrectionCandidate,
    ) -> bool {
        candidate.source == CorrectionSource::LexiconEdit
            && (self.config.auto_replace_roman_input || request.roman_input.is_none())
    }
}

fn candidate_order(left: &CorrectionCandidate, right: &CorrectionCandidate) -> std::cmp::Ordering {
    right
        .score
        .cmp(&left.score)
        .then_with(|| left.edit_cost.cmp(&right.edit_cost))
        .then_with(|| right.frequency.cmp(&left.frequency))
        .then_with(|| left.text.cmp(&right.text))
}

fn deduplicated_candidates(candidates: Vec<CorrectionCandidate>) -> Vec<CorrectionCandidate> {
    let mut unique = Vec::<CorrectionCandidate>::with_capacity(candidates.len());
    let mut by_text = HashMap::<String, usize>::with_capacity(candidates.len());

    for candidate in candidates {
        if let Some(&index) = by_text.get(&candidate.text) {
            if duplicate_candidate_is_better(&candidate, &unique[index]) {
                unique[index] = candidate;
            }
            continue;
        }

        by_text.insert(candidate.text.clone(), unique.len());
        unique.push(candidate);
    }

    unique.sort_by(candidate_order);
    unique
}

fn duplicate_candidate_is_better(
    candidate: &CorrectionCandidate,
    current: &CorrectionCandidate,
) -> bool {
    let candidate_priority = duplicate_source_priority(candidate.source);
    let current_priority = duplicate_source_priority(current.source);
    if candidate_priority != current_priority {
        return candidate_priority > current_priority;
    }

    candidate_order(candidate, current).is_lt()
}

fn duplicate_source_priority(source: CorrectionSource) -> u8 {
    match source {
        CorrectionSource::NoChange => 5,
        CorrectionSource::LexiconEdit => 4,
        CorrectionSource::PrefixCompletion => 3,
        CorrectionSource::PhoneticSkeleton => 2,
    }
}

fn frequency_score(frequency: u32) -> i32 {
    if frequency == 0 {
        0
    } else {
        (frequency.ilog2() as i32) * 6
    }
}

fn candidate_features(
    request: &CorrectionRequest,
    candidate: &str,
    source: CorrectionSource,
    edit_cost: EditCost,
    frequency: u32,
    input_unit_len: u16,
    candidate_unit_len: u16,
) -> CandidateFeatures {
    let unit_len_delta = (candidate_unit_len as i32 - input_unit_len as i32)
        .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    CandidateFeatures {
        source_id: source_id(source),
        edit_cost: edit_cost.0,
        input_unit_len,
        candidate_unit_len,
        unit_len_delta,
        frequency_log2: frequency_log2(frequency),
        input_known: frequency > 0 && source == CorrectionSource::NoChange,
        candidate_known: frequency > 0,
        obadh_baseline: request
            .obadh_output
            .as_deref()
            .is_some_and(|output| output == candidate),
    }
}

impl CandidateFeatures {
    fn with_input_known(mut self, input_known: bool) -> Self {
        self.input_known = input_known;
        self
    }

    fn with_candidate_known(mut self, candidate_known: bool) -> Self {
        self.candidate_known = candidate_known;
        self
    }
}

fn source_id(source: CorrectionSource) -> u8 {
    match source {
        CorrectionSource::NoChange => 0,
        CorrectionSource::LexiconEdit => 1,
        CorrectionSource::PhoneticSkeleton => 2,
        CorrectionSource::PrefixCompletion => 4,
    }
}

fn unit_len(text: &str) -> u16 {
    bangla_units(text).len().min(u16::MAX as usize) as u16
}

fn frequency_log2(frequency: u32) -> u8 {
    if frequency == 0 {
        0
    } else {
        frequency.ilog2().min(u8::MAX as u32) as u8
    }
}

fn clamped_i16(value: u16) -> i16 {
    value.min(i16::MAX as u16) as i16
}

#[cfg(test)]
mod tests {
    use super::{
        AutocorrectConfig, AutocorrectEngine, CorrectionRequest, CorrectionSource,
        AUTOCORRECT_FEATURE_DIM,
    };
    use crate::autocorrect::{Lexicon, LexiconEntry};

    #[test]
    fn exact_known_word_prefers_no_change() {
        let engine = AutocorrectEngine::from_entries([
            LexiconEntry::new("আমি", 100),
            LexiconEntry::new("আম", 500),
        ]);

        let decision = engine.decide(CorrectionRequest::new("আমি"));

        assert_eq!(decision.replacement, None);
        assert_eq!(decision.candidates[0].source, CorrectionSource::NoChange);
        assert_eq!(decision.candidates.len(), 1);
        assert_eq!(decision.candidates[0].features.input_known, true);
        assert_eq!(decision.candidates[0].features.candidate_known, true);
    }

    #[test]
    fn high_confidence_unknown_typo_can_be_corrected() {
        let engine = AutocorrectEngine::from_entries([LexiconEntry::new("আমি", 10_000)]);

        let decision = engine.decide(CorrectionRequest::new("আমী"));

        assert_eq!(
            decision
                .replacement
                .as_ref()
                .map(|candidate| candidate.text.as_str()),
            Some("আমি")
        );
        let replacement = decision.replacement.as_ref().expect("replacement expected");
        assert_eq!(replacement.features.source_id, 1);
        assert_eq!(replacement.features.edit_cost, 1);
        assert_eq!(replacement.features.input_unit_len, 2);
        assert_eq!(replacement.features.candidate_unit_len, 2);
        assert_eq!(
            replacement.features.as_i16_array().len(),
            AUTOCORRECT_FEATURE_DIM
        );
    }

    #[test]
    fn low_margin_candidate_stays_as_suggestion_only() {
        let engine = AutocorrectEngine::with_config(
            Lexicon::new([LexiconEntry::new("আম", 1)]),
            AutocorrectConfig {
                autocorrect_margin: 1_000,
                ..AutocorrectConfig::default()
            },
        );

        let decision = engine.decide(CorrectionRequest::new("আমি"));

        assert!(decision
            .candidates
            .iter()
            .any(|candidate| candidate.text == "আম"));
        assert_eq!(decision.replacement, None);
    }

    #[test]
    fn known_input_search_requires_explicit_opt_in() {
        let entries = [
            LexiconEntry::new("আমি", 10),
            LexiconEntry::new("আম", 10_000),
        ];
        let default_engine = AutocorrectEngine::from_entries(entries.clone());

        let decision = default_engine.decide(CorrectionRequest::new("আমি"));

        assert_eq!(decision.candidates.len(), 1);
        assert_eq!(decision.candidates[0].text, "আমি");

        let opt_in = AutocorrectEngine::with_config(
            Lexicon::new(entries),
            AutocorrectConfig {
                search_known_input: true,
                ..AutocorrectConfig::default()
            },
        );

        let decision = opt_in.decide(CorrectionRequest::new("আমি"));

        assert!(decision
            .candidates
            .iter()
            .any(|candidate| candidate.text == "আম"));
        assert_eq!(decision.replacement, None);
    }

    #[test]
    fn roman_origin_requests_need_explicit_replacement_opt_in() {
        let engine = AutocorrectEngine::from_entries([LexiconEntry::new("আমি", 10_000)]);

        let decision = engine.decide(
            CorrectionRequest::new("আমী")
                .with_roman_input("ami")
                .with_obadh_output("আমী"),
        );

        assert!(decision
            .candidates
            .iter()
            .any(|candidate| candidate.text == "আমি"
                && candidate.source == CorrectionSource::PhoneticSkeleton));
        assert_eq!(decision.replacement, None);

        let opt_in = AutocorrectEngine::with_config(
            Lexicon::new([LexiconEntry::new("আমি", 10_000)]),
            AutocorrectConfig {
                auto_replace_roman_input: true,
                roman_input_max_edit_cost: 4,
                ..AutocorrectConfig::default()
            },
        );

        let decision = opt_in.decide(
            CorrectionRequest::new("আমী")
                .with_roman_input("ami")
                .with_obadh_output("আমী"),
        );

        assert_eq!(
            decision
                .replacement
                .as_ref()
                .map(|candidate| candidate.text.as_str()),
            Some("আমি")
        );
    }

    #[test]
    fn skeleton_candidates_are_suggestions_not_replacements() {
        let engine = AutocorrectEngine::from_entries([LexiconEntry::new("কিরণ", 10_000)]);

        let decision = engine.decide(CorrectionRequest::new("করন"));

        let candidate = decision
            .candidates
            .iter()
            .find(|candidate| candidate.text == "কিরণ")
            .expect("skeleton candidate should be present");
        assert_eq!(candidate.source, CorrectionSource::PhoneticSkeleton);
        assert_eq!(candidate.features.source_id, 2);
        assert_eq!(decision.replacement, None);
    }
}
