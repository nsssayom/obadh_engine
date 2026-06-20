use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use fst::automaton::{Automaton, Str};
use fst::{IntoStreamer, Streamer};
use serde::Serialize;

use super::edit::weighted_edit_distance;

pub const DEFAULT_FST_MAX_DISTANCE: u32 = 1;
pub const DEFAULT_FST_PREFIX_CANDIDATES: usize = 24;
pub const FST_MAX_LEVENSHTEIN_DISTANCE: u32 = 2;

#[derive(Debug)]
pub struct FstLexicon<D: AsRef<[u8]> = Vec<u8>> {
    map: fst::Map<D>,
}

impl<D: AsRef<[u8]>> FstLexicon<D> {
    pub fn from_map(map: fst::Map<D>) -> Self {
        Self { map }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn exact_frequency(&self, word: &str) -> Option<u64> {
        self.map.get(word.as_bytes())
    }

    pub fn suggest(
        &self,
        baseline: &str,
        options: FstSuggestOptions,
    ) -> Result<FstSuggestResult, FstSuggestError> {
        if options.max_distance > FST_MAX_LEVENSHTEIN_DISTANCE {
            return Err(FstSuggestError::MaxDistanceTooLarge {
                requested: options.max_distance,
                max: FST_MAX_LEVENSHTEIN_DISTANCE,
            });
        }

        let exact_frequency = self.exact_frequency(baseline);
        let retrieval_limit = options
            .max_candidates
            .max(options.response_candidates)
            .max(1);
        let mut seeds = BTreeMap::<String, FstCandidate>::new();
        let mut truncated = false;

        if let Some(frequency) = exact_frequency {
            insert_fst_candidate(
                &mut seeds,
                baseline,
                frequency,
                FstCandidateSource::Exact,
                baseline,
                options.max_edit_cost,
            );
        }

        let levenshtein = UnicodeLevenshtein::new(baseline, options.max_distance);
        let mut edit_stream = self.map.search(levenshtein).into_stream();
        while let Some((key, frequency)) = edit_stream.next() {
            let text = std::str::from_utf8(key).map_err(FstSuggestError::InvalidUtf8)?;
            insert_fst_candidate(
                &mut seeds,
                text,
                frequency,
                FstCandidateSource::EditDistance,
                baseline,
                options.max_edit_cost,
            );
            if seeds.len() >= retrieval_limit {
                truncated = true;
                break;
            }
        }

        if options.max_prefix_candidates > 0 && seeds.len() < retrieval_limit {
            let prefix = Str::new(baseline).starts_with();
            let mut prefix_stream = self.map.search(prefix).into_stream();
            let mut prefix_count = 0_usize;
            while let Some((key, frequency)) = prefix_stream.next() {
                let text = std::str::from_utf8(key).map_err(FstSuggestError::InvalidUtf8)?;
                insert_fst_candidate(
                    &mut seeds,
                    text,
                    frequency,
                    FstCandidateSource::PrefixCompletion,
                    baseline,
                    options.max_edit_cost,
                );
                prefix_count += 1;
                if prefix_count >= options.max_prefix_candidates || seeds.len() >= retrieval_limit {
                    truncated = seeds.len() >= retrieval_limit;
                    break;
                }
            }
        }

        let mut ranked = seeds.into_values().collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| left.edit_cost.cmp(&right.edit_cost))
                .then_with(|| right.frequency.cmp(&left.frequency))
                .then_with(|| left.text.cmp(&right.text))
        });
        let candidate_count = ranked.len();
        ranked.truncate(options.response_candidates);

        Ok(FstSuggestResult {
            baseline: baseline.to_string(),
            exact_frequency,
            max_distance: options.max_distance,
            max_edit_cost: options.max_edit_cost,
            candidate_count,
            returned_candidates: ranked.len(),
            truncated,
            candidates: ranked,
        })
    }
}

impl FstLexicon<Vec<u8>> {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, fst::Error> {
        Ok(Self::from_map(fst::Map::new(bytes)?))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FstSuggestOptions {
    pub max_distance: u32,
    pub max_edit_cost: Option<u16>,
    pub max_candidates: usize,
    pub max_prefix_candidates: usize,
    pub response_candidates: usize,
}

impl Default for FstSuggestOptions {
    fn default() -> Self {
        Self {
            max_distance: DEFAULT_FST_MAX_DISTANCE,
            max_edit_cost: None,
            max_candidates: 512,
            max_prefix_candidates: DEFAULT_FST_PREFIX_CANDIDATES,
            response_candidates: DEFAULT_FST_PREFIX_CANDIDATES,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct FstSuggestResult {
    pub baseline: String,
    pub exact_frequency: Option<u64>,
    pub max_distance: u32,
    pub max_edit_cost: Option<u16>,
    pub candidate_count: usize,
    pub returned_candidates: usize,
    pub truncated: bool,
    pub candidates: Vec<FstCandidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FstCandidate {
    pub text: String,
    pub source: FstCandidateSource,
    pub edit_cost: u16,
    pub frequency: u64,
    pub score: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FstCandidateSource {
    Exact,
    EditDistance,
    PrefixCompletion,
}

impl FstCandidateSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "fst_exact",
            Self::EditDistance => "fst_edit_distance",
            Self::PrefixCompletion => "fst_prefix_completion",
        }
    }

    fn penalty(self) -> i64 {
        match self {
            Self::Exact => 0,
            Self::EditDistance => 0,
            Self::PrefixCompletion => 192,
        }
    }
}

#[derive(Debug)]
pub enum FstSuggestError {
    MaxDistanceTooLarge { requested: u32, max: u32 },
    InvalidUtf8(std::str::Utf8Error),
}

impl fmt::Display for FstSuggestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxDistanceTooLarge { requested, max } => write!(
                formatter,
                "max-distance {requested} is too large for keyboard-time FST lookup; use {max} or less"
            ),
            Self::InvalidUtf8(error) => write!(formatter, "invalid UTF-8 key in FST lexicon: {error}"),
        }
    }
}

impl Error for FstSuggestError {}

fn insert_fst_candidate(
    seeds: &mut BTreeMap<String, FstCandidate>,
    text: &str,
    frequency: u64,
    source: FstCandidateSource,
    baseline: &str,
    max_edit_cost: Option<u16>,
) {
    let edit_cost = weighted_edit_distance(baseline, text).0;
    if max_edit_cost.is_some_and(|limit| edit_cost > limit) {
        return;
    }
    let score = fst_candidate_score(source, edit_cost, frequency);
    let candidate = FstCandidate {
        text: text.to_string(),
        source,
        edit_cost,
        frequency,
        score,
    };
    seeds
        .entry(candidate.text.clone())
        .and_modify(|existing| {
            if candidate.score > existing.score {
                *existing = candidate.clone();
            }
        })
        .or_insert(candidate);
}

fn fst_candidate_score(source: FstCandidateSource, edit_cost: u16, frequency: u64) -> i64 {
    ((frequency.saturating_add(1) as f64).ln() * 1024.0).round() as i64
        - (edit_cost as i64 * 1536)
        - source.penalty()
}

#[derive(Debug, Clone)]
struct UnicodeLevenshtein {
    query: Vec<char>,
    distance: usize,
}

#[derive(Debug, Clone)]
struct UnicodeLevenshteinState {
    row: Vec<usize>,
    pending: [u8; 4],
    pending_len: u8,
    expected_len: u8,
    dead: bool,
}

impl UnicodeLevenshtein {
    fn new(query: &str, distance: u32) -> Self {
        Self {
            query: query.chars().collect(),
            distance: distance as usize,
        }
    }

    fn start_row(&self) -> Vec<usize> {
        (0..=self.query.len()).collect()
    }

    fn accept_char(&self, row: &[usize], ch: char) -> Vec<usize> {
        let mut next = Vec::with_capacity(self.query.len() + 1);
        next.push(row[0].saturating_add(1));
        for (index, query_ch) in self.query.iter().enumerate() {
            let substitution_cost = usize::from(*query_ch != ch);
            let substitution = row[index].saturating_add(substitution_cost);
            let deletion = row[index + 1].saturating_add(1);
            let insertion = next[index].saturating_add(1);
            next.push(
                substitution
                    .min(deletion)
                    .min(insertion)
                    .min(self.distance + 1),
            );
        }
        next
    }

    fn accept_decoded_byte(
        &self,
        state: &UnicodeLevenshteinState,
        byte: u8,
    ) -> UnicodeLevenshteinState {
        if state.dead {
            return state.clone();
        }

        if state.pending_len == 0 {
            if byte <= 0x7f {
                return UnicodeLevenshteinState {
                    row: self.accept_char(&state.row, byte as char),
                    pending: [0; 4],
                    pending_len: 0,
                    expected_len: 0,
                    dead: false,
                };
            }

            let Some(expected_len) = utf8_expected_len(byte) else {
                return state.dead();
            };
            let mut next = state.clone();
            next.pending = [0; 4];
            next.pending[0] = byte;
            next.pending_len = 1;
            next.expected_len = expected_len;
            return next;
        }

        if !is_utf8_continuation(byte) {
            return state.dead();
        }

        let mut next = state.clone();
        next.pending[next.pending_len as usize] = byte;
        next.pending_len += 1;
        if next.pending_len < next.expected_len {
            return next;
        }

        let bytes = &next.pending[..next.pending_len as usize];
        let Ok(decoded) = std::str::from_utf8(bytes) else {
            return state.dead();
        };
        let Some(ch) = decoded.chars().next() else {
            return state.dead();
        };
        UnicodeLevenshteinState {
            row: self.accept_char(&state.row, ch),
            pending: [0; 4],
            pending_len: 0,
            expected_len: 0,
            dead: false,
        }
    }
}

impl UnicodeLevenshteinState {
    fn dead(&self) -> Self {
        Self {
            row: self.row.clone(),
            pending: [0; 4],
            pending_len: 0,
            expected_len: 0,
            dead: true,
        }
    }

    fn is_between_codepoints(&self) -> bool {
        self.pending_len == 0
    }
}

impl Automaton for UnicodeLevenshtein {
    type State = UnicodeLevenshteinState;

    fn start(&self) -> Self::State {
        UnicodeLevenshteinState {
            row: self.start_row(),
            pending: [0; 4],
            pending_len: 0,
            expected_len: 0,
            dead: false,
        }
    }

    fn is_match(&self, state: &Self::State) -> bool {
        !state.dead
            && state.is_between_codepoints()
            && state
                .row
                .last()
                .is_some_and(|distance| *distance <= self.distance)
    }

    fn can_match(&self, state: &Self::State) -> bool {
        !state.dead
            && state
                .row
                .iter()
                .copied()
                .min()
                .is_some_and(|distance| distance <= self.distance)
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        self.accept_decoded_byte(state, byte)
    }
}

fn utf8_expected_len(byte: u8) -> Option<u8> {
    match byte {
        0xc2..=0xdf => Some(2),
        0xe0..=0xef => Some(3),
        0xf0..=0xf4 => Some(4),
        _ => None,
    }
}

fn is_utf8_continuation(byte: u8) -> bool {
    matches!(byte, 0x80..=0xbf)
}

#[cfg(test)]
mod tests {
    use super::{FstLexicon, FstSuggestOptions};

    #[test]
    fn fst_suggest_handles_bangla_unicode_edit_distance() {
        let lexicon = test_lexicon([("কমন", 20), ("কেমন", 100), ("যেমন", 30), ("বিজ্ঞান", 70)]);

        let result = lexicon
            .suggest(
                "ক্মন",
                FstSuggestOptions {
                    max_distance: 1,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        assert!(result
            .candidates
            .iter()
            .any(|candidate| candidate.text == "কেমন"));
    }

    #[test]
    fn fst_suggest_returns_prefix_completions() {
        let lexicon = test_lexicon([("কমন", 20), ("কেমন", 100), ("যেমন", 30)]);

        let result = lexicon
            .suggest(
                "কেম",
                FstSuggestOptions {
                    max_distance: 0,
                    max_candidates: 16,
                    max_prefix_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST prefix lookup should succeed");

        assert!(result
            .candidates
            .iter()
            .any(|candidate| candidate.text == "কেমন"));
    }

    fn test_lexicon<const N: usize>(entries: [(&str, u64); N]) -> FstLexicon<Vec<u8>> {
        let mut entries = entries;
        entries.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
        let mut builder = fst::MapBuilder::memory();
        for (word, frequency) in entries {
            builder
                .insert(word.as_bytes(), frequency)
                .expect("fixture key should insert");
        }
        FstLexicon::from_map(builder.into_map())
    }
}
