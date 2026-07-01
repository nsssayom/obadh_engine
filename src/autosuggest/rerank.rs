use serde::Serialize;
use std::error::Error;
use std::fmt;

use super::lm::{AutosuggestCandidateId, DEFAULT_AUTOSUGGEST_CANDIDATES};

pub const DEFAULT_AUTOSUGGEST_RERANK_LOCKED_PREFIX: usize = 1;
pub const DEFAULT_AUTOSUGGEST_RERANK_RANK_PENALTY: f32 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutosuggestRerankOptions {
    pub max_candidates: usize,
    pub locked_prefix: usize,
    pub rank_penalty: f32,
}

impl Default for AutosuggestRerankOptions {
    fn default() -> Self {
        Self {
            max_candidates: DEFAULT_AUTOSUGGEST_CANDIDATES,
            locked_prefix: DEFAULT_AUTOSUGGEST_RERANK_LOCKED_PREFIX,
            rank_penalty: DEFAULT_AUTOSUGGEST_RERANK_RANK_PENALTY,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AutosuggestScoredCandidateId {
    pub candidate: AutosuggestCandidateId,
    pub original_rank: usize,
    pub model_score: f32,
    pub rerank_score: f32,
}

impl AutosuggestScoredCandidateId {
    pub fn candidate_id(self) -> AutosuggestCandidateId {
        self.candidate
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutosuggestRerankError {
    ScoreLengthMismatch { candidates: usize, scores: usize },
}

impl fmt::Display for AutosuggestRerankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScoreLengthMismatch { candidates, scores } => write!(
                f,
                "autosuggest reranker received {scores} model scores for {candidates} candidates"
            ),
        }
    }
}

impl Error for AutosuggestRerankError {}

pub fn rerank_candidate_ids_with_scores_into(
    candidates: &[AutosuggestCandidateId],
    model_scores: &[f32],
    options: AutosuggestRerankOptions,
    output: &mut Vec<AutosuggestScoredCandidateId>,
) -> Result<(), AutosuggestRerankError> {
    if model_scores.len() != candidates.len() {
        output.clear();
        return Err(AutosuggestRerankError::ScoreLengthMismatch {
            candidates: candidates.len(),
            scores: model_scores.len(),
        });
    }
    rerank_candidate_ids_with_fixed_scores_into(candidates, model_scores, options, output)
}

pub fn rerank_candidate_ids_with_fixed_scores_into(
    candidates: &[AutosuggestCandidateId],
    model_scores: &[f32],
    options: AutosuggestRerankOptions,
    output: &mut Vec<AutosuggestScoredCandidateId>,
) -> Result<(), AutosuggestRerankError> {
    output.clear();
    if candidates.is_empty() {
        return Ok(());
    }
    if model_scores.len() < candidates.len() {
        return Err(AutosuggestRerankError::ScoreLengthMismatch {
            candidates: candidates.len(),
            scores: model_scores.len(),
        });
    }

    let rank_penalty = if options.rank_penalty.is_finite() && options.rank_penalty > 0.0 {
        options.rank_penalty
    } else {
        0.0
    };
    output.extend(candidates.iter().enumerate().map(|(index, candidate)| {
        let model_score = finite_model_score(model_scores[index]);
        AutosuggestScoredCandidateId {
            candidate: *candidate,
            original_rank: index,
            model_score,
            rerank_score: model_score - rank_penalty * index as f32,
        }
    }));

    let visible = options.max_candidates.max(1).min(output.len());
    let locked = options.locked_prefix.min(visible);
    output[locked..].sort_by(scored_candidate_order);
    output.truncate(visible);
    Ok(())
}

fn finite_model_score(score: f32) -> f32 {
    if score.is_finite() {
        score
    } else {
        f32::NEG_INFINITY
    }
}

fn scored_candidate_order(
    left: &AutosuggestScoredCandidateId,
    right: &AutosuggestScoredCandidateId,
) -> std::cmp::Ordering {
    right
        .rerank_score
        .total_cmp(&left.rerank_score)
        .then_with(|| left.original_rank.cmp(&right.original_rank))
        .then_with(|| left.candidate.token_id.cmp(&right.candidate.token_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autosuggest::AutosuggestSource;

    fn candidate(token_id: u32, score: i32) -> AutosuggestCandidateId {
        AutosuggestCandidateId {
            token_id,
            source: AutosuggestSource::Fourgram,
            count: 1,
            score,
        }
    }

    #[test]
    fn locked_first_rerank_preserves_static_top_candidate() {
        let candidates = [
            candidate(10, 100),
            candidate(20, 90),
            candidate(30, 80),
            candidate(40, 70),
        ];
        let model_scores = [0.0, 0.1, 5.0, 4.0];
        let mut output = Vec::new();

        rerank_candidate_ids_with_scores_into(
            &candidates,
            &model_scores,
            AutosuggestRerankOptions {
                max_candidates: 3,
                locked_prefix: 1,
                rank_penalty: 0.25,
            },
            &mut output,
        )
        .unwrap();

        assert_eq!(
            output
                .iter()
                .map(|candidate| candidate.candidate.token_id)
                .collect::<Vec<_>>(),
            vec![10, 30, 40]
        );
        assert_eq!(output[0].original_rank, 0);
        assert!(output[1].rerank_score > output[2].rerank_score);
    }

    #[test]
    fn rank_penalty_breaks_close_model_scores_toward_static_order() {
        let candidates = [candidate(10, 100), candidate(20, 90), candidate(30, 80)];
        let model_scores = [0.0, 1.0, 1.1];
        let mut output = Vec::new();

        rerank_candidate_ids_with_scores_into(
            &candidates,
            &model_scores,
            AutosuggestRerankOptions {
                max_candidates: 3,
                locked_prefix: 1,
                rank_penalty: 0.25,
            },
            &mut output,
        )
        .unwrap();

        assert_eq!(
            output
                .iter()
                .map(|candidate| candidate.candidate.token_id)
                .collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
    }

    #[test]
    fn non_finite_model_score_cannot_promote_candidate() {
        let candidates = [candidate(10, 100), candidate(20, 90), candidate(30, 80)];
        let model_scores = [0.0, f32::NAN, 1.0];
        let mut output = Vec::new();

        rerank_candidate_ids_with_scores_into(
            &candidates,
            &model_scores,
            AutosuggestRerankOptions {
                max_candidates: 3,
                locked_prefix: 0,
                rank_penalty: 0.0,
            },
            &mut output,
        )
        .unwrap();

        assert_eq!(
            output
                .iter()
                .map(|candidate| candidate.candidate.token_id)
                .collect::<Vec<_>>(),
            vec![30, 10, 20]
        );
        assert_eq!(output[2].model_score, f32::NEG_INFINITY);
    }

    #[test]
    fn score_length_mismatch_is_rejected() {
        let mut output = Vec::new();
        let error = rerank_candidate_ids_with_scores_into(
            &[candidate(10, 100), candidate(20, 90)],
            &[0.0],
            AutosuggestRerankOptions::default(),
            &mut output,
        )
        .unwrap_err();

        assert_eq!(
            error,
            AutosuggestRerankError::ScoreLengthMismatch {
                candidates: 2,
                scores: 1
            }
        );
        assert!(output.is_empty());
    }

    #[test]
    fn fixed_score_rerank_accepts_padded_model_output() {
        let candidates = [candidate(10, 100), candidate(20, 90)];
        let mut output = Vec::new();

        assert!(rerank_candidate_ids_with_scores_into(
            &candidates,
            &[0.0, 1.0, 2.0],
            AutosuggestRerankOptions::default(),
            &mut output,
        )
        .is_err());

        rerank_candidate_ids_with_fixed_scores_into(
            &candidates,
            &[0.0, 1.0, 2.0],
            AutosuggestRerankOptions {
                max_candidates: 2,
                locked_prefix: 0,
                rank_penalty: 0.0,
            },
            &mut output,
        )
        .unwrap();

        assert_eq!(
            output
                .iter()
                .map(|candidate| candidate.candidate.token_id)
                .collect::<Vec<_>>(),
            vec![20, 10]
        );
    }
}
