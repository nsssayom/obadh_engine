use std::collections::HashMap;

use serde::Deserialize;

use super::ranker::{CorrectionCandidate, AUTOCORRECT_FEATURE_DIM};

const PAD: usize = 0;
const UNK: usize = 1;
const RERANKER_FEATURE_DIM: usize = AUTOCORRECT_FEATURE_DIM + 1;

#[derive(Debug, Clone)]
pub struct CharCandidateReranker {
    source_vocab: HashMap<char, usize>,
    candidate_vocab: HashMap<char, usize>,
    feature_mean: [f32; RERANKER_FEATURE_DIM],
    feature_std: [f32; RERANKER_FEATURE_DIM],
    source_embedding: Vec<f32>,
    candidate_embedding: Vec<f32>,
    embed_size: usize,
    layer0_weight: Vec<f32>,
    layer0_bias: Vec<f32>,
    layer2_weight: Vec<f32>,
    layer2_bias: f32,
    hidden_size: usize,
    pair_dim: usize,
    replacement_policy: Option<CharReplacementPolicy>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub struct CharReplacementPolicy {
    pub min_score: f32,
    pub min_margin: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredCorrectionCandidate {
    pub candidate: CorrectionCandidate,
    pub model_score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CharCandidateRerankerError {
    InvalidJson,
    InvalidKind,
    InvalidFeatureDim { actual: usize },
    InvalidVocabKey { key: String },
    InvalidVocabIndex { index: usize, rows: usize },
    InvalidEmbeddingShape,
    InvalidHiddenShape,
    InvalidReplacementPolicy,
}

impl std::fmt::Display for CharCandidateRerankerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidJson => write!(formatter, "invalid char reranker JSON artifact"),
            Self::InvalidKind => write!(formatter, "invalid char reranker artifact kind"),
            Self::InvalidFeatureDim { actual } => {
                write!(
                    formatter,
                    "invalid char reranker feature dimension: {actual}"
                )
            }
            Self::InvalidVocabKey { key } => {
                write!(formatter, "invalid char reranker vocab key: {key}")
            }
            Self::InvalidVocabIndex { index, rows } => write!(
                formatter,
                "char reranker vocab index {index} exceeds embedding rows {rows}"
            ),
            Self::InvalidEmbeddingShape => {
                write!(formatter, "invalid char reranker embedding shape")
            }
            Self::InvalidHiddenShape => write!(formatter, "invalid char reranker hidden shape"),
            Self::InvalidReplacementPolicy => {
                write!(formatter, "invalid char reranker replacement policy")
            }
        }
    }
}

impl std::error::Error for CharCandidateRerankerError {}

impl CharReplacementPolicy {
    pub const fn high_precision() -> Self {
        Self {
            min_score: 2.0,
            min_margin: 10.0,
        }
    }

    pub fn is_valid(self) -> bool {
        self.min_score.is_finite() && self.min_margin.is_finite() && self.min_margin >= 0.0
    }
}

#[derive(Debug, Deserialize)]
struct CharCandidateRerankerArtifact {
    kind: String,
    source_vocab: HashMap<String, usize>,
    candidate_vocab: HashMap<String, usize>,
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    source_embedding: Vec<Vec<f32>>,
    candidate_embedding: Vec<Vec<f32>>,
    layer0_weight: Vec<Vec<f32>>,
    layer0_bias: Vec<f32>,
    layer2_weight: Vec<f32>,
    layer2_bias: f32,
    replacement_policy: Option<CharReplacementPolicy>,
}

impl CharCandidateReranker {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, CharCandidateRerankerError> {
        let artifact: CharCandidateRerankerArtifact =
            serde_json::from_slice(bytes).map_err(|_| CharCandidateRerankerError::InvalidJson)?;
        if artifact.kind != "obadh_char_candidate_reranker" {
            return Err(CharCandidateRerankerError::InvalidKind);
        }
        if artifact
            .replacement_policy
            .is_some_and(|policy| !policy.is_valid())
        {
            return Err(CharCandidateRerankerError::InvalidReplacementPolicy);
        }

        let feature_mean = fixed_array(artifact.feature_mean)?;
        let feature_std = fixed_array(artifact.feature_std)?;
        let source_embedding_rows = artifact.source_embedding.len();
        let candidate_embedding_rows = artifact.candidate_embedding.len();
        let embed_size = embedding_size(&artifact.source_embedding)?;
        if embed_size == 0 || embedding_size(&artifact.candidate_embedding)? != embed_size {
            return Err(CharCandidateRerankerError::InvalidEmbeddingShape);
        }

        let source_vocab = parse_vocab(artifact.source_vocab, source_embedding_rows)?;
        let candidate_vocab = parse_vocab(artifact.candidate_vocab, candidate_embedding_rows)?;
        let pair_dim = RERANKER_FEATURE_DIM + embed_size * 4;
        let hidden_size = artifact.layer0_bias.len();
        if hidden_size == 0
            || artifact.layer0_weight.len() != hidden_size
            || artifact.layer2_weight.len() != hidden_size
            || artifact
                .layer0_weight
                .iter()
                .any(|row| row.len() != pair_dim)
        {
            return Err(CharCandidateRerankerError::InvalidHiddenShape);
        }

        Ok(Self {
            source_vocab,
            candidate_vocab,
            feature_mean,
            feature_std,
            source_embedding: flatten_matrix(artifact.source_embedding),
            candidate_embedding: flatten_matrix(artifact.candidate_embedding),
            embed_size,
            layer0_weight: flatten_matrix(artifact.layer0_weight),
            layer0_bias: artifact.layer0_bias,
            layer2_weight: artifact.layer2_weight,
            layer2_bias: artifact.layer2_bias,
            hidden_size,
            pair_dim,
            replacement_policy: artifact.replacement_policy,
        })
    }

    pub fn replacement_policy(&self) -> Option<CharReplacementPolicy> {
        self.replacement_policy
    }

    pub fn calibrated_replacement_candidate(
        &self,
        source: &str,
        candidates: &[CorrectionCandidate],
    ) -> Option<CorrectionCandidate> {
        self.replacement_candidate(
            source,
            candidates,
            self.replacement_policy
                .unwrap_or_else(CharReplacementPolicy::high_precision),
        )
    }

    pub fn rank_candidates(
        &self,
        source: &str,
        candidates: &[CorrectionCandidate],
    ) -> Vec<ScoredCorrectionCandidate> {
        let source_repr = self.source_repr(source);
        let mut ranked = candidates
            .iter()
            .map(|candidate| ScoredCorrectionCandidate {
                candidate: candidate.clone(),
                model_score: self.score_candidate_with_source_repr(&source_repr, candidate),
            })
            .collect::<Vec<_>>();
        ranked.sort_by(scored_candidate_order);
        ranked
    }

    pub fn score_candidate(&self, source: &str, candidate: &CorrectionCandidate) -> f32 {
        let source_repr = self.source_repr(source);
        self.score_candidate_with_source_repr(&source_repr, candidate)
    }

    fn source_repr(&self, source: &str) -> Vec<f32> {
        pooled_embedding(
            source,
            &self.source_vocab,
            &self.source_embedding,
            self.embed_size,
        )
    }

    fn score_candidate_with_source_repr(
        &self,
        source_repr: &[f32],
        candidate: &CorrectionCandidate,
    ) -> f32 {
        let numeric =
            normalized_reranker_features(candidate, &self.feature_mean, &self.feature_std);
        let candidate_repr = pooled_embedding(
            &candidate.text,
            &self.candidate_vocab,
            &self.candidate_embedding,
            self.embed_size,
        );
        let mut output = self.layer2_bias;

        for hidden_index in 0..self.hidden_size {
            let weight_start = hidden_index * self.pair_dim;
            let weights = &self.layer0_weight[weight_start..weight_start + self.pair_dim];
            let mut hidden = self.layer0_bias[hidden_index];
            let mut weight_index = 0_usize;
            for value in numeric {
                hidden += value * weights[weight_index];
                weight_index += 1;
            }
            for value in source_repr {
                hidden += *value * weights[weight_index];
                weight_index += 1;
            }
            for value in &candidate_repr {
                hidden += value * weights[weight_index];
                weight_index += 1;
            }
            for index in 0..self.embed_size {
                hidden += source_repr[index] * candidate_repr[index] * weights[weight_index];
                weight_index += 1;
            }
            for index in 0..self.embed_size {
                hidden +=
                    (source_repr[index] - candidate_repr[index]).abs() * weights[weight_index];
                weight_index += 1;
            }
            output += hidden.max(0.0) * self.layer2_weight[hidden_index];
        }

        output
    }

    pub fn sort_candidates(&self, source: &str, candidates: &mut [CorrectionCandidate]) {
        let ranked = self.rank_candidates(source, candidates);
        for (slot, scored) in ranked.into_iter().enumerate() {
            candidates[slot] = scored.candidate;
        }
    }

    pub fn calibrated_replacement_from_ranked_candidates(
        &self,
        ranked: &[ScoredCorrectionCandidate],
    ) -> Option<CorrectionCandidate> {
        self.replacement_from_ranked_candidates(
            ranked,
            self.replacement_policy
                .unwrap_or_else(CharReplacementPolicy::high_precision),
        )
    }

    pub fn replacement_from_ranked_candidates(
        &self,
        ranked: &[ScoredCorrectionCandidate],
        policy: CharReplacementPolicy,
    ) -> Option<CorrectionCandidate> {
        let top = ranked.first()?;
        if top.candidate.source == super::ranker::CorrectionSource::NoChange {
            return None;
        }

        if top.model_score < policy.min_score {
            return None;
        }

        let next_score = ranked
            .get(1)
            .map(|candidate| candidate.model_score)
            .unwrap_or(f32::NEG_INFINITY);
        if top.model_score - next_score < policy.min_margin {
            return None;
        }

        Some(top.candidate.clone())
    }

    pub fn replacement_candidate(
        &self,
        source: &str,
        candidates: &[CorrectionCandidate],
        policy: CharReplacementPolicy,
    ) -> Option<CorrectionCandidate> {
        let top = candidates.first()?;
        if top.source == super::ranker::CorrectionSource::NoChange {
            return None;
        }

        let top_score = self.score_candidate(source, top);
        if top_score < policy.min_score {
            return None;
        }

        let next_score = candidates
            .get(1)
            .map(|candidate| self.score_candidate(source, candidate))
            .unwrap_or(f32::NEG_INFINITY);
        if top_score - next_score < policy.min_margin {
            return None;
        }

        Some(top.clone())
    }
}

fn scored_candidate_order(
    left: &ScoredCorrectionCandidate,
    right: &ScoredCorrectionCandidate,
) -> std::cmp::Ordering {
    right
        .model_score
        .total_cmp(&left.model_score)
        .then_with(|| right.candidate.score.cmp(&left.candidate.score))
        .then_with(|| left.candidate.text.cmp(&right.candidate.text))
}

fn fixed_array<const N: usize>(values: Vec<f32>) -> Result<[f32; N], CharCandidateRerankerError> {
    values.try_into().map_err(
        |values: Vec<f32>| CharCandidateRerankerError::InvalidFeatureDim {
            actual: values.len(),
        },
    )
}

fn parse_vocab(
    raw: HashMap<String, usize>,
    embedding_rows: usize,
) -> Result<HashMap<char, usize>, CharCandidateRerankerError> {
    let mut vocab = HashMap::with_capacity(raw.len());
    for (key, index) in raw {
        let mut chars = key.chars();
        let Some(ch) = chars.next() else {
            return Err(CharCandidateRerankerError::InvalidVocabKey { key });
        };
        if chars.next().is_some() {
            return Err(CharCandidateRerankerError::InvalidVocabKey { key });
        }
        if index >= embedding_rows {
            return Err(CharCandidateRerankerError::InvalidVocabIndex {
                index,
                rows: embedding_rows,
            });
        }
        vocab.insert(ch, index);
    }
    Ok(vocab)
}

fn embedding_size(embedding: &[Vec<f32>]) -> Result<usize, CharCandidateRerankerError> {
    if embedding.len() <= UNK {
        return Err(CharCandidateRerankerError::InvalidEmbeddingShape);
    }
    let size = embedding[PAD].len();
    if embedding.iter().any(|row| row.len() != size) {
        return Err(CharCandidateRerankerError::InvalidEmbeddingShape);
    }
    Ok(size)
}

fn flatten_matrix(rows: Vec<Vec<f32>>) -> Vec<f32> {
    rows.into_iter().flatten().collect()
}

fn pooled_embedding(
    text: &str,
    vocab: &HashMap<char, usize>,
    embedding: &[f32],
    embed_size: usize,
) -> Vec<f32> {
    let mut pooled = vec![0.0; embed_size];
    let mut count = 0_usize;

    for ch in text.chars() {
        let index = vocab.get(&ch).copied().unwrap_or(UNK);
        let start = index * embed_size;
        let row = &embedding[start..start + embed_size];
        for (slot, value) in pooled.iter_mut().zip(row.iter()) {
            *slot += value;
        }
        count += 1;
    }

    if count == 0 {
        let start = UNK * embed_size;
        let row = &embedding[start..start + embed_size];
        for (slot, value) in pooled.iter_mut().zip(row.iter()) {
            *slot += value;
        }
        count = 1;
    }

    let scale = 1.0 / count as f32;
    for value in &mut pooled {
        *value *= scale;
    }
    pooled
}

fn reranker_features(candidate: &CorrectionCandidate) -> [f32; RERANKER_FEATURE_DIM] {
    let base = candidate.features.as_i16_array();
    let mut features = [0.0; RERANKER_FEATURE_DIM];
    for (index, value) in base.iter().enumerate() {
        features[index] = f32::from(*value);
    }
    features[AUTOCORRECT_FEATURE_DIM] = candidate.score as f32;
    features
}

fn normalized_reranker_features(
    candidate: &CorrectionCandidate,
    mean: &[f32; RERANKER_FEATURE_DIM],
    std: &[f32; RERANKER_FEATURE_DIM],
) -> [f32; RERANKER_FEATURE_DIM] {
    let raw = reranker_features(candidate);
    let mut normalized = [0.0; RERANKER_FEATURE_DIM];
    for index in 0..RERANKER_FEATURE_DIM {
        normalized[index] = (raw[index] - mean[index]) / std[index].max(1.0e-6);
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::{CharCandidateReranker, CharCandidateRerankerError, CharReplacementPolicy};
    use crate::autocorrect::{AutocorrectEngine, LexiconEntry};

    #[test]
    fn char_reranker_scores_candidate_characters() {
        let artifact = r#"{
            "kind":"obadh_char_candidate_reranker",
            "source_vocab":{"a":2},
            "candidate_vocab":{"আ":2,"ম":3,"ি":4},
            "feature_mean":[0,0,0,0,0,0,0,0,0,0,0],
            "feature_std":[1,1,1,1,1,1,1,1,1,1,1],
            "source_embedding":[[0],[0],[0]],
            "candidate_embedding":[[0],[0],[0],[0],[9]],
            "layer0_weight":[[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]],
            "layer0_bias":[0],
            "layer2_weight":[1],
            "layer2_bias":0
        }"#;
        let reranker = CharCandidateReranker::from_json_bytes(artifact.as_bytes())
            .expect("artifact should load");
        let engine = AutocorrectEngine::from_entries([
            LexiconEntry::new("আমি", 1),
            LexiconEntry::new("আমার", 1000),
        ]);
        let candidates = engine.suggest("আমী");
        let ami = candidates
            .iter()
            .find(|candidate| candidate.text == "আমি")
            .expect("আমি candidate should exist");
        let amar = candidates
            .iter()
            .find(|candidate| candidate.text == "আমার")
            .expect("আমার candidate should exist");

        assert!(reranker.score_candidate("ami", ami) > reranker.score_candidate("ami", amar));
    }

    #[test]
    fn char_reranker_ranks_with_cached_scores() {
        let artifact = r#"{
            "kind":"obadh_char_candidate_reranker",
            "source_vocab":{"a":2},
            "candidate_vocab":{"আ":2,"ম":3,"ি":4},
            "feature_mean":[0,0,0,0,0,0,0,0,0,0,0],
            "feature_std":[1,1,1,1,1,1,1,1,1,1,1],
            "source_embedding":[[0],[0],[0]],
            "candidate_embedding":[[0],[0],[0],[0],[9]],
            "layer0_weight":[[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]],
            "layer0_bias":[0],
            "layer2_weight":[1],
            "layer2_bias":0,
            "replacement_policy":{"min_score":1.0,"min_margin":1.0}
        }"#;
        let reranker = CharCandidateReranker::from_json_bytes(artifact.as_bytes())
            .expect("artifact should load");
        let engine = AutocorrectEngine::from_entries([
            LexiconEntry::new("আমি", 1),
            LexiconEntry::new("আমার", 1000),
        ]);

        let ranked = reranker.rank_candidates("ami", &engine.suggest("আমী"));

        assert_eq!(ranked[0].candidate.text, "আমি");
        assert_eq!(
            reranker.calibrated_replacement_from_ranked_candidates(&ranked),
            Some(ranked[0].candidate.clone())
        );
        assert_eq!(
            ranked[0].model_score,
            reranker.score_candidate("ami", &ranked[0].candidate)
        );
    }

    #[test]
    fn char_reranker_reads_optional_replacement_policy() {
        let artifact = r#"{
            "kind":"obadh_char_candidate_reranker",
            "source_vocab":{"a":2},
            "candidate_vocab":{"আ":2},
            "feature_mean":[0,0,0,0,0,0,0,0,0,0,0],
            "feature_std":[1,1,1,1,1,1,1,1,1,1,1],
            "source_embedding":[[0],[0],[0]],
            "candidate_embedding":[[0],[0],[0]],
            "layer0_weight":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            "layer0_bias":[0],
            "layer2_weight":[1],
            "layer2_bias":0,
            "replacement_policy":{"min_score":1.5,"min_margin":2.5}
        }"#;
        let reranker = CharCandidateReranker::from_json_bytes(artifact.as_bytes())
            .expect("artifact should load");

        assert_eq!(
            reranker.replacement_policy(),
            Some(CharReplacementPolicy {
                min_score: 1.5,
                min_margin: 2.5
            })
        );
    }

    #[test]
    fn char_reranker_rejects_invalid_replacement_policy() {
        let artifact = r#"{
            "kind":"obadh_char_candidate_reranker",
            "source_vocab":{"a":2},
            "candidate_vocab":{"আ":2},
            "feature_mean":[0,0,0,0,0,0,0,0,0,0,0],
            "feature_std":[1,1,1,1,1,1,1,1,1,1,1],
            "source_embedding":[[0],[0],[0]],
            "candidate_embedding":[[0],[0],[0]],
            "layer0_weight":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
            "layer0_bias":[0],
            "layer2_weight":[1],
            "layer2_bias":0,
            "replacement_policy":{"min_score":1.5,"min_margin":-1.0}
        }"#;

        assert!(matches!(
            CharCandidateReranker::from_json_bytes(artifact.as_bytes()),
            Err(CharCandidateRerankerError::InvalidReplacementPolicy)
        ));
    }
}
