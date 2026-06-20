use serde::Deserialize;

use super::ranker::{CorrectionCandidate, AUTOCORRECT_FEATURE_DIM};

const RERANKER_FEATURE_DIM: usize = AUTOCORRECT_FEATURE_DIM + 1;

#[derive(Debug, Clone)]
pub struct MlpReranker {
    feature_mean: [f32; RERANKER_FEATURE_DIM],
    feature_std: [f32; RERANKER_FEATURE_DIM],
    layer0_weight: Vec<[f32; RERANKER_FEATURE_DIM]>,
    layer0_bias: Vec<f32>,
    layer2_weight: Vec<f32>,
    layer2_bias: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlpRerankerError {
    InvalidJson,
    InvalidKind,
    InvalidFeatureDim { actual: usize },
    InvalidHiddenShape,
}

impl std::fmt::Display for MlpRerankerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidJson => write!(formatter, "invalid MLP reranker JSON artifact"),
            Self::InvalidKind => write!(formatter, "invalid MLP reranker artifact kind"),
            Self::InvalidFeatureDim { actual } => {
                write!(
                    formatter,
                    "invalid MLP reranker feature dimension: {actual}"
                )
            }
            Self::InvalidHiddenShape => {
                write!(formatter, "invalid MLP reranker hidden layer shape")
            }
        }
    }
}

impl std::error::Error for MlpRerankerError {}

#[derive(Debug, Deserialize)]
struct MlpRerankerArtifact {
    kind: String,
    feature_dim: usize,
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    layer0_weight: Vec<Vec<f32>>,
    layer0_bias: Vec<f32>,
    layer2_weight: Vec<f32>,
    layer2_bias: f32,
}

impl MlpReranker {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, MlpRerankerError> {
        let artifact: MlpRerankerArtifact =
            serde_json::from_slice(bytes).map_err(|_| MlpRerankerError::InvalidJson)?;
        if artifact.kind != "obadh_candidate_mlp_reranker" {
            return Err(MlpRerankerError::InvalidKind);
        }
        if artifact.feature_dim != RERANKER_FEATURE_DIM {
            return Err(MlpRerankerError::InvalidFeatureDim {
                actual: artifact.feature_dim,
            });
        }

        let feature_mean = fixed_array(artifact.feature_mean)?;
        let feature_std = fixed_array(artifact.feature_std)?;
        let hidden_size = artifact.layer0_bias.len();
        if hidden_size == 0
            || artifact.layer0_weight.len() != hidden_size
            || artifact.layer2_weight.len() != hidden_size
        {
            return Err(MlpRerankerError::InvalidHiddenShape);
        }
        let mut layer0_weight = Vec::with_capacity(hidden_size);
        for row in artifact.layer0_weight {
            layer0_weight.push(fixed_array(row)?);
        }

        Ok(Self {
            feature_mean,
            feature_std,
            layer0_weight,
            layer0_bias: artifact.layer0_bias,
            layer2_weight: artifact.layer2_weight,
            layer2_bias: artifact.layer2_bias,
        })
    }

    pub fn score_candidate(&self, candidate: &CorrectionCandidate) -> f32 {
        let raw = reranker_features(candidate);
        let mut output = self.layer2_bias;

        for ((weights, bias), output_weight) in self
            .layer0_weight
            .iter()
            .zip(self.layer0_bias.iter())
            .zip(self.layer2_weight.iter())
        {
            let mut hidden = *bias;
            for index in 0..RERANKER_FEATURE_DIM {
                let std = self.feature_std[index].max(1.0e-6);
                hidden += ((raw[index] - self.feature_mean[index]) / std) * weights[index];
            }
            output += hidden.max(0.0) * output_weight;
        }

        output
    }

    pub fn sort_candidates(&self, candidates: &mut [CorrectionCandidate]) {
        let scores = candidates
            .iter()
            .map(|candidate| self.score_candidate(candidate))
            .collect::<Vec<_>>();
        let mut order = (0..candidates.len()).collect::<Vec<_>>();
        order.sort_by(|left, right| {
            scores[*right]
                .total_cmp(&scores[*left])
                .then_with(|| candidates[*right].score.cmp(&candidates[*left].score))
                .then_with(|| candidates[*left].text.cmp(&candidates[*right].text))
        });

        let original = candidates.to_vec();
        for (slot, index) in order.into_iter().enumerate() {
            candidates[slot] = original[index].clone();
        }
    }
}

fn fixed_array<const N: usize>(values: Vec<f32>) -> Result<[f32; N], MlpRerankerError> {
    values
        .try_into()
        .map_err(|_| MlpRerankerError::InvalidFeatureDim { actual: 0 })
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

#[cfg(test)]
mod tests {
    use super::MlpReranker;
    use crate::autocorrect::{AutocorrectEngine, CorrectionSource, LexiconEntry};

    #[test]
    fn mlp_reranker_loads_and_sorts_candidates() {
        let artifact = br#"{
            "kind":"obadh_candidate_mlp_reranker",
            "feature_dim":11,
            "hidden_size":1,
            "feature_mean":[0,0,0,0,0,0,0,0,0,0,0],
            "feature_std":[1,1,1,1,1,1,1,1,1,1,1],
            "layer0_weight":[[0,0,0,0,0,0,0,0,0,0,1]],
            "layer0_bias":[0],
            "layer2_weight":[1],
            "layer2_bias":0
        }"#;
        let reranker = MlpReranker::from_json_bytes(artifact).expect("artifact should load");
        let engine = AutocorrectEngine::from_entries([
            LexiconEntry::new("আমি", 10),
            LexiconEntry::new("আমার", 1),
        ]);
        let mut candidates = engine.suggest("আমী");

        reranker.sort_candidates(&mut candidates);

        assert_eq!(candidates[0].source, CorrectionSource::LexiconEdit);
    }
}
