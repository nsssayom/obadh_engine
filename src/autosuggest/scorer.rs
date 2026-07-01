use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Read;
use std::mem;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use sha2::{Digest, Sha256};

use super::adaptive::AutosuggestSession;
use super::artifact::AutosuggestArtifactError;
use super::lm::{
    scorer_candidate_i32s_for_candidates_into, scorer_candidate_ids_for_candidates_into,
    AutosuggestCandidateId, AutosuggestContext, AutosuggestLm, AutosuggestOptions,
    AutosuggestRerankInputMetadata, AutosuggestSource, AUTOSUGGEST_BOS_ID, AUTOSUGGEST_PAD_ID,
    AUTOSUGGEST_UNK_ID, DEFAULT_AUTOSUGGEST_CANDIDATES, MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS,
};
use super::rerank::{
    rerank_candidate_ids_with_fixed_scores_into, AutosuggestRerankError, AutosuggestRerankOptions,
    AutosuggestScoredCandidateId, DEFAULT_AUTOSUGGEST_RERANK_LOCKED_PREFIX,
    DEFAULT_AUTOSUGGEST_RERANK_RANK_PENALTY,
};

pub const AUTOSUGGEST_SCORER_PACKAGE_KIND: &str = "obadh-autosuggest-scorer-package";
pub const AUTOSUGGEST_SCORER_RUNTIME_ROLE: &str = "next_word_candidate_rerank";
pub const AUTOSUGGEST_SCORER_MANIFEST_VERSION: u32 = 1;
pub const AUTOSUGGEST_SCORER_TOKEN_ID_DTYPE: &str = "uint32";
pub const AUTOSUGGEST_SCORER_ONNX_INPUT_DTYPE: &str = "int64";
pub const AUTOSUGGEST_SCORER_COREML_INPUT_DTYPE: &str = "int32";
pub const AUTOSUGGEST_SCORER_SCORE_DTYPE: &str = "float32";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestScorerManifest {
    pub artifact: String,
    pub version: u32,
    pub runtime_role: String,
    pub runtime_contract: AutosuggestScorerRuntimeContract,
    pub ngram: AutosuggestScorerNgram,
    pub scorer: AutosuggestScorerModel,
    pub quality: AutosuggestScorerQuality,
}

impl AutosuggestScorerManifest {
    pub fn from_json_str(input: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(input)
    }

    pub fn validate_for_lm<D: AsRef<[u8]>>(
        &self,
        lm: &AutosuggestLm<D>,
    ) -> Result<AutosuggestScorerCompatibility, AutosuggestScorerManifestError> {
        require_equal_str("artifact", AUTOSUGGEST_SCORER_PACKAGE_KIND, &self.artifact)?;
        require_equal_u32("version", AUTOSUGGEST_SCORER_MANIFEST_VERSION, self.version)?;
        require_equal_str(
            "runtime_role",
            AUTOSUGGEST_SCORER_RUNTIME_ROLE,
            &self.runtime_role,
        )?;

        let model_info = lm.model_info();
        let artifact_fingerprint = format!("{:016x}", lm.artifact_fingerprint());
        require_equal_usize("ngram.bytes", model_info.artifact_bytes, self.ngram.bytes)?;
        require_equal_str(
            "ngram.artifact_fingerprint",
            &artifact_fingerprint,
            &self.ngram.artifact_fingerprint,
        )?;
        require_equal_usize(
            "ngram.vocab_size",
            model_info.vocab_size,
            self.ngram.vocab_size,
        )?;
        require_equal_u32(
            "ngram.vocab_fingerprint",
            model_info.vocab_fingerprint,
            self.ngram.vocab_fingerprint,
        )?;
        require_equal_usize(
            "ngram.candidate_record_len",
            model_info.candidate_record_len,
            self.ngram.candidate_record_len,
        )?;

        let contract = &self.runtime_contract;
        require_equal_str(
            "runtime_contract.token_id_dtype",
            AUTOSUGGEST_SCORER_TOKEN_ID_DTYPE,
            &contract.token_id_dtype,
        )?;
        require_equal_str(
            "runtime_contract.onnx_input_dtype",
            AUTOSUGGEST_SCORER_ONNX_INPUT_DTYPE,
            &contract.onnx_input_dtype,
        )?;
        require_equal_str(
            "runtime_contract.coreml_input_dtype",
            AUTOSUGGEST_SCORER_COREML_INPUT_DTYPE,
            &contract.coreml_input_dtype,
        )?;
        require_equal_str(
            "runtime_contract.scores_dtype",
            AUTOSUGGEST_SCORER_SCORE_DTYPE,
            &contract.scores_dtype,
        )?;
        require_shape(
            "runtime_contract.context_ids_shape",
            contract.context_ids_shape,
        )?;
        require_shape(
            "runtime_contract.candidate_ids_shape",
            contract.candidate_ids_shape,
        )?;
        require_shape("runtime_contract.scores_shape", contract.scores_shape)?;
        require_equal_usize("runtime_contract.batch_size", 1, contract.batch_size)?;
        require_equal_usize(
            "runtime_contract.context_ids_shape[0]",
            contract.batch_size,
            contract.context_ids_shape[0],
        )?;
        require_equal_usize(
            "runtime_contract.candidate_ids_shape[0]",
            contract.batch_size,
            contract.candidate_ids_shape[0],
        )?;
        require_equal_usize(
            "runtime_contract.scores_shape[0]",
            contract.batch_size,
            contract.scores_shape[0],
        )?;
        require_equal_usize(
            "runtime_contract.context_ids_shape[1]",
            self.scorer.context_window,
            contract.context_ids_shape[1],
        )?;
        require_equal_usize(
            "runtime_contract.candidate_ids_shape[1]",
            self.scorer.pool_k,
            contract.candidate_ids_shape[1],
        )?;
        require_equal_usize(
            "runtime_contract.scores_shape[1]",
            self.scorer.pool_k,
            contract.scores_shape[1],
        )?;
        require_equal_u32(
            "runtime_contract.pad_id",
            AUTOSUGGEST_PAD_ID,
            contract.pad_id,
        )?;
        require_equal_u32(
            "runtime_contract.bos_id",
            AUTOSUGGEST_BOS_ID,
            contract.bos_id,
        )?;
        require_equal_u32(
            "runtime_contract.unk_id",
            AUTOSUGGEST_UNK_ID,
            contract.unk_id,
        )?;
        require_equal_usize(
            "runtime_contract.locked_prefix",
            DEFAULT_AUTOSUGGEST_RERANK_LOCKED_PREFIX,
            contract.locked_prefix,
        )?;
        require_equal_f32(
            "runtime_contract.rank_penalty",
            DEFAULT_AUTOSUGGEST_RERANK_RANK_PENALTY,
            contract.rank_penalty,
        )?;
        require_equal_usize(
            "runtime_contract.visible_candidates",
            DEFAULT_AUTOSUGGEST_CANDIDATES,
            contract.visible_candidates,
        )?;

        if self.scorer.context_window > MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS {
            return Err(AutosuggestScorerManifestError::Invalid {
                field: "scorer.context_window",
                reason: format!(
                    "exceeds runtime context buffer {}",
                    MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS
                ),
            });
        }
        if self.ngram.max_candidates_per_prefix < self.scorer.pool_k {
            return Err(AutosuggestScorerManifestError::Invalid {
                field: "ngram.max_candidates_per_prefix",
                reason: format!("is smaller than scorer.pool_k {}", self.scorer.pool_k),
            });
        }
        if contract.visible_candidates > self.scorer.pool_k {
            return Err(AutosuggestScorerManifestError::Invalid {
                field: "runtime_contract.visible_candidates",
                reason: format!("exceeds scorer.pool_k {}", self.scorer.pool_k),
            });
        }
        if contract.locked_prefix > contract.visible_candidates {
            return Err(AutosuggestScorerManifestError::Invalid {
                field: "runtime_contract.locked_prefix",
                reason: format!(
                    "exceeds visible candidate count {}",
                    contract.visible_candidates
                ),
            });
        }
        let top5_gain = self.quality.top5_all_target_gain;
        if !top5_gain.is_finite() || top5_gain <= 0.0 {
            return Err(AutosuggestScorerManifestError::Invalid {
                field: "quality.top5_all_target_gain",
                reason: "must be finite and positive".to_string(),
            });
        }
        let top10_gain = self.quality.top10_all_target_gain;
        if !top10_gain.is_finite() {
            return Err(AutosuggestScorerManifestError::Invalid {
                field: "quality.top10_all_target_gain",
                reason: "must be finite".to_string(),
            });
        }

        Ok(AutosuggestScorerCompatibility {
            artifact: self.artifact.clone(),
            version: self.version,
            runtime_role: self.runtime_role.clone(),
            ngram_artifact_bytes: model_info.artifact_bytes,
            ngram_artifact_fingerprint: artifact_fingerprint,
            vocab_size: model_info.vocab_size,
            vocab_fingerprint: model_info.vocab_fingerprint,
            token_id_dtype: contract.token_id_dtype.clone(),
            onnx_input_dtype: contract.onnx_input_dtype.clone(),
            coreml_input_dtype: contract.coreml_input_dtype.clone(),
            scores_dtype: contract.scores_dtype.clone(),
            context_window: self.scorer.context_window,
            candidate_pool: self.scorer.pool_k,
            visible_candidates: contract.visible_candidates,
            locked_prefix: contract.locked_prefix,
            rank_penalty: contract.rank_penalty,
            top5_all_target_gain: top5_gain,
            top10_all_target_gain: top10_gain,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn validate_asset_paths(
        &self,
        root: impl AsRef<Path>,
    ) -> Result<AutosuggestScorerAssetReport, AutosuggestScorerManifestError> {
        let root = root.as_ref();
        let ngram =
            validate_file_asset(root, "ngram.path", &self.ngram.path, self.ngram.bytes, None)?;
        let onnx = validate_file_asset(
            root,
            "scorer.onnx.path",
            &self.scorer.onnx.path,
            self.scorer.onnx.bytes,
            Some(&self.scorer.onnx.sha256),
        )?;
        let quantized_onnx = validate_file_asset(
            root,
            "scorer.quantized_onnx.path",
            &self.scorer.quantized_onnx.path,
            self.scorer.quantized_onnx.bytes,
            Some(&self.scorer.quantized_onnx.sha256),
        )?;
        let coreml = validate_package_asset(
            root,
            "scorer.coreml.path",
            &self.scorer.coreml.path,
            self.scorer.coreml.bytes,
            Some(&self.scorer.coreml.sha256),
        )?;

        Ok(AutosuggestScorerAssetReport {
            root: root.display().to_string(),
            ngram,
            onnx,
            quantized_onnx,
            coreml,
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestScorerRuntimeContract {
    pub token_id_dtype: String,
    pub onnx_input_dtype: String,
    pub coreml_input_dtype: String,
    pub scores_dtype: String,
    pub batch_size: usize,
    pub context_ids_shape: [usize; 2],
    pub candidate_ids_shape: [usize; 2],
    pub scores_shape: [usize; 2],
    pub pad_id: u32,
    pub bos_id: u32,
    pub unk_id: u32,
    pub locked_prefix: usize,
    pub rank_penalty: f32,
    pub visible_candidates: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestScorerNgram {
    pub path: String,
    pub manifest: String,
    pub bytes: usize,
    pub artifact_fingerprint: String,
    pub vocab_size: usize,
    pub vocab_fingerprint: u32,
    pub max_candidates_per_prefix: usize,
    pub candidate_rows: usize,
    pub candidate_record_len: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestScorerModel {
    pub architecture: String,
    pub context_window: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub parameter_count: usize,
    pub pool_k: usize,
    pub onnx: AutosuggestScorerFile,
    pub quantized_onnx: AutosuggestScorerFile,
    pub coreml: AutosuggestScorerFile,
    pub coreml_target: String,
    pub coreml_precision: String,
    pub coreml_compute_unit: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestScorerFile {
    pub path: String,
    pub bytes: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestScorerQuality {
    pub heldout_targets: usize,
    pub eligible_targets: usize,
    pub static_pool: AutosuggestScorerQualityMetrics,
    pub selected_quantized_locked_first: AutosuggestScorerQualityMetrics,
    pub top5_all_target_gain: f64,
    pub top10_all_target_gain: f64,
    pub pool_recall_all_targets: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestScorerQualityMetrics {
    pub top1_all_targets: f64,
    pub top5_all_targets: f64,
    pub top10_all_targets: f64,
    pub mrr_all_targets: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AutosuggestScorerCompatibility {
    pub artifact: String,
    pub version: u32,
    pub runtime_role: String,
    pub ngram_artifact_bytes: usize,
    pub ngram_artifact_fingerprint: String,
    pub vocab_size: usize,
    pub vocab_fingerprint: u32,
    pub token_id_dtype: String,
    pub onnx_input_dtype: String,
    pub coreml_input_dtype: String,
    pub scores_dtype: String,
    pub context_window: usize,
    pub candidate_pool: usize,
    pub visible_candidates: usize,
    pub locked_prefix: usize,
    pub rank_penalty: f32,
    pub top5_all_target_gain: f64,
    pub top10_all_target_gain: f64,
}

/// Fixed-shape runtime bridge between the n-gram retriever and a platform scorer.
///
/// This type is intentionally model-runtime agnostic. Apple integrations can
/// call `coreml_inputs_for_context_into`, pass the returned buffers to Core ML,
/// then call `rerank_with_scores_into`. ONNX/native integrations can use the
/// u32 input path and widen token IDs to the model's input dtype at the boundary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AutosuggestScorerHandoff {
    pub compatibility: AutosuggestScorerCompatibility,
    pub context_window: usize,
    pub candidate_pool: usize,
    pub visible_candidates: usize,
    pub locked_prefix: usize,
    pub rank_penalty: f32,
}

impl AutosuggestScorerHandoff {
    pub fn from_manifest_for_lm<D: AsRef<[u8]>>(
        manifest: &AutosuggestScorerManifest,
        lm: &AutosuggestLm<D>,
    ) -> Result<Self, AutosuggestScorerManifestError> {
        let compatibility = manifest.validate_for_lm(lm)?;
        Ok(Self {
            context_window: manifest.runtime_contract.context_ids_shape[1],
            candidate_pool: manifest.runtime_contract.candidate_ids_shape[1],
            visible_candidates: manifest.runtime_contract.visible_candidates,
            locked_prefix: manifest.runtime_contract.locked_prefix,
            rank_penalty: manifest.runtime_contract.rank_penalty,
            compatibility,
        })
    }

    pub fn rerank_options(&self) -> AutosuggestRerankOptions {
        AutosuggestRerankOptions {
            max_candidates: self.visible_candidates,
            locked_prefix: self.locked_prefix,
            rank_penalty: self.rank_penalty,
        }
    }

    pub fn u32_inputs_for_context_into<D: AsRef<[u8]>>(
        &self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
        context_ids: &mut [u32],
        candidate_ids: &mut [u32],
        candidates: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestScorerHandoffError> {
        self.require_context_buffer("context_ids", context_ids.len())?;
        self.require_candidate_buffer("candidate_ids", candidate_ids.len())?;

        let scorer_context_token_count =
            lm.scorer_context_ids_for_context_into(context, context_ids)?;
        let metadata = lm.suggest_ids_for_context_into(
            context,
            AutosuggestOptions {
                max_candidates: self.candidate_pool,
            },
            candidates,
        )?;
        scorer_candidate_ids_for_candidates_into(candidates, candidate_ids);

        Ok(AutosuggestRerankInputMetadata {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            scorer_context_token_count,
            candidate_count: candidates.len(),
        })
    }

    pub fn coreml_inputs_for_context_into<D: AsRef<[u8]>>(
        &self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
        context_ids: &mut [i32],
        candidate_ids: &mut [i32],
        candidates: &mut Vec<AutosuggestCandidateId>,
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestScorerHandoffError> {
        self.require_context_buffer("context_ids", context_ids.len())?;
        self.require_candidate_buffer("candidate_ids", candidate_ids.len())?;

        let scorer_context_token_count =
            lm.scorer_context_i32s_for_context_into(context, context_ids)?;
        let metadata = lm.suggest_ids_for_context_into(
            context,
            AutosuggestOptions {
                max_candidates: self.candidate_pool,
            },
            candidates,
        )?;
        scorer_candidate_i32s_for_candidates_into(candidates, candidate_ids)?;

        Ok(AutosuggestRerankInputMetadata {
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            scorer_context_token_count,
            candidate_count: candidates.len(),
        })
    }

    pub fn rerank_with_scores_into(
        &self,
        candidates: &[AutosuggestCandidateId],
        model_scores: &[f32],
        output: &mut Vec<AutosuggestScoredCandidateId>,
    ) -> Result<(), AutosuggestScorerHandoffError> {
        if model_scores.len() != self.candidate_pool {
            output.clear();
            return Err(AutosuggestScorerHandoffError::InvalidBuffer {
                field: "model_scores",
                expected: self.candidate_pool,
                actual: model_scores.len(),
            });
        }
        rerank_candidate_ids_with_fixed_scores_into(
            candidates,
            model_scores,
            self.rerank_options(),
            output,
        )?;
        Ok(())
    }

    fn require_context_buffer(
        &self,
        field: &'static str,
        actual: usize,
    ) -> Result<(), AutosuggestScorerHandoffError> {
        self.require_buffer(field, self.context_window, actual)
    }

    fn require_candidate_buffer(
        &self,
        field: &'static str,
        actual: usize,
    ) -> Result<(), AutosuggestScorerHandoffError> {
        self.require_buffer(field, self.candidate_pool, actual)
    }

    fn require_buffer(
        &self,
        field: &'static str,
        expected: usize,
        actual: usize,
    ) -> Result<(), AutosuggestScorerHandoffError> {
        if actual != expected {
            return Err(AutosuggestScorerHandoffError::InvalidBuffer {
                field,
                expected,
                actual,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AutosuggestScorerHandoffError {
    Artifact(AutosuggestArtifactError),
    Rerank(AutosuggestRerankError),
    InvalidBuffer {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for AutosuggestScorerHandoffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => error.fmt(f),
            Self::Rerank(error) => error.fmt(f),
            Self::InvalidBuffer {
                field,
                expected,
                actual,
            } => write!(
                f,
                "autosuggest scorer handoff expected {field} length {expected}, got {actual}"
            ),
        }
    }
}

impl Error for AutosuggestScorerHandoffError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Artifact(error) => Some(error),
            Self::Rerank(error) => Some(error),
            Self::InvalidBuffer { .. } => None,
        }
    }
}

impl From<AutosuggestArtifactError> for AutosuggestScorerHandoffError {
    fn from(error: AutosuggestArtifactError) -> Self {
        Self::Artifact(error)
    }
}

impl From<AutosuggestRerankError> for AutosuggestScorerHandoffError {
    fn from(error: AutosuggestRerankError) -> Self {
        Self::Rerank(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AutosuggestMaterializedScoredCandidate<'a> {
    pub text: &'a str,
    pub token_id: u32,
    pub source: AutosuggestSource,
    pub count: u32,
    pub score: i32,
    pub original_rank: usize,
    pub model_score: f32,
    pub rerank_score: f32,
}

/// Reusable scorer handoff state for a keyboard/editor session.
///
/// The vectors are allocated once at construction using the fixed manifest
/// shape. Repeated `prepare_*` and `rerank_with_scores` calls then reuse the
/// same buffers, which keeps the per-keystroke path predictable for mobile
/// keyboard integrations.
#[derive(Debug, Clone)]
pub struct AutosuggestScorerSession {
    handoff: AutosuggestScorerHandoff,
    context_ids_u32: Vec<u32>,
    context_ids_i32: Vec<i32>,
    candidate_ids_u32: Vec<u32>,
    candidate_ids_i32: Vec<i32>,
    candidates: Vec<AutosuggestCandidateId>,
    ranked: Vec<AutosuggestScoredCandidateId>,
}

impl AutosuggestScorerSession {
    pub fn new(handoff: AutosuggestScorerHandoff) -> Self {
        let context_window = handoff.context_window;
        let candidate_pool = handoff.candidate_pool;
        Self {
            handoff,
            context_ids_u32: vec![AUTOSUGGEST_PAD_ID; context_window],
            context_ids_i32: vec![AUTOSUGGEST_PAD_ID as i32; context_window],
            candidate_ids_u32: vec![AUTOSUGGEST_PAD_ID; candidate_pool],
            candidate_ids_i32: vec![AUTOSUGGEST_PAD_ID as i32; candidate_pool],
            candidates: Vec::with_capacity(candidate_pool),
            ranked: Vec::with_capacity(candidate_pool),
        }
    }

    pub fn from_manifest_for_lm<D: AsRef<[u8]>>(
        manifest: &AutosuggestScorerManifest,
        lm: &AutosuggestLm<D>,
    ) -> Result<Self, AutosuggestScorerManifestError> {
        Ok(Self::new(AutosuggestScorerHandoff::from_manifest_for_lm(
            manifest, lm,
        )?))
    }

    pub fn handoff(&self) -> &AutosuggestScorerHandoff {
        &self.handoff
    }

    pub fn u32_context_ids(&self) -> &[u32] {
        &self.context_ids_u32
    }

    pub fn u32_candidate_ids(&self) -> &[u32] {
        &self.candidate_ids_u32
    }

    pub fn coreml_context_ids(&self) -> &[i32] {
        &self.context_ids_i32
    }

    pub fn coreml_candidate_ids(&self) -> &[i32] {
        &self.candidate_ids_i32
    }

    pub fn candidates(&self) -> &[AutosuggestCandidateId] {
        &self.candidates
    }

    pub fn ranked_candidates(&self) -> &[AutosuggestScoredCandidateId] {
        &self.ranked
    }

    pub fn prepare_u32_inputs<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestScorerHandoffError> {
        self.ranked.clear();
        self.handoff.u32_inputs_for_context_into(
            lm,
            context,
            &mut self.context_ids_u32,
            &mut self.candidate_ids_u32,
            &mut self.candidates,
        )
    }

    pub fn prepare_u32_inputs_for_autosuggest_session<'lm, D: AsRef<[u8]>>(
        &mut self,
        session: &mut AutosuggestSession<'lm, D>,
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestScorerHandoffError> {
        self.ranked.clear();
        let metadata = session.rerank_u32_input_with_options_into(
            AutosuggestOptions {
                max_candidates: self.handoff.candidate_pool,
            },
            &mut self.context_ids_u32,
            &mut self.candidate_ids_u32,
            &mut self.candidates,
        )?;
        Ok(metadata)
    }

    pub fn prepare_coreml_inputs<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestScorerHandoffError> {
        self.ranked.clear();
        self.handoff.coreml_inputs_for_context_into(
            lm,
            context,
            &mut self.context_ids_i32,
            &mut self.candidate_ids_i32,
            &mut self.candidates,
        )
    }

    pub fn prepare_coreml_inputs_for_autosuggest_session<'lm, D: AsRef<[u8]>>(
        &mut self,
        session: &mut AutosuggestSession<'lm, D>,
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestScorerHandoffError> {
        self.ranked.clear();
        let metadata = session.rerank_coreml_input_with_options_into(
            AutosuggestOptions {
                max_candidates: self.handoff.candidate_pool,
            },
            &mut self.context_ids_i32,
            &mut self.candidate_ids_i32,
            &mut self.candidates,
        )?;
        Ok(metadata)
    }

    pub fn rerank_with_scores(
        &mut self,
        model_scores: &[f32],
    ) -> Result<&[AutosuggestScoredCandidateId], AutosuggestScorerHandoffError> {
        self.handoff
            .rerank_with_scores_into(&self.candidates, model_scores, &mut self.ranked)?;
        Ok(&self.ranked)
    }

    pub fn materialized_ranked_candidates_into<'lm, D: AsRef<[u8]>>(
        &self,
        lm: &'lm AutosuggestLm<D>,
        output: &mut Vec<AutosuggestMaterializedScoredCandidate<'lm>>,
    ) -> Result<(), AutosuggestArtifactError> {
        output.clear();
        output.reserve(self.ranked.len());
        for candidate in &self.ranked {
            let materialized = lm.materialize_candidate(candidate.candidate_id())?;
            output.push(AutosuggestMaterializedScoredCandidate {
                text: materialized.text,
                token_id: materialized.token_id,
                source: materialized.source,
                count: materialized.count,
                score: materialized.score,
                original_rank: candidate.original_rank,
                model_score: candidate.model_score,
                rerank_score: candidate.rerank_score,
            });
        }
        Ok(())
    }

    pub fn estimated_heap_bytes(&self) -> usize {
        self.context_ids_u32.capacity() * mem::size_of::<u32>()
            + self.context_ids_i32.capacity() * mem::size_of::<i32>()
            + self.candidate_ids_u32.capacity() * mem::size_of::<u32>()
            + self.candidate_ids_i32.capacity() * mem::size_of::<i32>()
            + self.candidates.capacity() * mem::size_of::<AutosuggestCandidateId>()
            + self.ranked.capacity() * mem::size_of::<AutosuggestScoredCandidateId>()
    }

    pub fn heap_limit_bytes(&self) -> usize {
        let context_window = self.handoff.context_window;
        let candidate_pool = self.handoff.candidate_pool;
        context_window * mem::size_of::<u32>()
            + context_window * mem::size_of::<i32>()
            + candidate_pool * mem::size_of::<u32>()
            + candidate_pool * mem::size_of::<i32>()
            + candidate_pool * mem::size_of::<AutosuggestCandidateId>()
            + candidate_pool * mem::size_of::<AutosuggestScoredCandidateId>()
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AutosuggestScorerAssetReport {
    pub root: String,
    pub ngram: AutosuggestScorerAsset,
    pub onnx: AutosuggestScorerAsset,
    pub quantized_onnx: AutosuggestScorerAsset,
    pub coreml: AutosuggestScorerAsset,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AutosuggestScorerAsset {
    pub path: String,
    pub bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AutosuggestScorerManifestError {
    Mismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    Invalid {
        field: &'static str,
        reason: String,
    },
    Asset {
        field: &'static str,
        path: String,
        reason: String,
    },
}

impl fmt::Display for AutosuggestScorerManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "autosuggest scorer manifest mismatch for {field}: expected {expected}, got {actual}"
            ),
            Self::Invalid { field, reason } => {
                write!(f, "autosuggest scorer manifest has invalid {field}: {reason}")
            }
            Self::Asset {
                field,
                path,
                reason,
            } => write!(
                f,
                "autosuggest scorer manifest asset check failed for {field} at {path}: {reason}"
            ),
        }
    }
}

impl Error for AutosuggestScorerManifestError {}

fn require_shape(
    field: &'static str,
    actual: [usize; 2],
) -> Result<(), AutosuggestScorerManifestError> {
    if actual[0] == 0 || actual[1] == 0 {
        return Err(AutosuggestScorerManifestError::Invalid {
            field,
            reason: "must be a non-empty 2D fixed shape".to_string(),
        });
    }
    Ok(())
}

fn require_equal_str(
    field: &'static str,
    expected: &str,
    actual: &str,
) -> Result<(), AutosuggestScorerManifestError> {
    if expected != actual {
        return Err(AutosuggestScorerManifestError::Mismatch {
            field,
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

fn require_equal_usize(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), AutosuggestScorerManifestError> {
    if expected != actual {
        return Err(AutosuggestScorerManifestError::Mismatch {
            field,
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

fn require_equal_u32(
    field: &'static str,
    expected: u32,
    actual: u32,
) -> Result<(), AutosuggestScorerManifestError> {
    if expected != actual {
        return Err(AutosuggestScorerManifestError::Mismatch {
            field,
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

fn require_equal_f32(
    field: &'static str,
    expected: f32,
    actual: f32,
) -> Result<(), AutosuggestScorerManifestError> {
    if !actual.is_finite() || (expected - actual).abs() > f32::EPSILON {
        return Err(AutosuggestScorerManifestError::Mismatch {
            field,
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_file_asset(
    root: &Path,
    field: &'static str,
    manifest_path: &str,
    expected_bytes: usize,
    expected_sha256: Option<&str>,
) -> Result<AutosuggestScorerAsset, AutosuggestScorerManifestError> {
    let path = resolve_asset_path(root, manifest_path);
    let metadata = fs::metadata(&path).map_err(|error| AutosuggestScorerManifestError::Asset {
        field,
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    if !metadata.is_file() {
        return Err(AutosuggestScorerManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: "expected a file".to_string(),
        });
    }
    let actual_bytes =
        usize::try_from(metadata.len()).map_err(|_| AutosuggestScorerManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: "file is too large for this platform".to_string(),
        })?;
    if actual_bytes != expected_bytes {
        return Err(AutosuggestScorerManifestError::Mismatch {
            field,
            expected: expected_bytes.to_string(),
            actual: actual_bytes.to_string(),
        });
    }
    let sha256 = if let Some(expected) = expected_sha256 {
        validate_sha256_format(field, expected)?;
        let actual = sha256_file_hex(&path, field)?;
        if actual != expected {
            return Err(AutosuggestScorerManifestError::Mismatch {
                field,
                expected: expected.to_string(),
                actual,
            });
        }
        Some(actual)
    } else {
        None
    };
    Ok(AutosuggestScorerAsset {
        path: path.display().to_string(),
        bytes: actual_bytes,
        sha256,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_package_asset(
    root: &Path,
    field: &'static str,
    manifest_path: &str,
    expected_bytes: usize,
    expected_sha256: Option<&str>,
) -> Result<AutosuggestScorerAsset, AutosuggestScorerManifestError> {
    let path = resolve_asset_path(root, manifest_path);
    let metadata = fs::metadata(&path).map_err(|error| AutosuggestScorerManifestError::Asset {
        field,
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    if !metadata.is_dir() {
        return Err(AutosuggestScorerManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: "expected a directory package".to_string(),
        });
    }
    let actual_bytes = package_size(&path)?;
    if actual_bytes != expected_bytes {
        return Err(AutosuggestScorerManifestError::Mismatch {
            field,
            expected: expected_bytes.to_string(),
            actual: actual_bytes.to_string(),
        });
    }
    let sha256 = if let Some(expected) = expected_sha256 {
        validate_sha256_format(field, expected)?;
        let actual = sha256_package_hex(&path, field)?;
        if actual != expected {
            return Err(AutosuggestScorerManifestError::Mismatch {
                field,
                expected: expected.to_string(),
                actual,
            });
        }
        Some(actual)
    } else {
        None
    };
    Ok(AutosuggestScorerAsset {
        path: path.display().to_string(),
        bytes: actual_bytes,
        sha256,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_asset_path(root: &Path, manifest_path: &str) -> PathBuf {
    let path = Path::new(manifest_path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn package_size(path: &Path) -> Result<usize, AutosuggestScorerManifestError> {
    let mut total = 0usize;
    add_package_size(path, path, &mut total)?;
    Ok(total)
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_sha256_format(
    field: &'static str,
    value: &str,
) -> Result<(), AutosuggestScorerManifestError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(AutosuggestScorerManifestError::Invalid {
            field,
            reason: "sha256 must be 64 hex characters".to_string(),
        });
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn sha256_file_hex(
    path: &Path,
    field: &'static str,
) -> Result<String, AutosuggestScorerManifestError> {
    let mut file = fs::File::open(path).map_err(|error| AutosuggestScorerManifestError::Asset {
        field,
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read =
            file.read(&mut buffer)
                .map_err(|error| AutosuggestScorerManifestError::Asset {
                    field,
                    path: path.display().to_string(),
                    reason: error.to_string(),
                })?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(not(target_arch = "wasm32"))]
fn sha256_package_hex(
    path: &Path,
    field: &'static str,
) -> Result<String, AutosuggestScorerManifestError> {
    let mut files = Vec::new();
    collect_package_files(path, path, &mut files, field)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));

    let mut digest = Sha256::new();
    for (relative, full_path) in files {
        digest.update(relative.as_bytes());
        digest.update(b"\0");
        update_file_digest(&full_path, field, &mut digest)?;
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_package_files(
    root: &Path,
    path: &Path,
    output: &mut Vec<(String, PathBuf)>,
    field: &'static str,
) -> Result<(), AutosuggestScorerManifestError> {
    for entry in fs::read_dir(path).map_err(|error| AutosuggestScorerManifestError::Asset {
        field,
        path: path.display().to_string(),
        reason: error.to_string(),
    })? {
        let entry = entry.map_err(|error| AutosuggestScorerManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        let path = entry.path();
        let metadata = entry
            .metadata()
            .map_err(|error| AutosuggestScorerManifestError::Asset {
                field,
                path: path.display().to_string(),
                reason: error.to_string(),
            })?;
        if metadata.is_dir() {
            collect_package_files(root, &path, output, field)?;
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|error| AutosuggestScorerManifestError::Asset {
                    field,
                    path: path.display().to_string(),
                    reason: error.to_string(),
                })?
                .to_string_lossy()
                .replace('\\', "/");
            output.push((relative, path));
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn update_file_digest(
    path: &Path,
    field: &'static str,
    digest: &mut Sha256,
) -> Result<(), AutosuggestScorerManifestError> {
    let mut file = fs::File::open(path).map_err(|error| AutosuggestScorerManifestError::Asset {
        field,
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read =
            file.read(&mut buffer)
                .map_err(|error| AutosuggestScorerManifestError::Asset {
                    field,
                    path: path.display().to_string(),
                    reason: error.to_string(),
                })?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn add_package_size(
    root: &Path,
    path: &Path,
    total: &mut usize,
) -> Result<(), AutosuggestScorerManifestError> {
    for entry in fs::read_dir(path).map_err(|error| AutosuggestScorerManifestError::Asset {
        field: "scorer.coreml.path",
        path: path.display().to_string(),
        reason: error.to_string(),
    })? {
        let entry = entry.map_err(|error| AutosuggestScorerManifestError::Asset {
            field: "scorer.coreml.path",
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        let path = entry.path();
        let metadata = entry
            .metadata()
            .map_err(|error| AutosuggestScorerManifestError::Asset {
                field: "scorer.coreml.path",
                path: path.display().to_string(),
                reason: error.to_string(),
            })?;
        if metadata.is_dir() {
            add_package_size(root, &path, total)?;
        } else if metadata.is_file() {
            let len = usize::try_from(metadata.len()).map_err(|_| {
                AutosuggestScorerManifestError::Asset {
                    field: "scorer.coreml.path",
                    path: path.display().to_string(),
                    reason: "file is too large for this platform".to_string(),
                }
            })?;
            *total =
                total
                    .checked_add(len)
                    .ok_or_else(|| AutosuggestScorerManifestError::Asset {
                        field: "scorer.coreml.path",
                        path: root.display().to_string(),
                        reason: "package byte size overflowed usize".to_string(),
                    })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autosuggest::artifact::test_support::{build_fixture, Row};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_lm() -> AutosuggestLm<Vec<u8>> {
        let tokens = ["<pad>", "<bos>", "<unk>", "আমি", "আজ", "সকালে", "স্কুলে", "যাব"];
        AutosuggestLm::from_bytes(build_fixture(
            &tokens,
            &[(5, 100, 100), (7, 90, 90)],
            &[
                Row {
                    context: vec![4],
                    candidates: vec![(5, 40, 40), (7, 12, 12)],
                },
                Row {
                    context: vec![3, 4],
                    candidates: vec![(7, 9, 9), (5, 7, 7)],
                },
            ],
        ))
        .expect("fixture should parse")
    }

    fn fixture_manifest() -> AutosuggestScorerManifest {
        let lm = fixture_lm();
        AutosuggestScorerManifest {
            artifact: AUTOSUGGEST_SCORER_PACKAGE_KIND.to_string(),
            version: AUTOSUGGEST_SCORER_MANIFEST_VERSION,
            runtime_role: AUTOSUGGEST_SCORER_RUNTIME_ROLE.to_string(),
            runtime_contract: AutosuggestScorerRuntimeContract {
                token_id_dtype: AUTOSUGGEST_SCORER_TOKEN_ID_DTYPE.to_string(),
                onnx_input_dtype: AUTOSUGGEST_SCORER_ONNX_INPUT_DTYPE.to_string(),
                coreml_input_dtype: AUTOSUGGEST_SCORER_COREML_INPUT_DTYPE.to_string(),
                scores_dtype: AUTOSUGGEST_SCORER_SCORE_DTYPE.to_string(),
                batch_size: 1,
                context_ids_shape: [1, MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS],
                candidate_ids_shape: [1, 8],
                scores_shape: [1, 8],
                pad_id: AUTOSUGGEST_PAD_ID,
                bos_id: AUTOSUGGEST_BOS_ID,
                unk_id: AUTOSUGGEST_UNK_ID,
                locked_prefix: DEFAULT_AUTOSUGGEST_RERANK_LOCKED_PREFIX,
                rank_penalty: DEFAULT_AUTOSUGGEST_RERANK_RANK_PENALTY,
                visible_candidates: DEFAULT_AUTOSUGGEST_CANDIDATES,
            },
            ngram: AutosuggestScorerNgram {
                path: "fixture.bin".to_string(),
                manifest: "fixture.manifest.json".to_string(),
                bytes: lm.model_info().artifact_bytes,
                artifact_fingerprint: format!("{:016x}", lm.artifact_fingerprint()),
                vocab_size: lm.model_info().vocab_size,
                vocab_fingerprint: lm.model_info().vocab_fingerprint,
                max_candidates_per_prefix: 8,
                candidate_rows: 4,
                candidate_record_len: lm.model_info().candidate_record_len,
            },
            scorer: AutosuggestScorerModel {
                architecture: "gru".to_string(),
                context_window: MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS,
                embedding_dim: 256,
                hidden_dim: 256,
                parameter_count: 1,
                pool_k: 8,
                onnx: scorer_file("fixture.onnx"),
                quantized_onnx: scorer_file("fixture.int8.onnx"),
                coreml: coreml_package("fixture.mlpackage"),
                coreml_target: "ios17".to_string(),
                coreml_precision: "float16".to_string(),
                coreml_compute_unit: "all".to_string(),
            },
            quality: AutosuggestScorerQuality {
                heldout_targets: 100,
                eligible_targets: 90,
                static_pool: quality_metrics(0.1, 0.2, 0.3),
                selected_quantized_locked_first: quality_metrics(0.1, 0.25, 0.35),
                top5_all_target_gain: 0.05,
                top10_all_target_gain: 0.05,
                pool_recall_all_targets: 0.5,
            },
        }
    }

    fn scorer_file(path: &str) -> AutosuggestScorerFile {
        AutosuggestScorerFile {
            path: path.to_string(),
            bytes: 1,
            sha256: sha256_bytes_hex(&[0]),
        }
    }

    fn coreml_package(path: &str) -> AutosuggestScorerFile {
        AutosuggestScorerFile {
            path: path.to_string(),
            bytes: 1,
            sha256: sha256_tree_single_file_hex("Data/com.apple.CoreML/weights/weight.bin", &[0]),
        }
    }

    fn quality_metrics(
        top1_all_targets: f64,
        top5_all_targets: f64,
        top10_all_targets: f64,
    ) -> AutosuggestScorerQualityMetrics {
        AutosuggestScorerQualityMetrics {
            top1_all_targets,
            top5_all_targets,
            top10_all_targets,
            mrr_all_targets: top1_all_targets,
        }
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "obadh-scorer-{label}-{}-{nanos}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_sized_file(path: &Path, bytes: usize) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, vec![0_u8; bytes]).unwrap();
    }

    fn write_manifest_assets(root: &Path, manifest: &AutosuggestScorerManifest) {
        write_sized_file(&root.join(&manifest.ngram.path), manifest.ngram.bytes);
        write_sized_file(
            &root.join(&manifest.scorer.onnx.path),
            manifest.scorer.onnx.bytes,
        );
        write_sized_file(
            &root.join(&manifest.scorer.quantized_onnx.path),
            manifest.scorer.quantized_onnx.bytes,
        );
        write_sized_file(
            &root
                .join(&manifest.scorer.coreml.path)
                .join("Data/com.apple.CoreML/weights/weight.bin"),
            manifest.scorer.coreml.bytes,
        );
    }

    fn sha256_bytes_hex(bytes: &[u8]) -> String {
        let mut digest = Sha256::new();
        digest.update(bytes);
        format!("{:x}", digest.finalize())
    }

    fn sha256_tree_single_file_hex(relative_path: &str, bytes: &[u8]) -> String {
        let mut digest = Sha256::new();
        digest.update(relative_path.as_bytes());
        digest.update(b"\0");
        digest.update(bytes);
        format!("{:x}", digest.finalize())
    }

    #[test]
    fn scorer_manifest_validates_against_matching_lm() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();

        let compatibility = manifest.validate_for_lm(&lm).unwrap();

        assert_eq!(
            compatibility.context_window,
            MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS
        );
        assert_eq!(compatibility.candidate_pool, 8);
        assert_eq!(
            compatibility.visible_candidates,
            DEFAULT_AUTOSUGGEST_CANDIDATES
        );
        assert_eq!(compatibility.top5_all_target_gain, 0.05);
    }

    #[test]
    fn scorer_handoff_prepares_coreml_inputs_and_applies_locked_rerank() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let handoff = AutosuggestScorerHandoff::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut context = AutosuggestContext::new();
        lm.push_context_text(&mut context, "আমি আজ").unwrap();
        let mut context_ids = vec![99_i32; handoff.context_window];
        let mut candidate_ids = vec![99_i32; handoff.candidate_pool];
        let mut candidates = Vec::new();

        let metadata = handoff
            .coreml_inputs_for_context_into(
                &lm,
                context,
                &mut context_ids,
                &mut candidate_ids,
                &mut candidates,
            )
            .unwrap();

        assert_eq!(metadata.scorer_context_token_count, 2);
        assert_eq!(&context_ids[handoff.context_window - 2..], &[3, 4]);
        assert_eq!(&candidate_ids[..2], &[7, 5]);
        assert!(candidate_ids[2..]
            .iter()
            .all(|id| *id == AUTOSUGGEST_PAD_ID as i32));
        assert_eq!(metadata.candidate_count, candidates.len());

        let mut model_scores = vec![0.0_f32; handoff.candidate_pool];
        model_scores[0] = -10.0;
        model_scores[1] = 10.0;
        let mut ranked = Vec::new();
        handoff
            .rerank_with_scores_into(&candidates, &model_scores, &mut ranked)
            .unwrap();

        assert_eq!(
            ranked
                .iter()
                .map(|candidate| candidate.candidate.token_id)
                .collect::<Vec<_>>(),
            vec![7, 5]
        );
        assert_eq!(ranked[0].original_rank, 0);
    }

    #[test]
    fn scorer_handoff_rejects_wrong_fixed_buffer_shapes() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let handoff = AutosuggestScorerHandoff::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut context_ids = vec![0_u32; handoff.context_window - 1];
        let mut candidate_ids = vec![0_u32; handoff.candidate_pool];
        let mut candidates = Vec::new();

        let error = handoff
            .u32_inputs_for_context_into(
                &lm,
                AutosuggestContext::new(),
                &mut context_ids,
                &mut candidate_ids,
                &mut candidates,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            AutosuggestScorerHandoffError::InvalidBuffer {
                field: "context_ids",
                ..
            }
        ));
    }

    #[test]
    fn scorer_handoff_rejects_wrong_score_shape() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let handoff = AutosuggestScorerHandoff::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut ranked = Vec::new();

        let error = handoff
            .rerank_with_scores_into(&[], &[0.0], &mut ranked)
            .unwrap_err();

        assert!(matches!(
            error,
            AutosuggestScorerHandoffError::InvalidBuffer {
                field: "model_scores",
                ..
            }
        ));
    }

    #[test]
    fn scorer_session_reuses_fixed_buffers_across_coreml_calls() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let mut session = AutosuggestScorerSession::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut context = AutosuggestContext::new();
        lm.push_context_text(&mut context, "আমি আজ").unwrap();

        let context_ptr = session.coreml_context_ids().as_ptr();
        let candidate_id_ptr = session.coreml_candidate_ids().as_ptr();
        let heap_limit = session.heap_limit_bytes();

        session.prepare_coreml_inputs(&lm, context).unwrap();
        let candidate_ptr = session.candidates().as_ptr();
        let mut model_scores = vec![0.0_f32; session.handoff().candidate_pool];
        model_scores[1] = 10.0;
        let ranked_ptr = session.rerank_with_scores(&model_scores).unwrap().as_ptr();

        session.prepare_coreml_inputs(&lm, context).unwrap();
        session.rerank_with_scores(&model_scores).unwrap();

        assert_eq!(session.coreml_context_ids().as_ptr(), context_ptr);
        assert_eq!(session.coreml_candidate_ids().as_ptr(), candidate_id_ptr);
        assert_eq!(session.candidates().as_ptr(), candidate_ptr);
        assert_eq!(session.ranked_candidates().as_ptr(), ranked_ptr);
        assert_eq!(session.heap_limit_bytes(), heap_limit);
        assert!(session.estimated_heap_bytes() <= session.heap_limit_bytes());
    }

    #[test]
    fn scorer_session_u32_and_coreml_inputs_share_candidate_contract() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let mut session = AutosuggestScorerSession::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut context = AutosuggestContext::new();
        lm.push_context_text(&mut context, "আমি আজ").unwrap();

        session.prepare_u32_inputs(&lm, context).unwrap();
        let u32_candidates = session.u32_candidate_ids().to_vec();
        let u32_context = session.u32_context_ids().to_vec();

        session.prepare_coreml_inputs(&lm, context).unwrap();

        assert_eq!(
            session
                .coreml_candidate_ids()
                .iter()
                .map(|id| *id as u32)
                .collect::<Vec<_>>(),
            u32_candidates
        );
        assert_eq!(
            session
                .coreml_context_ids()
                .iter()
                .map(|id| *id as u32)
                .collect::<Vec<_>>(),
            u32_context
        );
    }

    #[test]
    fn scorer_session_materializes_ranked_candidates_without_reallocating_output() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let mut session = AutosuggestScorerSession::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut context = AutosuggestContext::new();
        lm.push_context_text(&mut context, "আমি আজ").unwrap();

        session.prepare_u32_inputs(&lm, context).unwrap();
        let mut model_scores = vec![0.0_f32; session.handoff().candidate_pool];
        model_scores[1] = 10.0;
        session.rerank_with_scores(&model_scores).unwrap();

        let mut output = Vec::<AutosuggestMaterializedScoredCandidate<'_>>::with_capacity(
            session.handoff().visible_candidates,
        );
        let output_ptr = output.as_ptr();
        session
            .materialized_ranked_candidates_into(&lm, &mut output)
            .unwrap();

        assert_eq!(output.as_ptr(), output_ptr);
        assert_eq!(
            output
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["যাব", "সকালে"]
        );
        assert_eq!(output[0].token_id, 7);
        assert_eq!(output[0].source, AutosuggestSource::Trigram);
        assert_eq!(output[0].original_rank, 0);
        assert_eq!(output[1].token_id, 5);
        assert_eq!(output[1].model_score, 10.0);
        assert_eq!(output[1].original_rank, 1);

        session
            .materialized_ranked_candidates_into(&lm, &mut output)
            .unwrap();

        assert_eq!(output.as_ptr(), output_ptr);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn scorer_session_prepares_personal_aware_coreml_inputs() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let mut scorer_session =
            AutosuggestScorerSession::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut autosuggest_session = AutosuggestSession::with_personal_config(
            &lm,
            crate::autosuggest::PersonalAutosuggestConfig {
                max_entries: 32,
                min_count: 2,
            },
            AutosuggestOptions { max_candidates: 2 },
        );

        for _ in 0..2 {
            autosuggest_session.clear_context();
            assert!(autosuggest_session.commit_token("আমি").unwrap());
            assert!(autosuggest_session.commit_token("আজ").unwrap());
            assert!(autosuggest_session.commit_token("স্কুলে").unwrap());
        }

        autosuggest_session.clear_context();
        assert!(autosuggest_session.commit_token("আমি").unwrap());
        assert!(autosuggest_session.commit_token("আজ").unwrap());
        autosuggest_session.suggest_ids().unwrap();
        let visible_candidate_ids = autosuggest_session.candidate_ids().to_vec();
        let metadata = scorer_session
            .prepare_coreml_inputs_for_autosuggest_session(&mut autosuggest_session)
            .unwrap();

        assert_eq!(autosuggest_session.options().max_candidates, 2);
        assert_eq!(metadata.scorer_context_token_count, 2);
        assert_eq!(
            &scorer_session.coreml_context_ids()[scorer_session.handoff().context_window - 2..],
            &[3, 4]
        );
        assert_eq!(autosuggest_session.candidate_ids(), visible_candidate_ids);
        assert_eq!(metadata.candidate_count, scorer_session.candidates().len());
        assert!(metadata.candidate_count > autosuggest_session.options().max_candidates);
        assert_eq!(scorer_session.coreml_candidate_ids()[0], 6);
        assert_eq!(scorer_session.candidates()[0].token_id, 6);
        assert_eq!(
            scorer_session.candidates()[0].source,
            AutosuggestSource::Personal
        );
        assert_eq!(
            scorer_session
                .candidates()
                .iter()
                .map(|candidate| lm.materialize_candidate(*candidate).unwrap().text)
                .take(3)
                .collect::<Vec<_>>(),
            vec!["স্কুলে", "যাব", "সকালে"]
        );

        let mut model_scores = vec![0.0_f32; scorer_session.handoff().candidate_pool];
        model_scores[1] = 10.0;
        scorer_session.rerank_with_scores(&model_scores).unwrap();
        let mut output = Vec::with_capacity(scorer_session.handoff().visible_candidates);
        scorer_session
            .materialized_ranked_candidates_into(&lm, &mut output)
            .unwrap();

        assert_eq!(output[0].text, "স্কুলে");
        assert_eq!(output[0].source, AutosuggestSource::Personal);
        assert_eq!(output[0].original_rank, 0);
    }

    #[test]
    fn scorer_manifest_rejects_wrong_ngram_fingerprint() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        manifest.ngram.artifact_fingerprint = "deadbeef".to_string();

        let error = manifest.validate_for_lm(&lm).unwrap_err();

        assert!(matches!(
            error,
            AutosuggestScorerManifestError::Mismatch {
                field: "ngram.artifact_fingerprint",
                ..
            }
        ));
    }

    #[test]
    fn scorer_manifest_rejects_unbounded_context_shape() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        manifest.runtime_contract.context_ids_shape = [1, 32];
        manifest.scorer.context_window = 32;

        let error = manifest.validate_for_lm(&lm).unwrap_err();

        assert!(matches!(
            error,
            AutosuggestScorerManifestError::Invalid {
                field: "scorer.context_window",
                ..
            }
        ));
    }

    #[test]
    fn scorer_manifest_rejects_non_improving_quality_gate() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        manifest.quality.top5_all_target_gain = 0.0;

        let error = manifest.validate_for_lm(&lm).unwrap_err();

        assert!(matches!(
            error,
            AutosuggestScorerManifestError::Invalid {
                field: "quality.top5_all_target_gain",
                ..
            }
        ));
    }

    #[test]
    fn scorer_manifest_validates_packaged_asset_sizes_and_hashes() {
        let manifest = fixture_manifest();
        let root = temp_root("asset-ok");
        write_manifest_assets(&root, &manifest);

        let report = manifest.validate_asset_paths(&root).unwrap();

        assert_eq!(report.ngram.bytes, manifest.ngram.bytes);
        assert_eq!(report.onnx.bytes, manifest.scorer.onnx.bytes);
        assert_eq!(
            report.quantized_onnx.bytes,
            manifest.scorer.quantized_onnx.bytes
        );
        assert_eq!(report.coreml.bytes, manifest.scorer.coreml.bytes);
        assert_eq!(
            report.onnx.sha256.as_deref(),
            Some(manifest.scorer.onnx.sha256.as_str())
        );
        assert_eq!(
            report.quantized_onnx.sha256.as_deref(),
            Some(manifest.scorer.quantized_onnx.sha256.as_str())
        );
        assert_eq!(
            report.coreml.sha256.as_deref(),
            Some(manifest.scorer.coreml.sha256.as_str())
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scorer_manifest_rejects_wrong_asset_size() {
        let manifest = fixture_manifest();
        let root = temp_root("asset-size");
        write_manifest_assets(&root, &manifest);
        write_sized_file(&root.join(&manifest.scorer.onnx.path), 2);

        let error = manifest.validate_asset_paths(&root).unwrap_err();

        assert!(matches!(
            error,
            AutosuggestScorerManifestError::Mismatch {
                field: "scorer.onnx.path",
                ..
            }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scorer_manifest_rejects_wrong_asset_hash() {
        let mut manifest = fixture_manifest();
        let root = temp_root("asset-hash");
        write_manifest_assets(&root, &manifest);
        manifest.scorer.quantized_onnx.sha256 = "00".repeat(32);

        let error = manifest.validate_asset_paths(&root).unwrap_err();

        assert!(matches!(
            error,
            AutosuggestScorerManifestError::Mismatch {
                field: "scorer.quantized_onnx.path",
                ..
            }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scorer_manifest_rejects_malformed_asset_hash() {
        let mut manifest = fixture_manifest();
        let root = temp_root("asset-hash-format");
        write_manifest_assets(&root, &manifest);
        manifest.scorer.onnx.sha256 = "not-a-sha".to_string();

        let error = manifest.validate_asset_paths(&root).unwrap_err();

        assert!(matches!(
            error,
            AutosuggestScorerManifestError::Invalid {
                field: "scorer.onnx.path",
                ..
            }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scorer_manifest_rejects_missing_coreml_package() {
        let manifest = fixture_manifest();
        let root = temp_root("asset-missing");
        write_sized_file(&root.join(&manifest.ngram.path), manifest.ngram.bytes);
        write_sized_file(
            &root.join(&manifest.scorer.onnx.path),
            manifest.scorer.onnx.bytes,
        );
        write_sized_file(
            &root.join(&manifest.scorer.quantized_onnx.path),
            manifest.scorer.quantized_onnx.bytes,
        );

        let error = manifest.validate_asset_paths(&root).unwrap_err();

        assert!(matches!(
            error,
            AutosuggestScorerManifestError::Asset {
                field: "scorer.coreml.path",
                ..
            }
        ));
        fs::remove_dir_all(root).unwrap();
    }
}
