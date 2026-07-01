use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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
use super::open_vocab::{
    accept_open_vocab_texts_into, merge_static_generated_and_open_vocab_candidates_into,
    AutosuggestOpenVocabError, AutosuggestOpenVocabPolicy, AutosuggestUnifiedCandidate,
    AutosuggestValidatedTextCandidate, DEFAULT_AUTOSUGGEST_OPEN_VOCAB_TEXT_HEAP_BYTES,
};

pub const AUTOSUGGEST_GENERATOR_PACKAGE_KIND: &str = "obadh-autosuggest-generator-package";
pub const AUTOSUGGEST_GENERATOR_RUNTIME_ROLE: &str = "next_word_candidate_generate";
pub const AUTOSUGGEST_GENERATOR_MANIFEST_VERSION: u32 = 1;
pub const AUTOSUGGEST_GENERATOR_TOKEN_ID_DTYPE: &str = "uint32";
pub const AUTOSUGGEST_GENERATOR_ONNX_INPUT_DTYPE: &str = "int64";
pub const AUTOSUGGEST_GENERATOR_COREML_INPUT_DTYPE: &str = "int32";
pub const AUTOSUGGEST_GENERATOR_SCORE_DTYPE: &str = "float32";
pub const DEFAULT_AUTOSUGGEST_GENERATOR_LOCKED_STATIC_PREFIX: usize = 1;
pub const MAX_AUTOSUGGEST_GENERATOR_PARAMETERS: usize = 10_000_000;
pub const MAX_AUTOSUGGEST_GENERATOR_TOP_K_OUTPUT: usize = 128;
pub const MAX_AUTOSUGGEST_GENERATOR_CANDIDATE_POOL: usize = 64;
pub const MAX_AUTOSUGGEST_GENERATOR_SESSION_HEAP_BYTES: usize = 64 * 1024;
pub const MAX_AUTOSUGGEST_GENERATOR_ONNX_BYTES: usize = 70 * 1024 * 1024;
pub const MAX_AUTOSUGGEST_GENERATOR_QUANTIZED_ONNX_BYTES: usize = 20 * 1024 * 1024;
pub const MAX_AUTOSUGGEST_GENERATOR_COREML_BYTES: usize = 20 * 1024 * 1024;
pub const MAX_AUTOSUGGEST_GENERATOR_GRAPH_US_PER_ITEM: f64 = 1_000.0;
pub const MIN_AUTOSUGGEST_GENERATOR_ELIGIBLE_TARGETS: usize = 10_000;
pub const MIN_AUTOSUGGEST_GENERATOR_SOURCE_ELIGIBLE_TARGETS: usize = 1_000;
const AUTOSUGGEST_GENERATOR_LOCKED_STATIC_BONUS: f32 = 1.0e6;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestGeneratorManifest {
    pub artifact: String,
    pub version: u32,
    pub runtime_role: String,
    pub runtime_contract: AutosuggestGeneratorRuntimeContract,
    pub ngram: AutosuggestGeneratorNgram,
    pub generator: AutosuggestGeneratorModel,
    pub quality: AutosuggestGeneratorQuality,
    pub benchmark: AutosuggestGeneratorBenchmark,
}

impl AutosuggestGeneratorManifest {
    pub fn from_json_str(input: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(input)
    }

    pub fn validate_for_lm<D: AsRef<[u8]>>(
        &self,
        lm: &AutosuggestLm<D>,
    ) -> Result<AutosuggestGeneratorCompatibility, AutosuggestGeneratorManifestError> {
        require_equal_str(
            "artifact",
            AUTOSUGGEST_GENERATOR_PACKAGE_KIND,
            &self.artifact,
        )?;
        require_equal_u32(
            "version",
            AUTOSUGGEST_GENERATOR_MANIFEST_VERSION,
            self.version,
        )?;
        require_equal_str(
            "runtime_role",
            AUTOSUGGEST_GENERATOR_RUNTIME_ROLE,
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
            AUTOSUGGEST_GENERATOR_TOKEN_ID_DTYPE,
            &contract.token_id_dtype,
        )?;
        require_equal_str(
            "runtime_contract.onnx_input_dtype",
            AUTOSUGGEST_GENERATOR_ONNX_INPUT_DTYPE,
            &contract.onnx_input_dtype,
        )?;
        require_equal_str(
            "runtime_contract.coreml_input_dtype",
            AUTOSUGGEST_GENERATOR_COREML_INPUT_DTYPE,
            &contract.coreml_input_dtype,
        )?;
        require_equal_str(
            "runtime_contract.scores_dtype",
            AUTOSUGGEST_GENERATOR_SCORE_DTYPE,
            &contract.scores_dtype,
        )?;
        require_shape(
            "runtime_contract.context_ids_shape",
            contract.context_ids_shape,
        )?;
        require_shape("runtime_contract.token_ids_shape", contract.token_ids_shape)?;
        require_shape("runtime_contract.scores_shape", contract.scores_shape)?;
        if contract.candidate_ids_shape.is_some() != contract.candidate_scores_shape.is_some() {
            return Err(AutosuggestGeneratorManifestError::Invalid {
                field: "runtime_contract.candidate_scores_shape",
                reason: "candidate input and score shapes must be declared together".to_string(),
            });
        }
        if let Some(candidate_ids_shape) = contract.candidate_ids_shape {
            require_shape("runtime_contract.candidate_ids_shape", candidate_ids_shape)?;
            require_equal_usize(
                "runtime_contract.candidate_ids_shape[0]",
                contract.batch_size,
                candidate_ids_shape[0],
            )?;
        }
        if let Some(candidate_scores_shape) = contract.candidate_scores_shape {
            require_shape(
                "runtime_contract.candidate_scores_shape",
                candidate_scores_shape,
            )?;
            require_equal_usize(
                "runtime_contract.candidate_scores_shape[0]",
                contract.batch_size,
                candidate_scores_shape[0],
            )?;
        }
        if let (Some(candidate_ids_shape), Some(candidate_scores_shape)) = (
            contract.candidate_ids_shape,
            contract.candidate_scores_shape,
        ) {
            require_equal_usize(
                "runtime_contract.candidate_scores_shape[1]",
                candidate_ids_shape[1],
                candidate_scores_shape[1],
            )?;
        }
        if let Some(policy) = contract.scored_union_policy {
            policy.validate("runtime_contract.scored_union_policy")?;
        }
        require_equal_usize("runtime_contract.batch_size", 1, contract.batch_size)?;
        require_equal_usize(
            "runtime_contract.context_ids_shape[0]",
            contract.batch_size,
            contract.context_ids_shape[0],
        )?;
        require_equal_usize(
            "runtime_contract.token_ids_shape[0]",
            contract.batch_size,
            contract.token_ids_shape[0],
        )?;
        require_equal_usize(
            "runtime_contract.scores_shape[0]",
            contract.batch_size,
            contract.scores_shape[0],
        )?;
        require_equal_usize(
            "runtime_contract.context_ids_shape[1]",
            self.generator.context_window,
            contract.context_ids_shape[1],
        )?;
        require_equal_usize(
            "runtime_contract.token_ids_shape[1]",
            self.generator.top_k_output,
            contract.token_ids_shape[1],
        )?;
        require_equal_usize(
            "runtime_contract.scores_shape[1]",
            self.generator.top_k_output,
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
            "runtime_contract.visible_candidates",
            DEFAULT_AUTOSUGGEST_CANDIDATES,
            contract.visible_candidates,
        )?;

        if self.generator.context_window > MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS {
            return Err(AutosuggestGeneratorManifestError::Invalid {
                field: "generator.context_window",
                reason: format!(
                    "exceeds runtime context buffer {}",
                    MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS
                ),
            });
        }
        if self.generator.parameter_count > MAX_AUTOSUGGEST_GENERATOR_PARAMETERS {
            return Err(AutosuggestGeneratorManifestError::Invalid {
                field: "generator.parameter_count",
                reason: format!(
                    "exceeds mobile budget {}",
                    MAX_AUTOSUGGEST_GENERATOR_PARAMETERS
                ),
            });
        }
        if self.generator.top_k_output > MAX_AUTOSUGGEST_GENERATOR_TOP_K_OUTPUT {
            return Err(AutosuggestGeneratorManifestError::Invalid {
                field: "generator.top_k_output",
                reason: format!(
                    "exceeds mobile budget {}",
                    MAX_AUTOSUGGEST_GENERATOR_TOP_K_OUTPUT
                ),
            });
        }
        if self.generator.top_k_output < contract.visible_candidates {
            return Err(AutosuggestGeneratorManifestError::Invalid {
                field: "generator.top_k_output",
                reason: format!(
                    "is smaller than visible candidate count {}",
                    contract.visible_candidates
                ),
            });
        }
        if contract.scored_union_policy.is_some()
            && (contract.candidate_ids_shape.is_none() || contract.candidate_scores_shape.is_none())
        {
            return Err(AutosuggestGeneratorManifestError::Invalid {
                field: "runtime_contract.scored_union_policy",
                reason: "requires candidate input and score shapes".to_string(),
            });
        }
        if let Some(candidate_ids_shape) = contract.candidate_ids_shape {
            if candidate_ids_shape[1] > MAX_AUTOSUGGEST_GENERATOR_CANDIDATE_POOL {
                return Err(AutosuggestGeneratorManifestError::Invalid {
                    field: "runtime_contract.candidate_ids_shape",
                    reason: format!(
                        "candidate pool exceeds mobile budget {}",
                        MAX_AUTOSUGGEST_GENERATOR_CANDIDATE_POOL
                    ),
                });
            }
            if let Some(pool_k) = self.generator.pool_k {
                require_equal_usize("generator.pool_k", candidate_ids_shape[1], pool_k)?;
            }
        }
        if let Some(export_kind) = &self.generator.export_kind {
            require_equal_str(
                "generator.export_kind",
                "full-vocab-topk-scorer",
                export_kind,
            )?;
        }
        require_equal_str(
            "generator.architecture",
            "gru",
            &self.generator.architecture,
        )?;
        require_equal_str(
            "generator.coreml_target",
            "ios17",
            &self.generator.coreml_target,
        )?;
        require_equal_str(
            "generator.coreml_precision",
            "float16",
            &self.generator.coreml_precision,
        )?;
        require_equal_str(
            "generator.coreml_compute_unit",
            "cpu_and_ne",
            &self.generator.coreml_compute_unit,
        )?;
        require_max_usize(
            "generator.onnx.bytes",
            self.generator.onnx.bytes,
            MAX_AUTOSUGGEST_GENERATOR_ONNX_BYTES,
        )?;
        require_max_usize(
            "generator.quantized_onnx.bytes",
            self.generator.quantized_onnx.bytes,
            MAX_AUTOSUGGEST_GENERATOR_QUANTIZED_ONNX_BYTES,
        )?;
        require_max_usize(
            "generator.coreml.bytes",
            self.generator.coreml.bytes,
            MAX_AUTOSUGGEST_GENERATOR_COREML_BYTES,
        )?;
        require_max_f64(
            "benchmark.onnx_mean_us_per_item",
            self.benchmark.onnx_mean_us_per_item,
            MAX_AUTOSUGGEST_GENERATOR_GRAPH_US_PER_ITEM,
        )?;
        require_max_f64(
            "benchmark.quantized_onnx_mean_us_per_item",
            self.benchmark.quantized_onnx_mean_us_per_item,
            MAX_AUTOSUGGEST_GENERATOR_GRAPH_US_PER_ITEM,
        )?;
        require_max_f64(
            "benchmark.coreml_mean_us_per_item",
            self.benchmark.coreml_mean_us_per_item,
            MAX_AUTOSUGGEST_GENERATOR_GRAPH_US_PER_ITEM,
        )?;
        let session_heap_limit = generator_session_heap_limit_bytes(
            self.generator.context_window,
            self.generator.top_k_output,
            contract.candidate_ids_shape.map(|shape| shape[1]),
        );
        require_max_usize(
            "runtime_contract.session_heap_limit_bytes",
            session_heap_limit,
            MAX_AUTOSUGGEST_GENERATOR_SESSION_HEAP_BYTES,
        )?;
        if self.quality.eligible_targets < MIN_AUTOSUGGEST_GENERATOR_ELIGIBLE_TARGETS {
            return Err(AutosuggestGeneratorManifestError::Invalid {
                field: "quality.eligible_targets",
                reason: format!(
                    "is below minimum production gate {}",
                    MIN_AUTOSUGGEST_GENERATOR_ELIGIBLE_TARGETS
                ),
            });
        }
        validate_quality_metrics("quality.static_pool", &self.quality.static_pool)?;
        validate_quality_metrics("quality.selected_topk", &self.quality.selected_topk)?;
        let union_gain = self.quality.union_recall_all_target_gain;
        if !union_gain.is_finite() || union_gain <= 0.0 {
            return Err(AutosuggestGeneratorManifestError::Invalid {
                field: "quality.union_recall_all_target_gain",
                reason: "must be finite and positive".to_string(),
            });
        }
        if let Some(merged) = &self.quality.selected_merged_visible {
            require_equal_usize(
                "quality.selected_merged_visible.visible_candidates",
                contract.visible_candidates,
                merged.visible_candidates,
            )?;
            if !merged.top5_all_target_gain_vs_static.is_finite()
                || !merged.top10_all_target_gain_vs_static.is_finite()
                || !merged.mrr_all_target_gain_vs_static.is_finite()
            {
                return Err(AutosuggestGeneratorManifestError::Invalid {
                    field: "quality.selected_merged_visible",
                    reason: "gain metrics must be finite".to_string(),
                });
            }
        }
        if let Some(scored_union) = &self.quality.selected_scored_union {
            validate_quality_metrics("quality.selected_scored_union", scored_union)?;
            let policy =
                contract
                    .scored_union_policy
                    .ok_or(AutosuggestGeneratorManifestError::Invalid {
                        field: "quality.selected_scored_union",
                        reason: "requires runtime_contract.scored_union_policy".to_string(),
                    })?;
            require_equal_usize(
                "quality.selected_scored_union.locked_static_prefix",
                policy.locked_static_prefix,
                scored_union.locked_static_prefix,
            )?;
            require_equal_f32(
                "quality.selected_scored_union.static_bonus",
                policy.static_bonus,
                scored_union.static_bonus,
            )?;
            require_equal_f32(
                "quality.selected_scored_union.static_rank_penalty",
                policy.static_rank_penalty,
                scored_union.static_rank_penalty,
            )?;
            require_equal_f32(
                "quality.selected_scored_union.generated_penalty",
                policy.generated_penalty,
                scored_union.generated_penalty,
            )?;
            require_equal_f32(
                "quality.selected_scored_union.static_source_bonus",
                policy.static_source_bonus,
                scored_union.static_source_bonus,
            )?;
            if !scored_union.top5_all_target_gain_vs_static.is_finite()
                || !scored_union.top10_all_target_gain_vs_static.is_finite()
                || !scored_union.mrr_all_target_gain_vs_static.is_finite()
            {
                return Err(AutosuggestGeneratorManifestError::Invalid {
                    field: "quality.selected_scored_union",
                    reason: "gain metrics must be finite".to_string(),
                });
            }
            if scored_union.top5_all_target_gain_vs_static <= 0.0 {
                return Err(AutosuggestGeneratorManifestError::Invalid {
                    field: "quality.selected_scored_union.top5_all_target_gain_vs_static",
                    reason: "must be positive".to_string(),
                });
            }
            if scored_union.mrr_all_target_gain_vs_static <= 0.0 {
                return Err(AutosuggestGeneratorManifestError::Invalid {
                    field: "quality.selected_scored_union.mrr_all_target_gain_vs_static",
                    reason: "must be positive".to_string(),
                });
            }
            if !scored_union.accepted_for_packaging {
                return Err(AutosuggestGeneratorManifestError::Invalid {
                    field: "quality.selected_scored_union.accepted_for_packaging",
                    reason: "must be true for production manifests".to_string(),
                });
            }
            if !scored_union.accepted_for_packaging_all_eval_sources {
                return Err(AutosuggestGeneratorManifestError::Invalid {
                    field: "quality.selected_scored_union.accepted_for_packaging_all_eval_sources",
                    reason: "must be true for source-balanced production manifests".to_string(),
                });
            }
            if scored_union.eval_per_source.is_empty() {
                return Err(AutosuggestGeneratorManifestError::Invalid {
                    field: "quality.selected_scored_union.eval_per_source",
                    reason: "must include per-source held-out metrics".to_string(),
                });
            }
            for source in scored_union.eval_per_source.values() {
                validate_source_quality(source)?;
            }
        } else if contract.scored_union_policy.is_some() {
            return Err(AutosuggestGeneratorManifestError::Invalid {
                field: "quality.selected_scored_union",
                reason: "is required when a scored-union policy is declared".to_string(),
            });
        }

        Ok(AutosuggestGeneratorCompatibility {
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
            context_window: self.generator.context_window,
            top_k_output: self.generator.top_k_output,
            candidate_pool: contract.candidate_ids_shape.map(|shape| shape[1]),
            visible_candidates: contract.visible_candidates,
            scored_union_policy: contract.scored_union_policy,
            neural_recall_all_targets: self.quality.neural_recall_all_targets,
            union_recall_all_targets: self.quality.union_recall_all_targets,
            union_recall_all_target_gain: union_gain,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn validate_asset_paths(
        &self,
        root: impl AsRef<Path>,
    ) -> Result<AutosuggestGeneratorAssetReport, AutosuggestGeneratorManifestError> {
        let root = root.as_ref();
        let ngram =
            validate_file_asset(root, "ngram.path", &self.ngram.path, self.ngram.bytes, None)?;
        let onnx = validate_file_asset(
            root,
            "generator.onnx.path",
            &self.generator.onnx.path,
            self.generator.onnx.bytes,
            Some(&self.generator.onnx.sha256),
        )?;
        let quantized_onnx = validate_file_asset(
            root,
            "generator.quantized_onnx.path",
            &self.generator.quantized_onnx.path,
            self.generator.quantized_onnx.bytes,
            Some(&self.generator.quantized_onnx.sha256),
        )?;
        let coreml = validate_package_asset(
            root,
            "generator.coreml.path",
            &self.generator.coreml.path,
            self.generator.coreml.bytes,
            Some(&self.generator.coreml.sha256),
        )?;

        Ok(AutosuggestGeneratorAssetReport {
            root: root.display().to_string(),
            ngram,
            onnx,
            quantized_onnx,
            coreml,
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestGeneratorRuntimeContract {
    pub token_id_dtype: String,
    pub onnx_input_dtype: String,
    pub coreml_input_dtype: String,
    pub scores_dtype: String,
    pub batch_size: usize,
    pub context_ids_shape: [usize; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_ids_shape: Option<[usize; 2]>,
    pub token_ids_shape: [usize; 2],
    pub scores_shape: [usize; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_scores_shape: Option<[usize; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scored_union_policy: Option<AutosuggestGeneratorScoredUnionPolicy>,
    pub pad_id: u32,
    pub bos_id: u32,
    pub unk_id: u32,
    pub visible_candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct AutosuggestGeneratorScoredUnionPolicy {
    pub locked_static_prefix: usize,
    pub static_bonus: f32,
    pub static_rank_penalty: f32,
    pub generated_penalty: f32,
    #[serde(default)]
    pub overlap_bonus: f32,
    #[serde(default)]
    pub generated_rank_penalty: f32,
    #[serde(default)]
    pub static_log_count_scale: f32,
    #[serde(default)]
    pub static_source_bonus: f32,
}

impl AutosuggestGeneratorScoredUnionPolicy {
    fn validate(&self, field: &'static str) -> Result<(), AutosuggestGeneratorManifestError> {
        if !self.static_bonus.is_finite()
            || !self.static_rank_penalty.is_finite()
            || !self.generated_penalty.is_finite()
            || !self.overlap_bonus.is_finite()
            || !self.generated_rank_penalty.is_finite()
            || !self.static_log_count_scale.is_finite()
            || !self.static_source_bonus.is_finite()
        {
            return Err(AutosuggestGeneratorManifestError::Invalid {
                field,
                reason: "score weights must be finite".to_string(),
            });
        }
        if self.static_rank_penalty < 0.0
            || self.generated_penalty < 0.0
            || self.generated_rank_penalty < 0.0
            || self.static_log_count_scale < 0.0
            || self.static_source_bonus < 0.0
        {
            return Err(AutosuggestGeneratorManifestError::Invalid {
                field,
                reason: "penalties must be non-negative".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestGeneratorNgram {
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
pub struct AutosuggestGeneratorModel {
    pub architecture: String,
    pub context_window: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub parameter_count: usize,
    pub top_k_output: usize,
    pub onnx: AutosuggestGeneratorFile,
    pub quantized_onnx: AutosuggestGeneratorFile,
    pub coreml: AutosuggestGeneratorFile,
    pub coreml_target: String,
    pub coreml_precision: String,
    pub coreml_compute_unit: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pool_k: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestGeneratorFile {
    pub path: String,
    pub bytes: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestGeneratorQuality {
    pub heldout_targets: usize,
    pub eligible_targets: usize,
    pub static_pool: AutosuggestGeneratorQualityMetrics,
    pub selected_topk: AutosuggestGeneratorQualityMetrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_merged_visible: Option<AutosuggestGeneratorMergedQuality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_scored_union: Option<AutosuggestGeneratorScoredUnionQuality>,
    pub static_pool_recall_all_targets: f64,
    pub neural_recall_all_targets: f64,
    pub union_recall_all_targets: f64,
    pub union_recall_all_target_gain: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestGeneratorQualityMetrics {
    pub top1_all_targets: f64,
    pub top5_all_targets: f64,
    pub top10_all_targets: f64,
    pub mrr_all_targets: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestGeneratorMergedQuality {
    pub top1_all_targets: f64,
    pub top5_all_targets: f64,
    pub top10_all_targets: f64,
    pub mrr_all_targets: f64,
    pub visible_candidates: usize,
    pub locked_static_prefix: usize,
    pub top5_all_target_gain_vs_static: f64,
    pub top10_all_target_gain_vs_static: f64,
    pub mrr_all_target_gain_vs_static: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestGeneratorScoredUnionQuality {
    pub top1_all_targets: f64,
    pub top5_all_targets: f64,
    pub top10_all_targets: f64,
    pub mrr_all_targets: f64,
    pub locked_static_prefix: usize,
    pub static_bonus: f32,
    pub static_rank_penalty: f32,
    pub generated_penalty: f32,
    #[serde(default)]
    pub overlap_bonus: f32,
    #[serde(default)]
    pub generated_rank_penalty: f32,
    #[serde(default)]
    pub static_log_count_scale: f32,
    #[serde(default)]
    pub static_source_bonus: f32,
    pub top5_all_target_gain_vs_static: f64,
    pub top10_all_target_gain_vs_static: f64,
    pub mrr_all_target_gain_vs_static: f64,
    #[serde(default)]
    pub accepted_for_packaging: bool,
    #[serde(default)]
    pub accepted_for_packaging_all_eval_sources: bool,
    #[serde(default)]
    pub eval_per_source: BTreeMap<String, AutosuggestGeneratorSourceQuality>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestGeneratorSourceQuality {
    pub eligible_targets: usize,
    pub top1_all_targets: f64,
    pub top5_all_targets: f64,
    pub top10_all_targets: f64,
    pub mrr_all_targets: f64,
    pub top5_all_target_gain_vs_static: f64,
    pub top10_all_target_gain_vs_static: f64,
    pub mrr_all_target_gain_vs_static: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutosuggestGeneratorBenchmark {
    pub onnx_mean_us_per_item: f64,
    pub quantized_onnx_mean_us_per_item: f64,
    pub coreml_mean_us_per_item: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AutosuggestGeneratorCompatibility {
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
    pub top_k_output: usize,
    pub candidate_pool: Option<usize>,
    pub visible_candidates: usize,
    pub scored_union_policy: Option<AutosuggestGeneratorScoredUnionPolicy>,
    pub neural_recall_all_targets: f64,
    pub union_recall_all_targets: f64,
    pub union_recall_all_target_gain: f64,
}

/// Fixed-shape runtime bridge between Obadh's context state and a platform
/// top-k next-word model.
///
/// This type deliberately does not run neural inference. Native integrations
/// prepare `contexts`, call Core ML/ONNX with a fixed batch of one, then feed
/// returned token IDs back through `generated_*_candidates_into` for validation,
/// deduping, and lazy materialization against the compact n-gram vocabulary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AutosuggestGeneratorHandoff {
    pub compatibility: AutosuggestGeneratorCompatibility,
    pub context_window: usize,
    pub top_k_output: usize,
    pub candidate_pool: Option<usize>,
    pub visible_candidates: usize,
    pub scored_union_policy: Option<AutosuggestGeneratorScoredUnionPolicy>,
}

impl AutosuggestGeneratorHandoff {
    pub fn from_manifest_for_lm<D: AsRef<[u8]>>(
        manifest: &AutosuggestGeneratorManifest,
        lm: &AutosuggestLm<D>,
    ) -> Result<Self, AutosuggestGeneratorManifestError> {
        let compatibility = manifest.validate_for_lm(lm)?;
        Ok(Self {
            context_window: manifest.runtime_contract.context_ids_shape[1],
            top_k_output: manifest.runtime_contract.token_ids_shape[1],
            candidate_pool: manifest
                .runtime_contract
                .candidate_ids_shape
                .map(|shape| shape[1]),
            visible_candidates: manifest.runtime_contract.visible_candidates,
            scored_union_policy: manifest.runtime_contract.scored_union_policy,
            compatibility,
        })
    }

    pub fn u32_context_for_context_into<D: AsRef<[u8]>>(
        &self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
        context_ids: &mut [u32],
    ) -> Result<usize, AutosuggestGeneratorHandoffError> {
        self.require_context_buffer("context_ids", context_ids.len())?;
        Ok(lm.scorer_context_ids_for_context_into(context, context_ids)?)
    }

    pub fn coreml_context_for_context_into<D: AsRef<[u8]>>(
        &self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
        context_ids: &mut [i32],
    ) -> Result<usize, AutosuggestGeneratorHandoffError> {
        self.require_context_buffer("context_ids", context_ids.len())?;
        Ok(lm.scorer_context_i32s_for_context_into(context, context_ids)?)
    }

    pub fn generated_u32_candidates_into<D: AsRef<[u8]>>(
        &self,
        lm: &AutosuggestLm<D>,
        token_ids: &[u32],
        model_scores: &[f32],
        output: &mut Vec<AutosuggestGeneratedCandidateId>,
    ) -> Result<(), AutosuggestGeneratorHandoffError> {
        self.require_output_buffers(token_ids.len(), model_scores.len())?;
        output.clear();
        output.reserve(self.top_k_output);
        for (index, (&token_id, &score)) in token_ids.iter().zip(model_scores.iter()).enumerate() {
            push_generated_candidate(lm, token_id, index, score, output)?;
        }
        Ok(())
    }

    pub fn generated_i32_candidates_into<D: AsRef<[u8]>>(
        &self,
        lm: &AutosuggestLm<D>,
        token_ids: &[i32],
        model_scores: &[f32],
        output: &mut Vec<AutosuggestGeneratedCandidateId>,
    ) -> Result<(), AutosuggestGeneratorHandoffError> {
        self.require_output_buffers(token_ids.len(), model_scores.len())?;
        output.clear();
        output.reserve(self.top_k_output);
        for (index, (&token_id, &score)) in token_ids.iter().zip(model_scores.iter()).enumerate() {
            if token_id < 0 {
                return Err(AutosuggestGeneratorHandoffError::InvalidGeneratedTokenId {
                    index,
                    token_id: i64::from(token_id),
                });
            }
            push_generated_candidate(lm, token_id as u32, index, score, output)?;
        }
        Ok(())
    }

    fn require_context_buffer(
        &self,
        field: &'static str,
        actual: usize,
    ) -> Result<(), AutosuggestGeneratorHandoffError> {
        if actual != self.context_window {
            return Err(AutosuggestGeneratorHandoffError::InvalidBuffer {
                field,
                expected: self.context_window,
                actual,
            });
        }
        Ok(())
    }

    fn require_output_buffers(
        &self,
        token_ids_len: usize,
        model_scores_len: usize,
    ) -> Result<(), AutosuggestGeneratorHandoffError> {
        if token_ids_len != self.top_k_output {
            return Err(AutosuggestGeneratorHandoffError::InvalidBuffer {
                field: "token_ids",
                expected: self.top_k_output,
                actual: token_ids_len,
            });
        }
        if model_scores_len != self.top_k_output {
            return Err(AutosuggestGeneratorHandoffError::InvalidBuffer {
                field: "model_scores",
                expected: self.top_k_output,
                actual: model_scores_len,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AutosuggestGeneratedCandidateId {
    pub token_id: u32,
    pub model_rank: usize,
    pub model_score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AutosuggestMergedCandidateSource {
    Static,
    Generated,
    StaticAndGenerated,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutosuggestGeneratorMergeOptions {
    pub max_candidates: usize,
    pub locked_static_prefix: usize,
}

impl Default for AutosuggestGeneratorMergeOptions {
    fn default() -> Self {
        Self {
            max_candidates: DEFAULT_AUTOSUGGEST_CANDIDATES,
            locked_static_prefix: DEFAULT_AUTOSUGGEST_GENERATOR_LOCKED_STATIC_PREFIX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AutosuggestMergedCandidateId {
    pub token_id: u32,
    pub source: AutosuggestMergedCandidateSource,
    pub static_rank: Option<usize>,
    pub static_candidate: Option<AutosuggestCandidateId>,
    pub model_rank: Option<usize>,
    pub model_score: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AutosuggestScoredUnionCandidateId {
    pub token_id: u32,
    pub source: AutosuggestMergedCandidateSource,
    pub static_rank: Option<usize>,
    pub static_candidate: Option<AutosuggestCandidateId>,
    pub model_rank: Option<usize>,
    pub generated_model_score: Option<f32>,
    pub static_model_score: Option<f32>,
    pub union_score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AutosuggestMaterializedGeneratedCandidate<'a> {
    pub text: &'a str,
    pub token_id: u32,
    pub model_rank: usize,
    pub model_score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AutosuggestMaterializedMergedCandidate<'a> {
    pub text: &'a str,
    pub token_id: u32,
    pub source: AutosuggestMergedCandidateSource,
    pub static_rank: Option<usize>,
    pub static_source: Option<&'static str>,
    pub static_count: Option<u32>,
    pub static_score: Option<i32>,
    pub model_rank: Option<usize>,
    pub model_score: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AutosuggestMaterializedScoredUnionCandidate<'a> {
    pub text: &'a str,
    pub token_id: u32,
    pub source: AutosuggestMergedCandidateSource,
    pub static_rank: Option<usize>,
    pub static_source: Option<&'static str>,
    pub static_count: Option<u32>,
    pub static_score: Option<i32>,
    pub model_rank: Option<usize>,
    pub generated_model_score: Option<f32>,
    pub static_model_score: Option<f32>,
    pub union_score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AutosuggestGeneratorHandoffError {
    Artifact(AutosuggestArtifactError),
    InvalidBuffer {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    MissingCandidatePool,
    MissingScoredUnionPolicy,
    InvalidGeneratedTokenId {
        index: usize,
        token_id: i64,
    },
}

impl fmt::Display for AutosuggestGeneratorHandoffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => error.fmt(f),
            Self::InvalidBuffer {
                field,
                expected,
                actual,
            } => write!(
                f,
                "autosuggest generator handoff expected {field} length {expected}, got {actual}"
            ),
            Self::MissingCandidatePool => write!(
                f,
                "autosuggest generator manifest does not declare a static candidate pool"
            ),
            Self::MissingScoredUnionPolicy => write!(
                f,
                "autosuggest generator manifest does not declare a scored-union policy"
            ),
            Self::InvalidGeneratedTokenId { index, token_id } => write!(
                f,
                "autosuggest generator returned invalid token id {token_id} at rank {index}"
            ),
        }
    }
}

impl Error for AutosuggestGeneratorHandoffError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Artifact(error) => Some(error),
            Self::InvalidBuffer { .. }
            | Self::MissingCandidatePool
            | Self::MissingScoredUnionPolicy
            | Self::InvalidGeneratedTokenId { .. } => None,
        }
    }
}

impl From<AutosuggestArtifactError> for AutosuggestGeneratorHandoffError {
    fn from(error: AutosuggestArtifactError) -> Self {
        Self::Artifact(error)
    }
}

/// Reusable generator handoff state for a keyboard/editor session.
///
/// The session owns only fixed-size input/output buffers and a small visible
/// result vector. Platform code still owns the actual model invocation.
#[derive(Debug, Clone)]
pub struct AutosuggestGeneratorSession {
    handoff: AutosuggestGeneratorHandoff,
    context_ids_u32: Vec<u32>,
    context_ids_i32: Vec<i32>,
    candidate_ids_u32: Vec<u32>,
    candidate_ids_i32: Vec<i32>,
    static_candidates: Vec<AutosuggestCandidateId>,
    generated: Vec<AutosuggestGeneratedCandidateId>,
    generated_text: Vec<AutosuggestValidatedTextCandidate>,
    merged: Vec<AutosuggestMergedCandidateId>,
    scored_union: Vec<AutosuggestScoredUnionCandidateId>,
    unified: Vec<AutosuggestUnifiedCandidate>,
}

impl AutosuggestGeneratorSession {
    pub fn new(handoff: AutosuggestGeneratorHandoff) -> Self {
        let context_window = handoff.context_window;
        let top_k_output = handoff.top_k_output;
        let candidate_pool = handoff.candidate_pool.unwrap_or(0);
        let union_capacity = top_k_output + candidate_pool;
        Self {
            handoff,
            context_ids_u32: vec![AUTOSUGGEST_PAD_ID; context_window],
            context_ids_i32: vec![AUTOSUGGEST_PAD_ID as i32; context_window],
            candidate_ids_u32: vec![AUTOSUGGEST_PAD_ID; candidate_pool],
            candidate_ids_i32: vec![AUTOSUGGEST_PAD_ID as i32; candidate_pool],
            static_candidates: Vec::with_capacity(candidate_pool),
            generated: Vec::with_capacity(top_k_output),
            generated_text: Vec::with_capacity(top_k_output),
            merged: Vec::with_capacity(top_k_output),
            scored_union: Vec::with_capacity(union_capacity),
            unified: Vec::with_capacity(union_capacity),
        }
    }

    pub fn from_manifest_for_lm<D: AsRef<[u8]>>(
        manifest: &AutosuggestGeneratorManifest,
        lm: &AutosuggestLm<D>,
    ) -> Result<Self, AutosuggestGeneratorManifestError> {
        Ok(Self::new(
            AutosuggestGeneratorHandoff::from_manifest_for_lm(manifest, lm)?,
        ))
    }

    pub fn handoff(&self) -> &AutosuggestGeneratorHandoff {
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

    pub fn static_candidates(&self) -> &[AutosuggestCandidateId] {
        &self.static_candidates
    }

    pub fn generated_candidates(&self) -> &[AutosuggestGeneratedCandidateId] {
        &self.generated
    }

    pub fn generated_text_candidates(&self) -> &[AutosuggestValidatedTextCandidate] {
        &self.generated_text
    }

    pub fn merged_candidates(&self) -> &[AutosuggestMergedCandidateId] {
        &self.merged
    }

    pub fn scored_union_candidates(&self) -> &[AutosuggestScoredUnionCandidateId] {
        &self.scored_union
    }

    pub fn unified_candidates(&self) -> &[AutosuggestUnifiedCandidate] {
        &self.unified
    }

    pub fn prepare_u32_context<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
    ) -> Result<usize, AutosuggestGeneratorHandoffError> {
        self.clear_outputs();
        self.clear_static_candidates();
        self.handoff
            .u32_context_for_context_into(lm, context, &mut self.context_ids_u32)
    }

    pub fn prepare_u32_inputs<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
    ) -> Result<usize, AutosuggestGeneratorHandoffError> {
        let context_len = self.prepare_u32_context(lm, context)?;
        self.fill_static_candidates_from_lm(lm, context)?;
        scorer_candidate_ids_for_candidates_into(
            &self.static_candidates,
            &mut self.candidate_ids_u32,
        );
        Ok(context_len)
    }

    pub fn prepare_u32_inputs_for_autosuggest_session<'lm, D: AsRef<[u8]>>(
        &mut self,
        session: &mut AutosuggestSession<'lm, D>,
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestGeneratorHandoffError> {
        self.clear_outputs();
        self.clear_static_candidates();
        let candidate_pool = self
            .handoff
            .candidate_pool
            .ok_or(AutosuggestGeneratorHandoffError::MissingCandidatePool)?;
        let metadata = session.rerank_u32_input_with_options_into(
            AutosuggestOptions {
                max_candidates: candidate_pool,
            },
            &mut self.context_ids_u32,
            &mut self.candidate_ids_u32,
            &mut self.static_candidates,
        )?;
        Ok(metadata)
    }

    pub fn prepare_coreml_context<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
    ) -> Result<usize, AutosuggestGeneratorHandoffError> {
        self.clear_outputs();
        self.clear_static_candidates();
        self.handoff
            .coreml_context_for_context_into(lm, context, &mut self.context_ids_i32)
    }

    pub fn prepare_coreml_inputs<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
    ) -> Result<usize, AutosuggestGeneratorHandoffError> {
        let context_len = self.prepare_coreml_context(lm, context)?;
        self.fill_static_candidates_from_lm(lm, context)?;
        scorer_candidate_i32s_for_candidates_into(
            &self.static_candidates,
            &mut self.candidate_ids_i32,
        )?;
        Ok(context_len)
    }

    pub fn prepare_coreml_inputs_for_autosuggest_session<'lm, D: AsRef<[u8]>>(
        &mut self,
        session: &mut AutosuggestSession<'lm, D>,
    ) -> Result<AutosuggestRerankInputMetadata, AutosuggestGeneratorHandoffError> {
        self.clear_outputs();
        self.clear_static_candidates();
        let candidate_pool = self
            .handoff
            .candidate_pool
            .ok_or(AutosuggestGeneratorHandoffError::MissingCandidatePool)?;
        let metadata = session.rerank_coreml_input_with_options_into(
            AutosuggestOptions {
                max_candidates: candidate_pool,
            },
            &mut self.context_ids_i32,
            &mut self.candidate_ids_i32,
            &mut self.static_candidates,
        )?;
        Ok(metadata)
    }

    pub fn accept_u32_outputs<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        token_ids: &[u32],
        model_scores: &[f32],
    ) -> Result<&[AutosuggestGeneratedCandidateId], AutosuggestGeneratorHandoffError> {
        self.handoff.generated_u32_candidates_into(
            lm,
            token_ids,
            model_scores,
            &mut self.generated,
        )?;
        Ok(&self.generated)
    }

    pub fn accept_i32_outputs<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        token_ids: &[i32],
        model_scores: &[f32],
    ) -> Result<&[AutosuggestGeneratedCandidateId], AutosuggestGeneratorHandoffError> {
        self.handoff.generated_i32_candidates_into(
            lm,
            token_ids,
            model_scores,
            &mut self.generated,
        )?;
        Ok(&self.generated)
    }

    pub fn accept_open_vocab_text_outputs<S: AsRef<str>>(
        &mut self,
        texts: &[S],
        model_scores: &[f32],
        policy: AutosuggestOpenVocabPolicy,
    ) -> Result<&[AutosuggestValidatedTextCandidate], AutosuggestOpenVocabError> {
        accept_open_vocab_texts_into(texts, model_scores, policy, &mut self.generated_text)?;
        self.unified.clear();
        Ok(&self.generated_text)
    }

    pub fn merge_static_candidates(
        &mut self,
        static_candidates: &[AutosuggestCandidateId],
        options: AutosuggestGeneratorMergeOptions,
    ) -> &[AutosuggestMergedCandidateId] {
        merge_static_and_generated_candidates_into(
            static_candidates,
            &self.generated,
            options,
            &mut self.merged,
        );
        &self.merged
    }

    pub fn unified_static_generated_and_text_candidates<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        policy: AutosuggestOpenVocabPolicy,
    ) -> Result<&[AutosuggestUnifiedCandidate], AutosuggestOpenVocabError> {
        merge_static_generated_and_open_vocab_candidates_into(
            lm,
            &self.static_candidates,
            &self.generated,
            &self.generated_text,
            policy,
            &mut self.unified,
        )?;
        Ok(&self.unified)
    }

    pub fn unified_candidates_for_static_candidates<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        static_candidates: &[AutosuggestCandidateId],
        policy: AutosuggestOpenVocabPolicy,
    ) -> Result<&[AutosuggestUnifiedCandidate], AutosuggestOpenVocabError> {
        merge_static_generated_and_open_vocab_candidates_into(
            lm,
            static_candidates,
            &self.generated,
            &self.generated_text,
            policy,
            &mut self.unified,
        )?;
        Ok(&self.unified)
    }

    pub fn scored_union_static_candidates(
        &mut self,
        static_candidates: &[AutosuggestCandidateId],
        static_model_scores: &[f32],
    ) -> Result<&[AutosuggestScoredUnionCandidateId], AutosuggestGeneratorHandoffError> {
        let policy = self
            .handoff
            .scored_union_policy
            .ok_or(AutosuggestGeneratorHandoffError::MissingScoredUnionPolicy)?;
        scored_union_static_and_generated_candidates_into(
            static_candidates,
            static_model_scores,
            &self.generated,
            policy,
            self.handoff.visible_candidates,
            &mut self.scored_union,
        )?;
        Ok(&self.scored_union)
    }

    pub fn scored_union_with_static_scores(
        &mut self,
        static_model_scores: &[f32],
    ) -> Result<&[AutosuggestScoredUnionCandidateId], AutosuggestGeneratorHandoffError> {
        let policy = self
            .handoff
            .scored_union_policy
            .ok_or(AutosuggestGeneratorHandoffError::MissingScoredUnionPolicy)?;
        scored_union_static_and_generated_candidates_into(
            &self.static_candidates,
            static_model_scores,
            &self.generated,
            policy,
            self.handoff.visible_candidates,
            &mut self.scored_union,
        )?;
        Ok(&self.scored_union)
    }

    pub fn materialized_generated_candidates_into<'lm, D: AsRef<[u8]>>(
        &self,
        lm: &'lm AutosuggestLm<D>,
        output: &mut Vec<AutosuggestMaterializedGeneratedCandidate<'lm>>,
    ) -> Result<(), AutosuggestArtifactError> {
        materialize_generated_candidates_into(lm, &self.generated, output)
    }

    pub fn materialized_merged_candidates_into<'lm, D: AsRef<[u8]>>(
        &self,
        lm: &'lm AutosuggestLm<D>,
        output: &mut Vec<AutosuggestMaterializedMergedCandidate<'lm>>,
    ) -> Result<(), AutosuggestArtifactError> {
        materialize_merged_candidates_into(lm, &self.merged, output)
    }

    pub fn materialized_scored_union_candidates_into<'lm, D: AsRef<[u8]>>(
        &self,
        lm: &'lm AutosuggestLm<D>,
        output: &mut Vec<AutosuggestMaterializedScoredUnionCandidate<'lm>>,
    ) -> Result<(), AutosuggestArtifactError> {
        materialize_scored_union_candidates_into(lm, &self.scored_union, output)
    }

    pub fn estimated_heap_bytes(&self) -> usize {
        self.context_ids_u32.capacity() * mem::size_of::<u32>()
            + self.context_ids_i32.capacity() * mem::size_of::<i32>()
            + self.candidate_ids_u32.capacity() * mem::size_of::<u32>()
            + self.candidate_ids_i32.capacity() * mem::size_of::<i32>()
            + self.static_candidates.capacity() * mem::size_of::<AutosuggestCandidateId>()
            + self.generated.capacity() * mem::size_of::<AutosuggestGeneratedCandidateId>()
            + self.generated_text.capacity() * mem::size_of::<AutosuggestValidatedTextCandidate>()
            + self
                .generated_text
                .iter()
                .map(|candidate| candidate.text.capacity())
                .sum::<usize>()
            + self.merged.capacity() * mem::size_of::<AutosuggestMergedCandidateId>()
            + self.scored_union.capacity() * mem::size_of::<AutosuggestScoredUnionCandidateId>()
            + self.unified.capacity() * mem::size_of::<AutosuggestUnifiedCandidate>()
            + self
                .unified
                .iter()
                .map(|candidate| candidate.text.capacity())
                .sum::<usize>()
    }

    pub fn heap_limit_bytes(&self) -> usize {
        let context_window = self.handoff.context_window;
        let top_k_output = self.handoff.top_k_output;
        let candidate_pool = self.handoff.candidate_pool.unwrap_or(0);
        let union_capacity = top_k_output + candidate_pool;
        context_window * mem::size_of::<u32>()
            + context_window * mem::size_of::<i32>()
            + candidate_pool * mem::size_of::<u32>()
            + candidate_pool * mem::size_of::<i32>()
            + candidate_pool * mem::size_of::<AutosuggestCandidateId>()
            + top_k_output * mem::size_of::<AutosuggestGeneratedCandidateId>()
            + top_k_output * mem::size_of::<AutosuggestValidatedTextCandidate>()
            + top_k_output * DEFAULT_AUTOSUGGEST_OPEN_VOCAB_TEXT_HEAP_BYTES
            + top_k_output * mem::size_of::<AutosuggestMergedCandidateId>()
            + union_capacity * mem::size_of::<AutosuggestScoredUnionCandidateId>()
            + union_capacity * mem::size_of::<AutosuggestUnifiedCandidate>()
            + union_capacity * DEFAULT_AUTOSUGGEST_OPEN_VOCAB_TEXT_HEAP_BYTES
    }

    fn clear_outputs(&mut self) {
        self.generated.clear();
        self.generated_text.clear();
        self.merged.clear();
        self.scored_union.clear();
        self.unified.clear();
    }

    fn clear_static_candidates(&mut self) {
        self.static_candidates.clear();
        self.candidate_ids_u32.fill(AUTOSUGGEST_PAD_ID);
        self.candidate_ids_i32.fill(AUTOSUGGEST_PAD_ID as i32);
    }

    fn fill_static_candidates_from_lm<D: AsRef<[u8]>>(
        &mut self,
        lm: &AutosuggestLm<D>,
        context: AutosuggestContext,
    ) -> Result<(), AutosuggestGeneratorHandoffError> {
        let candidate_pool = self
            .handoff
            .candidate_pool
            .ok_or(AutosuggestGeneratorHandoffError::MissingCandidatePool)?;
        lm.suggest_ids_for_context_into(
            context,
            AutosuggestOptions {
                max_candidates: candidate_pool,
            },
            &mut self.static_candidates,
        )?;
        Ok(())
    }
}

pub fn merge_static_and_generated_candidates_into(
    static_candidates: &[AutosuggestCandidateId],
    generated: &[AutosuggestGeneratedCandidateId],
    options: AutosuggestGeneratorMergeOptions,
    output: &mut Vec<AutosuggestMergedCandidateId>,
) {
    output.clear();
    let limit = options.max_candidates.max(1);
    output.reserve(limit);

    let locked = options
        .locked_static_prefix
        .min(static_candidates.len())
        .min(limit);
    for (rank, candidate) in static_candidates.iter().copied().enumerate().take(locked) {
        push_or_merge_static_candidate(rank, candidate, limit, output);
    }
    for candidate in generated {
        push_or_merge_generated_candidate(*candidate, limit, output);
    }
    for (rank, candidate) in static_candidates.iter().copied().enumerate().skip(locked) {
        push_or_merge_static_candidate(rank, candidate, limit, output);
    }
}

pub fn scored_union_static_and_generated_candidates_into(
    static_candidates: &[AutosuggestCandidateId],
    static_model_scores: &[f32],
    generated: &[AutosuggestGeneratedCandidateId],
    policy: AutosuggestGeneratorScoredUnionPolicy,
    max_candidates: usize,
    output: &mut Vec<AutosuggestScoredUnionCandidateId>,
) -> Result<(), AutosuggestGeneratorHandoffError> {
    output.clear();
    if static_model_scores.len() < static_candidates.len() {
        return Err(AutosuggestGeneratorHandoffError::InvalidBuffer {
            field: "static_model_scores",
            expected: static_candidates.len(),
            actual: static_model_scores.len(),
        });
    }

    let limit = max_candidates.max(1);
    output.reserve(static_candidates.len() + generated.len());
    for (rank, candidate) in static_candidates.iter().copied().enumerate() {
        let static_model_score = finite_model_score(static_model_scores[rank]);
        output.push(AutosuggestScoredUnionCandidateId {
            token_id: candidate.token_id,
            source: AutosuggestMergedCandidateSource::Static,
            static_rank: Some(rank),
            static_candidate: Some(candidate),
            model_rank: None,
            generated_model_score: None,
            static_model_score: Some(static_model_score),
            union_score: static_union_score(static_model_score, rank, candidate, policy),
        });
    }
    for candidate in generated {
        push_or_merge_scored_generated_candidate(*candidate, policy, output);
    }
    output.sort_by(scored_union_candidate_order);
    output.truncate(limit);
    Ok(())
}

pub fn materialize_generated_candidates_into<'lm, D: AsRef<[u8]>>(
    lm: &'lm AutosuggestLm<D>,
    generated: &[AutosuggestGeneratedCandidateId],
    output: &mut Vec<AutosuggestMaterializedGeneratedCandidate<'lm>>,
) -> Result<(), AutosuggestArtifactError> {
    output.clear();
    output.reserve(generated.len());
    for candidate in generated {
        output.push(AutosuggestMaterializedGeneratedCandidate {
            text: lm.token_text(candidate.token_id)?,
            token_id: candidate.token_id,
            model_rank: candidate.model_rank,
            model_score: candidate.model_score,
        });
    }
    Ok(())
}

pub fn materialize_merged_candidates_into<'lm, D: AsRef<[u8]>>(
    lm: &'lm AutosuggestLm<D>,
    merged: &[AutosuggestMergedCandidateId],
    output: &mut Vec<AutosuggestMaterializedMergedCandidate<'lm>>,
) -> Result<(), AutosuggestArtifactError> {
    output.clear();
    output.reserve(merged.len());
    for candidate in merged {
        let static_candidate = candidate.static_candidate;
        output.push(AutosuggestMaterializedMergedCandidate {
            text: lm.token_text(candidate.token_id)?,
            token_id: candidate.token_id,
            source: candidate.source,
            static_rank: candidate.static_rank,
            static_source: static_candidate.map(|candidate| candidate.source.as_str()),
            static_count: static_candidate.map(|candidate| candidate.count),
            static_score: static_candidate.map(|candidate| candidate.score),
            model_rank: candidate.model_rank,
            model_score: candidate.model_score,
        });
    }
    Ok(())
}

pub fn materialize_scored_union_candidates_into<'lm, D: AsRef<[u8]>>(
    lm: &'lm AutosuggestLm<D>,
    scored: &[AutosuggestScoredUnionCandidateId],
    output: &mut Vec<AutosuggestMaterializedScoredUnionCandidate<'lm>>,
) -> Result<(), AutosuggestArtifactError> {
    output.clear();
    output.reserve(scored.len());
    for candidate in scored {
        let static_candidate = candidate.static_candidate;
        output.push(AutosuggestMaterializedScoredUnionCandidate {
            text: lm.token_text(candidate.token_id)?,
            token_id: candidate.token_id,
            source: candidate.source,
            static_rank: candidate.static_rank,
            static_source: static_candidate.map(|candidate| candidate.source.as_str()),
            static_count: static_candidate.map(|candidate| candidate.count),
            static_score: static_candidate.map(|candidate| candidate.score),
            model_rank: candidate.model_rank,
            generated_model_score: candidate.generated_model_score,
            static_model_score: candidate.static_model_score,
            union_score: candidate.union_score,
        });
    }
    Ok(())
}

fn push_generated_candidate<D: AsRef<[u8]>>(
    lm: &AutosuggestLm<D>,
    token_id: u32,
    model_rank: usize,
    model_score: f32,
    output: &mut Vec<AutosuggestGeneratedCandidateId>,
) -> Result<(), AutosuggestGeneratorHandoffError> {
    if token_id >= lm.vocab_size() as u32 {
        return Err(AutosuggestGeneratorHandoffError::InvalidGeneratedTokenId {
            index: model_rank,
            token_id: i64::from(token_id),
        });
    }
    if !lm.is_word_token_id(token_id)
        || output
            .iter()
            .any(|candidate| candidate.token_id == token_id)
    {
        return Ok(());
    }
    output.push(AutosuggestGeneratedCandidateId {
        token_id,
        model_rank,
        model_score: finite_model_score(model_score),
    });
    Ok(())
}

fn push_or_merge_static_candidate(
    rank: usize,
    candidate: AutosuggestCandidateId,
    limit: usize,
    output: &mut Vec<AutosuggestMergedCandidateId>,
) {
    if let Some(existing) = output
        .iter_mut()
        .find(|existing| existing.token_id == candidate.token_id)
    {
        existing.static_rank = Some(rank);
        existing.static_candidate = Some(candidate);
        existing.source = match existing.source {
            AutosuggestMergedCandidateSource::Generated => {
                AutosuggestMergedCandidateSource::StaticAndGenerated
            }
            source => source,
        };
        return;
    }
    if output.len() >= limit {
        return;
    }
    output.push(AutosuggestMergedCandidateId {
        token_id: candidate.token_id,
        source: AutosuggestMergedCandidateSource::Static,
        static_rank: Some(rank),
        static_candidate: Some(candidate),
        model_rank: None,
        model_score: None,
    });
}

fn push_or_merge_generated_candidate(
    candidate: AutosuggestGeneratedCandidateId,
    limit: usize,
    output: &mut Vec<AutosuggestMergedCandidateId>,
) {
    if let Some(existing) = output
        .iter_mut()
        .find(|existing| existing.token_id == candidate.token_id)
    {
        existing.model_rank = Some(candidate.model_rank);
        existing.model_score = Some(candidate.model_score);
        existing.source = match existing.source {
            AutosuggestMergedCandidateSource::Static => {
                AutosuggestMergedCandidateSource::StaticAndGenerated
            }
            source => source,
        };
        return;
    }
    if output.len() >= limit {
        return;
    }
    output.push(AutosuggestMergedCandidateId {
        token_id: candidate.token_id,
        source: AutosuggestMergedCandidateSource::Generated,
        static_rank: None,
        static_candidate: None,
        model_rank: Some(candidate.model_rank),
        model_score: Some(candidate.model_score),
    });
}

fn push_or_merge_scored_generated_candidate(
    candidate: AutosuggestGeneratedCandidateId,
    policy: AutosuggestGeneratorScoredUnionPolicy,
    output: &mut Vec<AutosuggestScoredUnionCandidateId>,
) {
    let generated_model_score = finite_model_score(candidate.model_score);
    if let Some(existing) = output
        .iter_mut()
        .find(|existing| existing.token_id == candidate.token_id)
    {
        existing.model_rank = Some(candidate.model_rank);
        existing.generated_model_score = Some(generated_model_score);
        existing.source = match existing.source {
            AutosuggestMergedCandidateSource::Static => {
                AutosuggestMergedCandidateSource::StaticAndGenerated
            }
            source => source,
        };
        existing.union_score = scored_union_score_for_candidate(existing, policy);
        return;
    }
    output.push(AutosuggestScoredUnionCandidateId {
        token_id: candidate.token_id,
        source: AutosuggestMergedCandidateSource::Generated,
        static_rank: None,
        static_candidate: None,
        model_rank: Some(candidate.model_rank),
        generated_model_score: Some(generated_model_score),
        static_model_score: None,
        union_score: generated_model_score
            - policy.generated_penalty
            - policy.generated_rank_penalty * candidate.model_rank as f32,
    });
}

fn finite_model_score(score: f32) -> f32 {
    if score.is_finite() {
        score
    } else {
        f32::NEG_INFINITY
    }
}

fn scored_union_score_for_candidate(
    candidate: &AutosuggestScoredUnionCandidateId,
    policy: AutosuggestGeneratorScoredUnionPolicy,
) -> f32 {
    let base_score = candidate
        .static_model_score
        .into_iter()
        .chain(candidate.generated_model_score)
        .fold(f32::NEG_INFINITY, f32::max);
    let generated_rank_penalty = candidate
        .model_rank
        .map(|rank| policy.generated_rank_penalty * rank as f32)
        .unwrap_or(0.0);
    let overlap_bonus = if candidate.static_rank.is_some() && candidate.model_rank.is_some() {
        policy.overlap_bonus
    } else {
        0.0
    };
    if let Some(rank) = candidate.static_rank {
        let static_candidate = candidate
            .static_candidate
            .unwrap_or(AutosuggestCandidateId {
                token_id: candidate.token_id,
                source: AutosuggestSource::Unigram,
                count: 0,
                score: 0,
            });
        static_union_score(base_score, rank, static_candidate, policy) + overlap_bonus
            - generated_rank_penalty
    } else {
        base_score - policy.generated_penalty - generated_rank_penalty
    }
}

fn static_union_score(
    model_score: f32,
    static_rank: usize,
    static_candidate: AutosuggestCandidateId,
    policy: AutosuggestGeneratorScoredUnionPolicy,
) -> f32 {
    model_score + policy.static_bonus - policy.static_rank_penalty * static_rank as f32
        + policy.static_log_count_scale * (static_candidate.count as f32).ln_1p()
        + policy.static_source_bonus * static_source_order(static_candidate.source)
        + if static_rank < policy.locked_static_prefix {
            AUTOSUGGEST_GENERATOR_LOCKED_STATIC_BONUS
        } else {
            0.0
        }
}

fn static_source_order(source: AutosuggestSource) -> f32 {
    match source {
        AutosuggestSource::Personal => 4.0,
        AutosuggestSource::Fourgram => 3.0,
        AutosuggestSource::Trigram => 2.0,
        AutosuggestSource::Bigram => 1.0,
        AutosuggestSource::Unigram => 0.0,
    }
}

fn scored_union_candidate_order(
    left: &AutosuggestScoredUnionCandidateId,
    right: &AutosuggestScoredUnionCandidateId,
) -> std::cmp::Ordering {
    right
        .union_score
        .total_cmp(&left.union_score)
        .then_with(|| {
            left.static_rank
                .unwrap_or(usize::MAX)
                .cmp(&right.static_rank.unwrap_or(usize::MAX))
        })
        .then_with(|| {
            left.model_rank
                .unwrap_or(usize::MAX)
                .cmp(&right.model_rank.unwrap_or(usize::MAX))
        })
        .then_with(|| left.token_id.cmp(&right.token_id))
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AutosuggestGeneratorAssetReport {
    pub root: String,
    pub ngram: AutosuggestGeneratorAsset,
    pub onnx: AutosuggestGeneratorAsset,
    pub quantized_onnx: AutosuggestGeneratorAsset,
    pub coreml: AutosuggestGeneratorAsset,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AutosuggestGeneratorAsset {
    pub path: String,
    pub bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AutosuggestGeneratorManifestError {
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

impl fmt::Display for AutosuggestGeneratorManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "autosuggest generator manifest mismatch for {field}: expected {expected}, got {actual}"
            ),
            Self::Invalid { field, reason } => write!(
                f,
                "autosuggest generator manifest has invalid {field}: {reason}"
            ),
            Self::Asset {
                field,
                path,
                reason,
            } => write!(
                f,
                "autosuggest generator manifest asset check failed for {field} at {path}: {reason}"
            ),
        }
    }
}

impl Error for AutosuggestGeneratorManifestError {}

fn require_shape(
    field: &'static str,
    actual: [usize; 2],
) -> Result<(), AutosuggestGeneratorManifestError> {
    if actual[0] == 0 || actual[1] == 0 {
        return Err(AutosuggestGeneratorManifestError::Invalid {
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
) -> Result<(), AutosuggestGeneratorManifestError> {
    if expected != actual {
        return Err(AutosuggestGeneratorManifestError::Mismatch {
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
) -> Result<(), AutosuggestGeneratorManifestError> {
    if expected != actual {
        return Err(AutosuggestGeneratorManifestError::Mismatch {
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
) -> Result<(), AutosuggestGeneratorManifestError> {
    if expected != actual {
        return Err(AutosuggestGeneratorManifestError::Mismatch {
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
) -> Result<(), AutosuggestGeneratorManifestError> {
    if !expected.is_finite() || !actual.is_finite() || (expected - actual).abs() > f32::EPSILON {
        return Err(AutosuggestGeneratorManifestError::Mismatch {
            field,
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

fn require_max_usize(
    field: &'static str,
    actual: usize,
    max: usize,
) -> Result<(), AutosuggestGeneratorManifestError> {
    if actual > max {
        return Err(AutosuggestGeneratorManifestError::Invalid {
            field,
            reason: format!("exceeds maximum {max}"),
        });
    }
    Ok(())
}

fn require_max_f64(
    field: &'static str,
    actual: f64,
    max: f64,
) -> Result<(), AutosuggestGeneratorManifestError> {
    if !actual.is_finite() || actual <= 0.0 || actual > max {
        return Err(AutosuggestGeneratorManifestError::Invalid {
            field,
            reason: format!("must be finite, positive, and at most {max}"),
        });
    }
    Ok(())
}

trait AutosuggestQualityValues {
    fn top1_all_targets(&self) -> f64;
    fn top5_all_targets(&self) -> f64;
    fn top10_all_targets(&self) -> f64;
    fn mrr_all_targets(&self) -> f64;
}

impl AutosuggestQualityValues for AutosuggestGeneratorQualityMetrics {
    fn top1_all_targets(&self) -> f64 {
        self.top1_all_targets
    }

    fn top5_all_targets(&self) -> f64 {
        self.top5_all_targets
    }

    fn top10_all_targets(&self) -> f64 {
        self.top10_all_targets
    }

    fn mrr_all_targets(&self) -> f64 {
        self.mrr_all_targets
    }
}

impl AutosuggestQualityValues for AutosuggestGeneratorScoredUnionQuality {
    fn top1_all_targets(&self) -> f64 {
        self.top1_all_targets
    }

    fn top5_all_targets(&self) -> f64 {
        self.top5_all_targets
    }

    fn top10_all_targets(&self) -> f64 {
        self.top10_all_targets
    }

    fn mrr_all_targets(&self) -> f64 {
        self.mrr_all_targets
    }
}

impl AutosuggestQualityValues for AutosuggestGeneratorSourceQuality {
    fn top1_all_targets(&self) -> f64 {
        self.top1_all_targets
    }

    fn top5_all_targets(&self) -> f64 {
        self.top5_all_targets
    }

    fn top10_all_targets(&self) -> f64 {
        self.top10_all_targets
    }

    fn mrr_all_targets(&self) -> f64 {
        self.mrr_all_targets
    }
}

fn validate_quality_metrics<T: AutosuggestQualityValues>(
    field: &'static str,
    metrics: &T,
) -> Result<(), AutosuggestGeneratorManifestError> {
    let top1 = metrics.top1_all_targets();
    let top5 = metrics.top5_all_targets();
    let top10 = metrics.top10_all_targets();
    let mrr = metrics.mrr_all_targets();
    if !is_probability(top1)
        || !is_probability(top5)
        || !is_probability(top10)
        || !is_probability(mrr)
    {
        return Err(AutosuggestGeneratorManifestError::Invalid {
            field,
            reason: "quality metrics must be finite probabilities".to_string(),
        });
    }
    if top1 > top5 || top5 > top10 {
        return Err(AutosuggestGeneratorManifestError::Invalid {
            field,
            reason: "top-k metrics must be monotonic".to_string(),
        });
    }
    if mrr < top1 || mrr > top5 {
        return Err(AutosuggestGeneratorManifestError::Invalid {
            field,
            reason: "MRR must stay between top-1 and top-5".to_string(),
        });
    }
    Ok(())
}

fn validate_source_quality(
    source: &AutosuggestGeneratorSourceQuality,
) -> Result<(), AutosuggestGeneratorManifestError> {
    if source.eligible_targets < MIN_AUTOSUGGEST_GENERATOR_SOURCE_ELIGIBLE_TARGETS {
        return Err(AutosuggestGeneratorManifestError::Invalid {
            field: "quality.selected_scored_union.eval_per_source",
            reason: format!(
                "source eligible targets are below minimum production gate {}",
                MIN_AUTOSUGGEST_GENERATOR_SOURCE_ELIGIBLE_TARGETS
            ),
        });
    }
    validate_quality_metrics("quality.selected_scored_union.eval_per_source", source)?;
    if !source.top5_all_target_gain_vs_static.is_finite()
        || !source.top10_all_target_gain_vs_static.is_finite()
        || !source.mrr_all_target_gain_vs_static.is_finite()
    {
        return Err(AutosuggestGeneratorManifestError::Invalid {
            field: "quality.selected_scored_union.eval_per_source",
            reason: "source gains must be finite".to_string(),
        });
    }
    if source.top5_all_target_gain_vs_static <= 0.0 || source.mrr_all_target_gain_vs_static <= 0.0 {
        return Err(AutosuggestGeneratorManifestError::Invalid {
            field: "quality.selected_scored_union.eval_per_source",
            reason: "source top-5 and MRR gains must be positive".to_string(),
        });
    }
    Ok(())
}

fn is_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn generator_session_heap_limit_bytes(
    context_window: usize,
    top_k_output: usize,
    candidate_pool: Option<usize>,
) -> usize {
    let candidate_pool = candidate_pool.unwrap_or(0);
    let union_capacity = top_k_output + candidate_pool;
    context_window * mem::size_of::<u32>()
        + context_window * mem::size_of::<i32>()
        + candidate_pool * mem::size_of::<u32>()
        + candidate_pool * mem::size_of::<i32>()
        + candidate_pool * mem::size_of::<AutosuggestCandidateId>()
        + top_k_output * mem::size_of::<AutosuggestGeneratedCandidateId>()
        + top_k_output * mem::size_of::<AutosuggestMergedCandidateId>()
        + union_capacity * mem::size_of::<AutosuggestScoredUnionCandidateId>()
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_file_asset(
    root: &Path,
    field: &'static str,
    manifest_path: &str,
    expected_bytes: usize,
    expected_sha256: Option<&str>,
) -> Result<AutosuggestGeneratorAsset, AutosuggestGeneratorManifestError> {
    let path = resolve_asset_path(root, manifest_path);
    let metadata =
        fs::metadata(&path).map_err(|error| AutosuggestGeneratorManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
    if !metadata.is_file() {
        return Err(AutosuggestGeneratorManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: "expected a file".to_string(),
        });
    }
    let actual_bytes =
        usize::try_from(metadata.len()).map_err(|_| AutosuggestGeneratorManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: "file is too large for this platform".to_string(),
        })?;
    if actual_bytes != expected_bytes {
        return Err(AutosuggestGeneratorManifestError::Mismatch {
            field,
            expected: expected_bytes.to_string(),
            actual: actual_bytes.to_string(),
        });
    }
    let sha256 = if let Some(expected) = expected_sha256 {
        validate_sha256_format(field, expected)?;
        let actual = sha256_file_hex(&path, field)?;
        if actual != expected {
            return Err(AutosuggestGeneratorManifestError::Mismatch {
                field,
                expected: expected.to_string(),
                actual,
            });
        }
        Some(actual)
    } else {
        None
    };
    Ok(AutosuggestGeneratorAsset {
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
) -> Result<AutosuggestGeneratorAsset, AutosuggestGeneratorManifestError> {
    let path = resolve_asset_path(root, manifest_path);
    let metadata =
        fs::metadata(&path).map_err(|error| AutosuggestGeneratorManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
    if !metadata.is_dir() {
        return Err(AutosuggestGeneratorManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: "expected a directory package".to_string(),
        });
    }
    let actual_bytes = package_size(&path, field)?;
    if actual_bytes != expected_bytes {
        return Err(AutosuggestGeneratorManifestError::Mismatch {
            field,
            expected: expected_bytes.to_string(),
            actual: actual_bytes.to_string(),
        });
    }
    let sha256 = if let Some(expected) = expected_sha256 {
        validate_sha256_format(field, expected)?;
        let actual = sha256_package_hex(&path, field)?;
        if actual != expected {
            return Err(AutosuggestGeneratorManifestError::Mismatch {
                field,
                expected: expected.to_string(),
                actual,
            });
        }
        Some(actual)
    } else {
        None
    };
    Ok(AutosuggestGeneratorAsset {
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
fn package_size(
    path: &Path,
    field: &'static str,
) -> Result<usize, AutosuggestGeneratorManifestError> {
    let mut total = 0usize;
    add_package_size(path, &mut total, field)?;
    Ok(total)
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_sha256_format(
    field: &'static str,
    value: &str,
) -> Result<(), AutosuggestGeneratorManifestError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(AutosuggestGeneratorManifestError::Invalid {
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
) -> Result<String, AutosuggestGeneratorManifestError> {
    let mut file =
        fs::File::open(path).map_err(|error| AutosuggestGeneratorManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read =
            file.read(&mut buffer)
                .map_err(|error| AutosuggestGeneratorManifestError::Asset {
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
) -> Result<String, AutosuggestGeneratorManifestError> {
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
) -> Result<(), AutosuggestGeneratorManifestError> {
    for entry in fs::read_dir(path).map_err(|error| AutosuggestGeneratorManifestError::Asset {
        field,
        path: path.display().to_string(),
        reason: error.to_string(),
    })? {
        let entry = entry.map_err(|error| AutosuggestGeneratorManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        let path = entry.path();
        let metadata =
            entry
                .metadata()
                .map_err(|error| AutosuggestGeneratorManifestError::Asset {
                    field,
                    path: path.display().to_string(),
                    reason: error.to_string(),
                })?;
        if metadata.is_dir() {
            collect_package_files(root, &path, output, field)?;
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|error| AutosuggestGeneratorManifestError::Asset {
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
) -> Result<(), AutosuggestGeneratorManifestError> {
    let mut file =
        fs::File::open(path).map_err(|error| AutosuggestGeneratorManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read =
            file.read(&mut buffer)
                .map_err(|error| AutosuggestGeneratorManifestError::Asset {
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
    path: &Path,
    total: &mut usize,
    field: &'static str,
) -> Result<(), AutosuggestGeneratorManifestError> {
    for entry in fs::read_dir(path).map_err(|error| AutosuggestGeneratorManifestError::Asset {
        field,
        path: path.display().to_string(),
        reason: error.to_string(),
    })? {
        let entry = entry.map_err(|error| AutosuggestGeneratorManifestError::Asset {
            field,
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        let path = entry.path();
        let metadata =
            entry
                .metadata()
                .map_err(|error| AutosuggestGeneratorManifestError::Asset {
                    field,
                    path: path.display().to_string(),
                    reason: error.to_string(),
                })?;
        if metadata.is_dir() {
            add_package_size(&path, total, field)?;
        } else if metadata.is_file() {
            let len = usize::try_from(metadata.len()).map_err(|_| {
                AutosuggestGeneratorManifestError::Asset {
                    field,
                    path: path.display().to_string(),
                    reason: "file is too large for this platform".to_string(),
                }
            })?;
            *total =
                total
                    .checked_add(len)
                    .ok_or_else(|| AutosuggestGeneratorManifestError::Asset {
                        field,
                        path: path.display().to_string(),
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
    use crate::autosuggest::{
        AutosuggestSource, AutosuggestUnifiedCandidateKind, PersonalAutosuggestConfig,
    };
    #[cfg(not(target_arch = "wasm32"))]
    use std::fs;
    #[cfg(not(target_arch = "wasm32"))]
    use std::path::{Path, PathBuf};
    #[cfg(not(target_arch = "wasm32"))]
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_lm() -> AutosuggestLm<Vec<u8>> {
        let tokens = [
            "<pad>",
            "<bos>",
            "<unk>",
            "আমি",
            "আজ",
            "সকালে",
            "স্কুলে",
            "যাব",
            "খাই",
            "ঘুমাই",
            "পড়ি",
        ];
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

    fn fixture_manifest() -> AutosuggestGeneratorManifest {
        let lm = fixture_lm();
        AutosuggestGeneratorManifest {
            artifact: AUTOSUGGEST_GENERATOR_PACKAGE_KIND.to_string(),
            version: AUTOSUGGEST_GENERATOR_MANIFEST_VERSION,
            runtime_role: AUTOSUGGEST_GENERATOR_RUNTIME_ROLE.to_string(),
            runtime_contract: AutosuggestGeneratorRuntimeContract {
                token_id_dtype: AUTOSUGGEST_GENERATOR_TOKEN_ID_DTYPE.to_string(),
                onnx_input_dtype: AUTOSUGGEST_GENERATOR_ONNX_INPUT_DTYPE.to_string(),
                coreml_input_dtype: AUTOSUGGEST_GENERATOR_COREML_INPUT_DTYPE.to_string(),
                scores_dtype: AUTOSUGGEST_GENERATOR_SCORE_DTYPE.to_string(),
                batch_size: 1,
                context_ids_shape: [1, MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS],
                candidate_ids_shape: None,
                token_ids_shape: [1, 8],
                scores_shape: [1, 8],
                candidate_scores_shape: None,
                scored_union_policy: None,
                pad_id: AUTOSUGGEST_PAD_ID,
                bos_id: AUTOSUGGEST_BOS_ID,
                unk_id: AUTOSUGGEST_UNK_ID,
                visible_candidates: DEFAULT_AUTOSUGGEST_CANDIDATES,
            },
            ngram: AutosuggestGeneratorNgram {
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
            generator: AutosuggestGeneratorModel {
                architecture: "gru".to_string(),
                context_window: MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS,
                embedding_dim: 128,
                hidden_dim: 128,
                parameter_count: 1,
                top_k_output: 8,
                onnx: generator_file("fixture.onnx"),
                quantized_onnx: generator_file("fixture.int8.onnx"),
                coreml: coreml_package("fixture.mlpackage"),
                coreml_target: "ios17".to_string(),
                coreml_precision: "float16".to_string(),
                coreml_compute_unit: "cpu_and_ne".to_string(),
                export_kind: Some("full-vocab-topk-scorer".to_string()),
                pool_k: None,
            },
            quality: AutosuggestGeneratorQuality {
                heldout_targets: 12_000,
                eligible_targets: 10_000,
                static_pool: quality_metrics(0.1, 0.2, 0.3),
                selected_topk: quality_metrics(0.12, 0.25, 0.35),
                selected_merged_visible: Some(AutosuggestGeneratorMergedQuality {
                    top1_all_targets: 0.1,
                    top5_all_targets: 0.19,
                    top10_all_targets: 0.19,
                    mrr_all_targets: 0.14,
                    visible_candidates: DEFAULT_AUTOSUGGEST_CANDIDATES,
                    locked_static_prefix: DEFAULT_AUTOSUGGEST_GENERATOR_LOCKED_STATIC_PREFIX,
                    top5_all_target_gain_vs_static: -0.01,
                    top10_all_target_gain_vs_static: -0.11,
                    mrr_all_target_gain_vs_static: -0.02,
                }),
                selected_scored_union: None,
                static_pool_recall_all_targets: 0.4,
                neural_recall_all_targets: 0.42,
                union_recall_all_targets: 0.5,
                union_recall_all_target_gain: 0.1,
            },
            benchmark: AutosuggestGeneratorBenchmark {
                onnx_mean_us_per_item: 100.0,
                quantized_onnx_mean_us_per_item: 90.0,
                coreml_mean_us_per_item: 80.0,
            },
        }
    }

    fn generator_file(path: &str) -> AutosuggestGeneratorFile {
        AutosuggestGeneratorFile {
            path: path.to_string(),
            bytes: 1,
            sha256: sha256_bytes_hex(&[0]),
        }
    }

    fn coreml_package(path: &str) -> AutosuggestGeneratorFile {
        AutosuggestGeneratorFile {
            path: path.to_string(),
            bytes: 1,
            sha256: sha256_tree_single_file_hex("Data/com.apple.CoreML/weights/weight.bin", &[0]),
        }
    }

    fn quality_metrics(
        top1_all_targets: f64,
        top5_all_targets: f64,
        top10_all_targets: f64,
    ) -> AutosuggestGeneratorQualityMetrics {
        AutosuggestGeneratorQualityMetrics {
            top1_all_targets,
            top5_all_targets,
            top10_all_targets,
            mrr_all_targets: top1_all_targets,
        }
    }

    fn static_candidate(token_id: u32, score: i32) -> AutosuggestCandidateId {
        AutosuggestCandidateId {
            token_id,
            source: AutosuggestSource::Fourgram,
            count: 10,
            score,
        }
    }

    fn scored_union_policy() -> AutosuggestGeneratorScoredUnionPolicy {
        AutosuggestGeneratorScoredUnionPolicy {
            locked_static_prefix: 1,
            static_bonus: 1.0,
            static_rank_penalty: 0.5,
            generated_penalty: 2.0,
            overlap_bonus: 0.0,
            generated_rank_penalty: 0.0,
            static_log_count_scale: 0.0,
            static_source_bonus: 0.0,
        }
    }

    fn source_quality() -> AutosuggestGeneratorSourceQuality {
        AutosuggestGeneratorSourceQuality {
            eligible_targets: MIN_AUTOSUGGEST_GENERATOR_SOURCE_ELIGIBLE_TARGETS,
            top1_all_targets: 0.11,
            top5_all_targets: 0.24,
            top10_all_targets: 0.34,
            mrr_all_targets: 0.16,
            top5_all_target_gain_vs_static: 0.01,
            top10_all_target_gain_vs_static: 0.01,
            mrr_all_target_gain_vs_static: 0.01,
        }
    }

    fn scored_union_quality() -> AutosuggestGeneratorScoredUnionQuality {
        let policy = scored_union_policy();
        AutosuggestGeneratorScoredUnionQuality {
            top1_all_targets: 0.11,
            top5_all_targets: 0.24,
            top10_all_targets: 0.34,
            mrr_all_targets: 0.16,
            locked_static_prefix: policy.locked_static_prefix,
            static_bonus: policy.static_bonus,
            static_rank_penalty: policy.static_rank_penalty,
            generated_penalty: policy.generated_penalty,
            overlap_bonus: policy.overlap_bonus,
            generated_rank_penalty: policy.generated_rank_penalty,
            static_log_count_scale: policy.static_log_count_scale,
            static_source_bonus: policy.static_source_bonus,
            top5_all_target_gain_vs_static: 0.04,
            top10_all_target_gain_vs_static: 0.04,
            mrr_all_target_gain_vs_static: 0.02,
            accepted_for_packaging: true,
            accepted_for_packaging_all_eval_sources: true,
            eval_per_source: BTreeMap::from([
                ("epub".to_string(), source_quality()),
                ("news".to_string(), source_quality()),
                ("wiki".to_string(), source_quality()),
            ]),
        }
    }

    fn add_scored_union_gate(manifest: &mut AutosuggestGeneratorManifest) {
        manifest.runtime_contract.candidate_ids_shape = Some([1, 3]);
        manifest.runtime_contract.candidate_scores_shape = Some([1, 3]);
        manifest.runtime_contract.scored_union_policy = Some(scored_union_policy());
        manifest.generator.pool_k = Some(3);
        manifest.quality.selected_scored_union = Some(scored_union_quality());
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
    fn generator_manifest_validates_against_matching_lm() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();

        let compatibility = manifest.validate_for_lm(&lm).unwrap();

        assert_eq!(
            compatibility.context_window,
            MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS
        );
        assert_eq!(compatibility.top_k_output, 8);
        assert_eq!(compatibility.union_recall_all_target_gain, 0.1);
    }

    #[test]
    fn generator_manifest_accepts_combined_candidate_score_shapes() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        manifest.runtime_contract.candidate_ids_shape = Some([1, 6]);
        manifest.runtime_contract.candidate_scores_shape = Some([1, 6]);

        manifest.validate_for_lm(&lm).unwrap();
    }

    #[test]
    fn generator_manifest_rejects_half_declared_candidate_score_shapes() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        manifest.runtime_contract.candidate_ids_shape = Some([1, 6]);

        let error = manifest.validate_for_lm(&lm).unwrap_err();

        assert!(matches!(
            error,
            AutosuggestGeneratorManifestError::Invalid {
                field: "runtime_contract.candidate_scores_shape",
                ..
            }
        ));
    }

    #[test]
    fn generator_handoff_prepares_context_and_filters_generated_outputs() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let handoff = AutosuggestGeneratorHandoff::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut context = AutosuggestContext::new();
        lm.push_context_text(&mut context, "আমি আজ").unwrap();
        let mut context_ids = vec![99_i32; handoff.context_window];
        let mut generated = Vec::new();

        let context_len = handoff
            .coreml_context_for_context_into(&lm, context, &mut context_ids)
            .unwrap();
        handoff
            .generated_i32_candidates_into(
                &lm,
                &[7, 7, 2, 6, 5, 0, 1, 3],
                &[5.0, 4.0, 3.0, f32::NAN, 1.0, 0.0, -1.0, -2.0],
                &mut generated,
            )
            .unwrap();

        assert_eq!(context_len, 2);
        assert_eq!(&context_ids[context_ids.len() - 2..], &[3_i32, 4_i32]);
        assert_eq!(
            generated
                .iter()
                .map(|candidate| (candidate.token_id, candidate.model_rank))
                .collect::<Vec<_>>(),
            vec![(7, 0), (6, 3), (5, 4), (3, 7)]
        );
        assert_eq!(generated[1].model_score, f32::NEG_INFINITY);
    }

    #[test]
    fn generator_handoff_keeps_full_topk_candidate_pool() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let handoff = AutosuggestGeneratorHandoff::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut generated = Vec::new();

        handoff
            .generated_i32_candidates_into(
                &lm,
                &[7, 6, 5, 4, 3, 8, 9, 10],
                &[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                &mut generated,
            )
            .unwrap();

        assert_eq!(handoff.visible_candidates, 5);
        assert_eq!(generated.len(), 8);
        assert_eq!(
            generated
                .iter()
                .map(|candidate| candidate.token_id)
                .collect::<Vec<_>>(),
            vec![7, 6, 5, 4, 3, 8, 9, 10]
        );
    }

    #[test]
    fn merge_preserves_static_prefix_then_adds_neural_pool_and_static_tail() {
        let static_candidates = [
            static_candidate(5, 100),
            static_candidate(7, 80),
            static_candidate(4, 60),
        ];
        let generated = [
            AutosuggestGeneratedCandidateId {
                token_id: 7,
                model_rank: 0,
                model_score: 4.0,
            },
            AutosuggestGeneratedCandidateId {
                token_id: 6,
                model_rank: 1,
                model_score: 3.0,
            },
            AutosuggestGeneratedCandidateId {
                token_id: 3,
                model_rank: 2,
                model_score: 2.0,
            },
        ];
        let mut merged = Vec::new();

        merge_static_and_generated_candidates_into(
            &static_candidates,
            &generated,
            AutosuggestGeneratorMergeOptions {
                max_candidates: 5,
                locked_static_prefix: 1,
            },
            &mut merged,
        );

        assert_eq!(
            merged
                .iter()
                .map(|candidate| (candidate.token_id, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                (5, AutosuggestMergedCandidateSource::Static),
                (7, AutosuggestMergedCandidateSource::StaticAndGenerated),
                (6, AutosuggestMergedCandidateSource::Generated),
                (3, AutosuggestMergedCandidateSource::Generated),
                (4, AutosuggestMergedCandidateSource::Static),
            ]
        );
        assert_eq!(merged[1].static_rank, Some(1));
        assert_eq!(merged[1].model_rank, Some(0));
    }

    #[test]
    fn scored_union_uses_static_scores_and_generated_scores_without_naive_merge() {
        let static_candidates = [
            static_candidate(5, 100),
            static_candidate(7, 80),
            static_candidate(4, 60),
        ];
        let static_scores = [0.0, 2.0, 1.0];
        let generated = [
            AutosuggestGeneratedCandidateId {
                token_id: 6,
                model_rank: 0,
                model_score: 7.0,
            },
            AutosuggestGeneratedCandidateId {
                token_id: 7,
                model_rank: 1,
                model_score: 5.0,
            },
        ];
        let mut output = Vec::new();

        scored_union_static_and_generated_candidates_into(
            &static_candidates,
            &static_scores,
            &generated,
            scored_union_policy(),
            4,
            &mut output,
        )
        .unwrap();

        assert_eq!(
            output
                .iter()
                .map(|candidate| (candidate.token_id, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                (5, AutosuggestMergedCandidateSource::Static),
                (7, AutosuggestMergedCandidateSource::StaticAndGenerated),
                (6, AutosuggestMergedCandidateSource::Generated),
                (4, AutosuggestMergedCandidateSource::Static),
            ]
        );
        assert_eq!(output[1].static_model_score, Some(2.0));
        assert_eq!(output[1].generated_model_score, Some(5.0));
        assert!(output[1].union_score > output[2].union_score);
    }

    #[test]
    fn scored_union_rejects_short_static_score_buffer() {
        let static_candidates = [static_candidate(5, 100), static_candidate(7, 80)];
        let mut output = Vec::new();

        let error = scored_union_static_and_generated_candidates_into(
            &static_candidates,
            &[0.0],
            &[],
            scored_union_policy(),
            5,
            &mut output,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            AutosuggestGeneratorHandoffError::InvalidBuffer {
                field: "static_model_scores",
                expected: 2,
                actual: 1,
            }
        ));
        assert!(output.is_empty());
    }

    #[test]
    fn scored_union_applies_overlap_and_generated_rank_terms() {
        let static_candidates = [static_candidate(5, 100), static_candidate(7, 80)];
        let static_scores = [0.0, 4.0];
        let generated = [
            AutosuggestGeneratedCandidateId {
                token_id: 6,
                model_rank: 0,
                model_score: 8.0,
            },
            AutosuggestGeneratedCandidateId {
                token_id: 7,
                model_rank: 8,
                model_score: 8.0,
            },
        ];
        let mut output = Vec::new();

        scored_union_static_and_generated_candidates_into(
            &static_candidates,
            &static_scores,
            &generated,
            AutosuggestGeneratorScoredUnionPolicy {
                overlap_bonus: 2.0,
                generated_rank_penalty: 1.0,
                ..scored_union_policy()
            },
            4,
            &mut output,
        )
        .unwrap();

        let overlapped = output
            .iter()
            .find(|candidate| candidate.token_id == 7)
            .unwrap();
        let generated_only = output
            .iter()
            .find(|candidate| candidate.token_id == 6)
            .unwrap();
        assert_eq!(
            overlapped.source,
            AutosuggestMergedCandidateSource::StaticAndGenerated
        );
        assert!(overlapped.union_score < generated_only.union_score);
        assert_eq!(overlapped.model_rank, Some(8));
    }

    #[test]
    fn scored_union_can_use_static_count_and_source_terms() {
        let static_candidates = [
            AutosuggestCandidateId {
                token_id: 5,
                source: AutosuggestSource::Unigram,
                count: 1,
                score: 1,
            },
            AutosuggestCandidateId {
                token_id: 7,
                source: AutosuggestSource::Fourgram,
                count: 100,
                score: 100,
            },
        ];
        let mut output = Vec::new();

        scored_union_static_and_generated_candidates_into(
            &static_candidates,
            &[4.0, 4.0],
            &[],
            AutosuggestGeneratorScoredUnionPolicy {
                locked_static_prefix: 0,
                static_bonus: 0.0,
                static_rank_penalty: 0.0,
                generated_penalty: 0.0,
                overlap_bonus: 0.0,
                generated_rank_penalty: 0.0,
                static_log_count_scale: 0.5,
                static_source_bonus: 1.0,
            },
            2,
            &mut output,
        )
        .unwrap();

        assert_eq!(output[0].token_id, 7);
        assert!(output[0].union_score > output[1].union_score);
    }

    #[test]
    fn generator_session_scores_union_with_manifest_policy() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        add_scored_union_gate(&mut manifest);
        let mut session =
            AutosuggestGeneratorSession::from_manifest_for_lm(&manifest, &lm).unwrap();
        let static_candidates = [
            static_candidate(5, 100),
            static_candidate(7, 80),
            static_candidate(4, 60),
        ];
        let mut context = AutosuggestContext::new();
        lm.push_context_text(&mut context, "আমি আজ").unwrap();

        session.prepare_coreml_context(&lm, context).unwrap();
        session
            .accept_i32_outputs(
                &lm,
                &[6, 7, 5, 4, 3, 8, 9, 10],
                &[10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
            )
            .unwrap();
        let scored_ptr = session
            .scored_union_static_candidates(&static_candidates, &[0.0, 2.0, 1.0])
            .unwrap()
            .as_ptr();
        session
            .scored_union_static_candidates(&static_candidates, &[0.0, 2.0, 1.0])
            .unwrap();

        assert_eq!(session.scored_union_candidates().as_ptr(), scored_ptr);
        assert_eq!(session.scored_union_candidates().len(), 5);
        assert!(session.estimated_heap_bytes() <= session.heap_limit_bytes());
    }

    #[test]
    fn generator_session_reuses_fixed_buffers() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let mut session =
            AutosuggestGeneratorSession::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut context = AutosuggestContext::new();
        lm.push_context_text(&mut context, "আমি আজ").unwrap();

        session.prepare_u32_context(&lm, context).unwrap();
        let context_ptr = session.u32_context_ids().as_ptr();
        session
            .accept_u32_outputs(
                &lm,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            )
            .unwrap();
        let generated_ptr = session.generated_candidates().as_ptr();
        session.prepare_u32_context(&lm, context).unwrap();
        session
            .accept_u32_outputs(
                &lm,
                &[6, 7, 5, 4, 3, 2, 1, 0],
                &[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            )
            .unwrap();

        assert_eq!(session.u32_context_ids().as_ptr(), context_ptr);
        assert_eq!(session.generated_candidates().as_ptr(), generated_ptr);
        assert!(session.estimated_heap_bytes() <= session.heap_limit_bytes());
    }

    #[test]
    fn generator_session_merges_static_candidates_without_reallocating() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let mut session =
            AutosuggestGeneratorSession::from_manifest_for_lm(&manifest, &lm).unwrap();
        let static_candidates = [
            static_candidate(5, 100),
            static_candidate(7, 80),
            static_candidate(4, 60),
        ];
        let mut context = AutosuggestContext::new();
        lm.push_context_text(&mut context, "আমি আজ").unwrap();

        session.prepare_coreml_context(&lm, context).unwrap();
        session
            .accept_i32_outputs(
                &lm,
                &[7, 6, 5, 4, 3, 8, 9, 10],
                &[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            )
            .unwrap();
        session.merge_static_candidates(
            &static_candidates,
            AutosuggestGeneratorMergeOptions {
                max_candidates: 8,
                locked_static_prefix: 1,
            },
        );
        let merged_ptr = session.merged_candidates().as_ptr();
        session.merge_static_candidates(
            &static_candidates,
            AutosuggestGeneratorMergeOptions {
                max_candidates: 8,
                locked_static_prefix: 1,
            },
        );

        assert_eq!(session.merged_candidates().as_ptr(), merged_ptr);
        assert_eq!(session.merged_candidates().len(), 8);
        assert!(session.estimated_heap_bytes() <= session.heap_limit_bytes());
    }

    #[test]
    fn generator_session_unifies_known_tokens_and_open_vocab_text() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let mut session =
            AutosuggestGeneratorSession::from_manifest_for_lm(&manifest, &lm).unwrap();
        let static_candidates = [
            static_candidate(7, 100),
            static_candidate(6, 80),
            static_candidate(5, 60),
        ];
        let mut context = AutosuggestContext::new();
        lm.push_context_text(&mut context, "আমি আজ").unwrap();

        session.prepare_coreml_context(&lm, context).unwrap();
        session
            .accept_i32_outputs(
                &lm,
                &[6, 5, 7, 4, 8, 9, 10, 3],
                &[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            )
            .unwrap();
        session
            .accept_open_vocab_text_outputs(
                &["গিয়েছিলাম", "স্কুলে", "hello"],
                &[8.0, 4.0, 9.0],
                AutosuggestOpenVocabPolicy {
                    max_candidates: 4,
                    locked_static_prefix: 0,
                    generated_text_penalty: 0.0,
                    generated_token_penalty: 0.0,
                    ..AutosuggestOpenVocabPolicy::default()
                },
            )
            .unwrap();

        let unified = session
            .unified_candidates_for_static_candidates(
                &lm,
                &static_candidates,
                AutosuggestOpenVocabPolicy {
                    max_candidates: 6,
                    locked_static_prefix: 0,
                    generated_text_penalty: 0.0,
                    generated_token_penalty: 0.0,
                    ..AutosuggestOpenVocabPolicy::default()
                },
            )
            .unwrap();

        let open_vocab = unified
            .iter()
            .find(|candidate| candidate.text == "গিয়েছিলাম")
            .unwrap();
        assert_eq!(open_vocab.token_id, None);
        assert_eq!(
            open_vocab.kind,
            AutosuggestUnifiedCandidateKind::GeneratedText
        );
        assert!(open_vocab.has_generated_text_signal());

        let overlapped = unified
            .iter()
            .find(|candidate| candidate.text == "স্কুলে")
            .unwrap();
        assert_eq!(overlapped.token_id, Some(6));
        assert_eq!(overlapped.kind, AutosuggestUnifiedCandidateKind::Mixed);
        assert!(overlapped.has_static_signal());
        assert!(overlapped.has_generated_token_signal());
        assert!(overlapped.has_generated_text_signal());
        assert!(!unified.iter().any(|candidate| candidate.text == "hello"));
        assert!(session.estimated_heap_bytes() <= session.heap_limit_bytes());
    }

    #[test]
    fn generator_session_clears_open_vocab_outputs_on_prepare() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let mut session =
            AutosuggestGeneratorSession::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut context = AutosuggestContext::new();
        lm.push_context_text(&mut context, "আমি আজ").unwrap();

        session
            .accept_open_vocab_text_outputs(
                &["গিয়েছিলাম"],
                &[8.0],
                AutosuggestOpenVocabPolicy::default(),
            )
            .unwrap();
        assert_eq!(session.generated_text_candidates().len(), 1);

        session.prepare_coreml_context(&lm, context).unwrap();

        assert!(session.generated_text_candidates().is_empty());
        assert!(session.unified_candidates().is_empty());
    }

    #[test]
    fn generator_session_prepares_personal_aware_scored_union_inputs() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        add_scored_union_gate(&mut manifest);
        let mut generator_session =
            AutosuggestGeneratorSession::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut autosuggest_session = AutosuggestSession::with_personal_config(
            &lm,
            PersonalAutosuggestConfig {
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
        let metadata = generator_session
            .prepare_coreml_inputs_for_autosuggest_session(&mut autosuggest_session)
            .unwrap();

        assert_eq!(autosuggest_session.options().max_candidates, 2);
        assert_eq!(metadata.scorer_context_token_count, 2);
        assert_eq!(metadata.candidate_count, 3);
        assert_eq!(autosuggest_session.candidate_ids(), visible_candidate_ids);
        assert_eq!(
            &generator_session.coreml_context_ids()
                [generator_session.handoff().context_window - 2..],
            &[3, 4]
        );
        assert_eq!(generator_session.coreml_candidate_ids(), &[6, 7, 5]);
        assert_eq!(generator_session.static_candidates()[0].token_id, 6);
        assert_eq!(
            generator_session.static_candidates()[0].source,
            AutosuggestSource::Personal
        );

        generator_session
            .accept_i32_outputs(
                &lm,
                &[7, 6, 5, 4, 3, 8, 9, 10],
                &[4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            )
            .unwrap();
        generator_session
            .scored_union_with_static_scores(&[0.0, 8.0, 2.0])
            .unwrap();

        assert_eq!(generator_session.scored_union_candidates()[0].token_id, 6);
        assert_eq!(
            generator_session.scored_union_candidates()[0]
                .static_candidate
                .unwrap()
                .source,
            AutosuggestSource::Personal
        );
        assert!(generator_session.estimated_heap_bytes() <= generator_session.heap_limit_bytes());
    }

    #[test]
    fn generator_session_clears_static_pool_when_preparing_context_only() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        add_scored_union_gate(&mut manifest);
        let mut session =
            AutosuggestGeneratorSession::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut context = AutosuggestContext::new();
        lm.push_context_text(&mut context, "আমি আজ").unwrap();

        session.prepare_coreml_inputs(&lm, context).unwrap();
        assert!(!session.static_candidates().is_empty());

        session.prepare_coreml_context(&lm, context).unwrap();

        assert!(session.static_candidates().is_empty());
        assert!(session
            .coreml_candidate_ids()
            .iter()
            .all(|id| *id == AUTOSUGGEST_PAD_ID as i32));
        assert!(session.estimated_heap_bytes() <= session.heap_limit_bytes());
    }

    #[test]
    fn materializes_generated_candidates_lazily() {
        let lm = fixture_lm();
        let generated = [
            AutosuggestGeneratedCandidateId {
                token_id: 7,
                model_rank: 0,
                model_score: 1.0,
            },
            AutosuggestGeneratedCandidateId {
                token_id: 6,
                model_rank: 1,
                model_score: 0.5,
            },
        ];
        let mut materialized = Vec::new();

        materialize_generated_candidates_into(&lm, &generated, &mut materialized).unwrap();

        assert_eq!(
            materialized
                .iter()
                .map(|candidate| candidate.text)
                .collect::<Vec<_>>(),
            vec!["যাব", "স্কুলে"]
        );
    }

    #[test]
    fn materializes_merged_candidates_lazily() {
        let lm = fixture_lm();
        let merged = [
            AutosuggestMergedCandidateId {
                token_id: 7,
                source: AutosuggestMergedCandidateSource::StaticAndGenerated,
                static_rank: Some(1),
                static_candidate: Some(static_candidate(7, 80)),
                model_rank: Some(0),
                model_score: Some(4.0),
            },
            AutosuggestMergedCandidateId {
                token_id: 6,
                source: AutosuggestMergedCandidateSource::Generated,
                static_rank: None,
                static_candidate: None,
                model_rank: Some(1),
                model_score: Some(3.0),
            },
        ];
        let mut materialized = Vec::new();

        materialize_merged_candidates_into(&lm, &merged, &mut materialized).unwrap();

        assert_eq!(
            materialized
                .iter()
                .map(|candidate| (candidate.text, candidate.source))
                .collect::<Vec<_>>(),
            vec![
                ("যাব", AutosuggestMergedCandidateSource::StaticAndGenerated),
                ("স্কুলে", AutosuggestMergedCandidateSource::Generated),
            ]
        );
        assert_eq!(materialized[0].static_source, Some("fourgram"));
        assert_eq!(materialized[0].model_rank, Some(0));
    }

    #[test]
    fn generator_manifest_rejects_non_improving_quality() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        manifest.quality.union_recall_all_target_gain = 0.0;

        let error = manifest.validate_for_lm(&lm).unwrap_err();

        assert_eq!(
            error,
            AutosuggestGeneratorManifestError::Invalid {
                field: "quality.union_recall_all_target_gain",
                reason: "must be finite and positive".to_string(),
            }
        );
    }

    #[test]
    fn generator_manifest_rejects_mobile_budget_violations() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        manifest.generator.coreml.bytes = MAX_AUTOSUGGEST_GENERATOR_COREML_BYTES + 1;

        let error = manifest.validate_for_lm(&lm).unwrap_err();

        assert_eq!(
            error,
            AutosuggestGeneratorManifestError::Invalid {
                field: "generator.coreml.bytes",
                reason: format!("exceeds maximum {}", MAX_AUTOSUGGEST_GENERATOR_COREML_BYTES),
            }
        );
    }

    #[test]
    fn generator_manifest_rejects_slow_mobile_benchmarks() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        manifest.benchmark.coreml_mean_us_per_item =
            MAX_AUTOSUGGEST_GENERATOR_GRAPH_US_PER_ITEM + 1.0;

        let error = manifest.validate_for_lm(&lm).unwrap_err();

        assert_eq!(
            error,
            AutosuggestGeneratorManifestError::Invalid {
                field: "benchmark.coreml_mean_us_per_item",
                reason: format!(
                    "must be finite, positive, and at most {}",
                    MAX_AUTOSUGGEST_GENERATOR_GRAPH_US_PER_ITEM
                ),
            }
        );
    }

    #[test]
    fn generator_manifest_rejects_missing_scored_union_quality() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        manifest.runtime_contract.candidate_ids_shape = Some([1, 3]);
        manifest.runtime_contract.candidate_scores_shape = Some([1, 3]);
        manifest.runtime_contract.scored_union_policy = Some(scored_union_policy());
        manifest.generator.pool_k = Some(3);

        let error = manifest.validate_for_lm(&lm).unwrap_err();

        assert_eq!(
            error,
            AutosuggestGeneratorManifestError::Invalid {
                field: "quality.selected_scored_union",
                reason: "is required when a scored-union policy is declared".to_string(),
            }
        );
    }

    #[test]
    fn generator_manifest_rejects_source_regressing_scored_union() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        add_scored_union_gate(&mut manifest);
        manifest
            .quality
            .selected_scored_union
            .as_mut()
            .unwrap()
            .eval_per_source
            .get_mut("news")
            .unwrap()
            .mrr_all_target_gain_vs_static = -0.01;

        let error = manifest.validate_for_lm(&lm).unwrap_err();

        assert_eq!(
            error,
            AutosuggestGeneratorManifestError::Invalid {
                field: "quality.selected_scored_union.eval_per_source",
                reason: "source top-5 and MRR gains must be positive".to_string(),
            }
        );
    }

    #[test]
    fn generator_manifest_rejects_unaccepted_source_balanced_policy() {
        let lm = fixture_lm();
        let mut manifest = fixture_manifest();
        add_scored_union_gate(&mut manifest);
        manifest
            .quality
            .selected_scored_union
            .as_mut()
            .unwrap()
            .accepted_for_packaging_all_eval_sources = false;

        let error = manifest.validate_for_lm(&lm).unwrap_err();

        assert_eq!(
            error,
            AutosuggestGeneratorManifestError::Invalid {
                field: "quality.selected_scored_union.accepted_for_packaging_all_eval_sources",
                reason: "must be true for source-balanced production manifests".to_string(),
            }
        );
    }

    #[test]
    fn generated_outputs_reject_bad_shapes_and_out_of_vocab_ids() {
        let lm = fixture_lm();
        let manifest = fixture_manifest();
        let handoff = AutosuggestGeneratorHandoff::from_manifest_for_lm(&manifest, &lm).unwrap();
        let mut generated = Vec::new();

        let shape_error = handoff
            .generated_u32_candidates_into(&lm, &[7], &[1.0], &mut generated)
            .unwrap_err();
        let id_error = handoff
            .generated_i32_candidates_into(
                &lm,
                &[7, -1, 5, 4, 3, 2, 1, 0],
                &[1.0; 8],
                &mut generated,
            )
            .unwrap_err();

        assert_eq!(
            shape_error,
            AutosuggestGeneratorHandoffError::InvalidBuffer {
                field: "token_ids",
                expected: 8,
                actual: 1,
            }
        );
        assert_eq!(
            id_error,
            AutosuggestGeneratorHandoffError::InvalidGeneratedTokenId {
                index: 1,
                token_id: -1,
            }
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "obadh-generator-{label}-{}-{nanos}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_sized_file(path: &Path, bytes: usize) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, vec![0_u8; bytes]).unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_manifest_assets(root: &Path, manifest: &AutosuggestGeneratorManifest) {
        write_sized_file(&root.join(&manifest.ngram.path), manifest.ngram.bytes);
        write_sized_file(
            &root.join(&manifest.generator.onnx.path),
            manifest.generator.onnx.bytes,
        );
        write_sized_file(
            &root.join(&manifest.generator.quantized_onnx.path),
            manifest.generator.quantized_onnx.bytes,
        );
        write_sized_file(
            &root
                .join(&manifest.generator.coreml.path)
                .join("Data/com.apple.CoreML/weights/weight.bin"),
            manifest.generator.coreml.bytes,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn generator_manifest_validates_assets() {
        let manifest = fixture_manifest();
        let root = temp_root("assets");
        write_manifest_assets(&root, &manifest);

        let report = manifest.validate_asset_paths(&root).unwrap();

        assert_eq!(report.ngram.bytes, manifest.ngram.bytes);
        assert_eq!(
            report.onnx.sha256.as_deref(),
            Some(manifest.generator.onnx.sha256.as_str())
        );
        assert_eq!(
            report.coreml.sha256.as_deref(),
            Some(manifest.generator.coreml.sha256.as_str())
        );
        fs::remove_dir_all(root).unwrap();
    }
}
