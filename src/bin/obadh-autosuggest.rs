use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use obadh_engine::{
    rerank_candidate_ids_with_fixed_scores_into, scorer_candidate_i32s_for_candidates_into,
    scorer_candidate_ids_for_candidates_into, AutosuggestCandidateId, AutosuggestContext,
    AutosuggestGeneratorAssetReport, AutosuggestGeneratorCompatibility,
    AutosuggestGeneratorManifest, AutosuggestGeneratorMergeOptions, AutosuggestGeneratorSession,
    AutosuggestLm, AutosuggestOptions, AutosuggestRerankInputMetadata, AutosuggestRerankOptions,
    AutosuggestResult, AutosuggestScoredCandidateId, AutosuggestScorerAssetReport,
    AutosuggestScorerCompatibility, AutosuggestScorerManifest, AutosuggestScorerSession,
    AutosuggestSession, PersonalAutosuggestConfig, DEFAULT_AUTOSUGGEST_CANDIDATES,
    DEFAULT_AUTOSUGGEST_RERANK_RANK_PENALTY, MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS,
};
use serde::Serialize;

#[cfg(not(target_arch = "wasm32"))]
type AutosuggestMapData = memmap2::Mmap;
#[cfg(target_arch = "wasm32")]
type AutosuggestMapData = Vec<u8>;

#[derive(Debug, Parser)]
#[command(
    name = "obadh-autosuggest",
    about = "Inspect and query compact Obadh autosuggest artifacts"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Inspect {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
    ValidateScorer {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value = ".")]
        asset_root: PathBuf,
        #[arg(long, default_value_t = false)]
        skip_assets: bool,
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
    ValidateGenerator {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value = ".")]
        asset_root: PathBuf,
        #[arg(long, default_value_t = false)]
        skip_assets: bool,
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
    Suggest {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        context: String,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
    RerankInput {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        context: String,
        #[arg(long, default_value_t = MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS)]
        context_window: usize,
        #[arg(long, default_value_t = 64)]
        candidate_pool: usize,
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
    RerankApply {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        context: String,
        #[arg(long)]
        scores: String,
        #[arg(long, default_value_t = MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS)]
        context_window: usize,
        #[arg(long, default_value_t = 64)]
        candidate_pool: usize,
        #[arg(long, default_value_t = DEFAULT_AUTOSUGGEST_CANDIDATES)]
        top_k: usize,
        #[arg(long, default_value_t = 1)]
        locked_prefix: usize,
        #[arg(long, default_value_t = DEFAULT_AUTOSUGGEST_RERANK_RANK_PENALTY)]
        rank_penalty: f32,
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        scorer_manifest: Option<PathBuf>,
        #[arg(long)]
        generator_manifest: Option<PathBuf>,
        #[arg(long, required = true)]
        context: Vec<String>,
        #[arg(long, value_enum, default_value_t = BenchMode::Text)]
        mode: BenchMode,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long, default_value_t = 100_000)]
        iterations: usize,
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
}

#[derive(Debug, Serialize)]
struct InspectReport {
    artifact: &'static str,
    version: u32,
    bytes: usize,
    artifact_fingerprint: String,
    vocab_size: usize,
    vocab_fingerprint: u32,
    unigram_count: usize,
    bigram_rows: usize,
    trigram_rows: usize,
    fourgram_rows: usize,
    candidate_record_len: usize,
}

#[derive(Debug, Serialize)]
struct ValidateScorerReport {
    compatibility: AutosuggestScorerCompatibility,
    assets: Option<AutosuggestScorerAssetReport>,
}

#[derive(Debug, Serialize)]
struct ValidateGeneratorReport {
    compatibility: AutosuggestGeneratorCompatibility,
    assets: Option<AutosuggestGeneratorAssetReport>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.command {
        Command::Inspect { input, pretty } => {
            let lm = read_autosuggest_lm(&input)?;
            let model = lm.model_info();
            print_json(
                &InspectReport {
                    artifact: model.artifact_kind,
                    version: model.version,
                    bytes: model.artifact_bytes,
                    artifact_fingerprint: artifact_fingerprint_hex(lm.artifact_fingerprint()),
                    vocab_size: model.vocab_size,
                    vocab_fingerprint: model.vocab_fingerprint,
                    unigram_count: model.unigram_count,
                    bigram_rows: model.bigram_rows,
                    trigram_rows: model.trigram_rows,
                    fourgram_rows: model.fourgram_rows,
                    candidate_record_len: model.candidate_record_len,
                },
                pretty,
            )?;
        }
        Command::ValidateScorer {
            model,
            manifest,
            asset_root,
            skip_assets,
            pretty,
        } => {
            let lm = read_autosuggest_lm(&model)?;
            let manifest =
                AutosuggestScorerManifest::from_json_str(&fs::read_to_string(manifest)?)?;
            let compatibility = manifest.validate_for_lm(&lm)?;
            let assets = if skip_assets {
                None
            } else {
                Some(manifest.validate_asset_paths(asset_root)?)
            };
            print_json(
                &ValidateScorerReport {
                    compatibility,
                    assets,
                },
                pretty,
            )?;
        }
        Command::ValidateGenerator {
            model,
            manifest,
            asset_root,
            skip_assets,
            pretty,
        } => {
            let lm = read_autosuggest_lm(&model)?;
            let manifest = read_generator_manifest(&manifest)?;
            let compatibility = manifest.validate_for_lm(&lm)?;
            let assets = if skip_assets {
                None
            } else {
                Some(manifest.validate_asset_paths(asset_root)?)
            };
            print_json(
                &ValidateGeneratorReport {
                    compatibility,
                    assets,
                },
                pretty,
            )?;
        }
        Command::Suggest {
            model,
            context,
            top_k,
            pretty,
        } => {
            let lm = read_autosuggest_lm(&model)?;
            let result = lm.suggest_for_text(
                &context,
                AutosuggestOptions {
                    max_candidates: top_k,
                },
            )?;
            print_result(result, pretty)?;
        }
        Command::RerankInput {
            model,
            context,
            context_window,
            candidate_pool,
            pretty,
        } => {
            let lm = read_autosuggest_lm(&model)?;
            let mut autosuggest_context = AutosuggestContext::new();
            lm.push_context_text(&mut autosuggest_context, &context)?;
            let mut scorer_context_ids = vec![0; context_window];
            let mut candidates = Vec::with_capacity(candidate_pool.max(1));
            let metadata = lm.rerank_input_for_context_into(
                autosuggest_context,
                AutosuggestOptions {
                    max_candidates: candidate_pool,
                },
                &mut scorer_context_ids,
                &mut candidates,
            )?;
            let mut scorer_candidate_ids = vec![0; candidate_pool];
            scorer_candidate_ids_for_candidates_into(&candidates, &mut scorer_candidate_ids);
            print_rerank_input(
                &lm,
                &context,
                metadata,
                scorer_context_ids,
                scorer_candidate_ids,
                candidates,
                pretty,
            )?;
        }
        Command::RerankApply {
            model,
            context,
            scores,
            context_window,
            candidate_pool,
            top_k,
            locked_prefix,
            rank_penalty,
            pretty,
        } => {
            let lm = read_autosuggest_lm(&model)?;
            let model_scores = parse_score_list(&scores)?;
            let mut autosuggest_context = AutosuggestContext::new();
            lm.push_context_text(&mut autosuggest_context, &context)?;
            let mut scorer_context_ids = vec![0; context_window];
            let mut candidates = Vec::with_capacity(candidate_pool.max(1));
            let metadata = lm.rerank_input_for_context_into(
                autosuggest_context,
                AutosuggestOptions {
                    max_candidates: candidate_pool,
                },
                &mut scorer_context_ids,
                &mut candidates,
            )?;
            let mut scorer_candidate_ids = vec![0; candidate_pool];
            scorer_candidate_ids_for_candidates_into(&candidates, &mut scorer_candidate_ids);
            let mut ranked = Vec::with_capacity(top_k.max(1));
            rerank_candidate_ids_with_fixed_scores_into(
                &candidates,
                &model_scores,
                AutosuggestRerankOptions {
                    max_candidates: top_k,
                    locked_prefix,
                    rank_penalty,
                },
                &mut ranked,
            )?;
            print_rerank_apply(
                &lm,
                &context,
                metadata,
                scorer_context_ids,
                scorer_candidate_ids,
                candidates.len(),
                ranked,
                pretty,
            )?;
        }
        Command::Bench {
            model,
            scorer_manifest,
            generator_manifest,
            context,
            mode,
            top_k,
            iterations,
            pretty,
        } => {
            let load_started = Instant::now();
            let lm = read_autosuggest_lm(&model)?;
            let load_elapsed = load_started.elapsed();
            let iterations = iterations.max(1);
            let started = Instant::now();
            let mut candidate_total = 0_usize;
            let mut candidates = Vec::with_capacity(top_k.max(1));
            let mut candidate_ids = Vec::with_capacity(top_k.max(1));
            let mut model_scores = Vec::<f32>::with_capacity(top_k.max(1));
            let mut static_model_scores = Vec::<f32>::with_capacity(top_k.max(1));
            let mut reranked = Vec::<AutosuggestScoredCandidateId>::with_capacity(top_k.max(1));
            let mut scorer_context_ids = vec![0; MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS];
            let mut scorer_candidate_ids = vec![0; top_k.max(1)];
            let mut coreml_context_ids = vec![0; MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS];
            let mut coreml_candidate_ids = vec![0; top_k.max(1)];
            let mut generated_token_ids = Vec::<i32>::new();
            let mut personal_heap_bytes = None;
            let mut personal_heap_limit_bytes = None;
            let mut personal_session_heap_bytes = None;
            let mut personal_session_heap_limit_bytes = None;
            let mut personal_snapshot_bytes = None;
            let mut personal_snapshot_limit_bytes = None;
            let mut personal_session_snapshot_bytes = None;
            let mut personal_session_snapshot_limit_bytes = None;
            let mut scorer_heap_bytes = None;
            let mut scorer_heap_limit_bytes = None;
            let mut scorer_candidate_pool = None;
            let mut scorer_visible_candidates = None;
            let mut generator_heap_bytes = None;
            let mut generator_heap_limit_bytes = None;
            let mut generator_top_k_output = None;
            let mut generator_candidate_pool = None;
            let mut generator_visible_candidates = None;
            match mode {
                BenchMode::Text => {
                    for index in 0..iterations {
                        let context = &context[index % context.len()];
                        lm.suggest_for_text_into(
                            black_box(context),
                            AutosuggestOptions {
                                max_candidates: top_k,
                            },
                            &mut candidates,
                        )?;
                        candidate_total += black_box(candidates.len());
                    }
                }
                BenchMode::Context => {
                    let contexts = context
                        .iter()
                        .map(|text| autosuggest_context_from_text(&lm, text))
                        .collect::<Result<Vec<_>, _>>()?;
                    for index in 0..iterations {
                        let context = contexts[index % contexts.len()];
                        lm.suggest_for_context_into(
                            black_box(context),
                            AutosuggestOptions {
                                max_candidates: top_k,
                            },
                            &mut candidates,
                        )?;
                        candidate_total += black_box(candidates.len());
                    }
                }
                BenchMode::RerankInput => {
                    let contexts = context
                        .iter()
                        .map(|text| autosuggest_context_from_text(&lm, text))
                        .collect::<Result<Vec<_>, _>>()?;
                    for index in 0..iterations {
                        let context = contexts[index % contexts.len()];
                        lm.rerank_input_for_context_into(
                            black_box(context),
                            AutosuggestOptions {
                                max_candidates: top_k,
                            },
                            black_box(&mut scorer_context_ids),
                            &mut candidate_ids,
                        )?;
                        let copied = scorer_candidate_ids_for_candidates_into(
                            &candidate_ids,
                            black_box(&mut scorer_candidate_ids),
                        );
                        candidate_total += black_box(candidate_ids.len());
                        black_box(copied);
                    }
                }
                BenchMode::CoremlInput => {
                    let contexts = context
                        .iter()
                        .map(|text| autosuggest_context_from_text(&lm, text))
                        .collect::<Result<Vec<_>, _>>()?;
                    for index in 0..iterations {
                        let context = contexts[index % contexts.len()];
                        let context_len = lm.scorer_context_i32s_for_context_into(
                            black_box(context),
                            black_box(&mut coreml_context_ids),
                        )?;
                        lm.suggest_ids_for_context_into(
                            black_box(context),
                            AutosuggestOptions {
                                max_candidates: top_k,
                            },
                            &mut candidate_ids,
                        )?;
                        let copied = scorer_candidate_i32s_for_candidates_into(
                            &candidate_ids,
                            black_box(&mut coreml_candidate_ids),
                        )?;
                        candidate_total += black_box(candidate_ids.len());
                        black_box((context_len, copied));
                    }
                }
                BenchMode::RerankApply => {
                    let contexts = context
                        .iter()
                        .map(|text| autosuggest_context_from_text(&lm, text))
                        .collect::<Result<Vec<_>, _>>()?;
                    for index in 0..iterations {
                        let context = contexts[index % contexts.len()];
                        lm.rerank_input_for_context_into(
                            black_box(context),
                            AutosuggestOptions {
                                max_candidates: top_k,
                            },
                            black_box(&mut scorer_context_ids),
                            &mut candidate_ids,
                        )?;
                        scorer_candidate_ids_for_candidates_into(
                            &candidate_ids,
                            black_box(&mut scorer_candidate_ids),
                        );
                        fill_synthetic_model_scores(&candidate_ids, &mut model_scores);
                        rerank_candidate_ids_with_fixed_scores_into(
                            &candidate_ids,
                            &model_scores,
                            AutosuggestRerankOptions {
                                max_candidates: DEFAULT_AUTOSUGGEST_CANDIDATES,
                                locked_prefix: 1,
                                rank_penalty: DEFAULT_AUTOSUGGEST_RERANK_RANK_PENALTY,
                            },
                            &mut reranked,
                        )?;
                        candidate_total += black_box(reranked.len());
                    }
                }
                BenchMode::Session => {
                    let mut sessions = context
                        .iter()
                        .map(|text| autosuggest_session_from_text(&lm, text, top_k))
                        .collect::<Result<Vec<_>, _>>()?;
                    let stats = personal_bench_stats(&sessions);
                    personal_heap_bytes = Some(stats.heap_bytes);
                    personal_heap_limit_bytes = Some(stats.heap_limit_bytes);
                    personal_session_heap_bytes = Some(stats.session_heap_bytes);
                    personal_session_heap_limit_bytes = Some(stats.session_heap_limit_bytes);
                    personal_snapshot_bytes = Some(stats.snapshot_bytes);
                    personal_snapshot_limit_bytes = Some(stats.snapshot_limit_bytes);
                    personal_session_snapshot_bytes = Some(stats.session_snapshot_bytes);
                    personal_session_snapshot_limit_bytes =
                        Some(stats.session_snapshot_limit_bytes);
                    let session_count = sessions.len();
                    for index in 0..iterations {
                        let session = &mut sessions[index % session_count];
                        session.suggest()?;
                        candidate_total += black_box(session.candidates().len());
                    }
                }
                BenchMode::ScorerSession => {
                    let manifest_path = scorer_manifest.ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "--scorer-manifest is required when --mode scorer-session",
                        )
                    })?;
                    let manifest = read_scorer_manifest(&manifest_path)?;
                    let mut session =
                        AutosuggestScorerSession::from_manifest_for_lm(&manifest, &lm)?;
                    let contexts = context
                        .iter()
                        .map(|text| autosuggest_context_from_text(&lm, text))
                        .collect::<Result<Vec<_>, _>>()?;
                    if model_scores.capacity() < session.handoff().candidate_pool {
                        model_scores.reserve_exact(session.handoff().candidate_pool);
                    }
                    scorer_heap_bytes = Some(session.estimated_heap_bytes());
                    scorer_heap_limit_bytes = Some(session.heap_limit_bytes());
                    scorer_candidate_pool = Some(session.handoff().candidate_pool);
                    scorer_visible_candidates = Some(session.handoff().visible_candidates);
                    for index in 0..iterations {
                        let context = contexts[index % contexts.len()];
                        session.prepare_coreml_inputs(&lm, black_box(context))?;
                        fill_fixed_synthetic_model_scores(
                            session.candidates(),
                            session.handoff().candidate_pool,
                            &mut model_scores,
                        );
                        let ranked = session.rerank_with_scores(black_box(&model_scores))?;
                        candidate_total += black_box(ranked.len());
                    }
                }
                BenchMode::ScorerSessionPersonal => {
                    let manifest_path = scorer_manifest.ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "--scorer-manifest is required when --mode scorer-session-personal",
                        )
                    })?;
                    let manifest = read_scorer_manifest(&manifest_path)?;
                    let mut scorer_session =
                        AutosuggestScorerSession::from_manifest_for_lm(&manifest, &lm)?;
                    let mut sessions = context
                        .iter()
                        .map(|text| autosuggest_session_from_text(&lm, text, top_k))
                        .collect::<Result<Vec<_>, _>>()?;
                    if model_scores.capacity() < scorer_session.handoff().candidate_pool {
                        model_scores.reserve_exact(scorer_session.handoff().candidate_pool);
                    }
                    for session in &mut sessions {
                        scorer_session.prepare_coreml_inputs_for_autosuggest_session(session)?;
                    }
                    let stats = personal_bench_stats(&sessions);
                    personal_heap_bytes = Some(stats.heap_bytes);
                    personal_heap_limit_bytes = Some(stats.heap_limit_bytes);
                    personal_session_heap_bytes = Some(stats.session_heap_bytes);
                    personal_session_heap_limit_bytes = Some(stats.session_heap_limit_bytes);
                    personal_snapshot_bytes = Some(stats.snapshot_bytes);
                    personal_snapshot_limit_bytes = Some(stats.snapshot_limit_bytes);
                    personal_session_snapshot_bytes = Some(stats.session_snapshot_bytes);
                    personal_session_snapshot_limit_bytes =
                        Some(stats.session_snapshot_limit_bytes);
                    scorer_heap_bytes = Some(scorer_session.estimated_heap_bytes());
                    scorer_heap_limit_bytes = Some(scorer_session.heap_limit_bytes());
                    scorer_candidate_pool = Some(scorer_session.handoff().candidate_pool);
                    scorer_visible_candidates = Some(scorer_session.handoff().visible_candidates);
                    let session_count = sessions.len();
                    for index in 0..iterations {
                        let session = &mut sessions[index % session_count];
                        scorer_session.prepare_coreml_inputs_for_autosuggest_session(session)?;
                        fill_fixed_synthetic_model_scores(
                            scorer_session.candidates(),
                            scorer_session.handoff().candidate_pool,
                            &mut model_scores,
                        );
                        let ranked = scorer_session.rerank_with_scores(black_box(&model_scores))?;
                        candidate_total += black_box(ranked.len());
                    }
                }
                BenchMode::GeneratorSession => {
                    let manifest_path = generator_manifest.ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "--generator-manifest is required when --mode generator-session",
                        )
                    })?;
                    let manifest = read_generator_manifest(&manifest_path)?;
                    let mut generator_session =
                        AutosuggestGeneratorSession::from_manifest_for_lm(&manifest, &lm)?;
                    let contexts = context
                        .iter()
                        .map(|text| autosuggest_context_from_text(&lm, text))
                        .collect::<Result<Vec<_>, _>>()?;
                    let top_k_output = generator_session.handoff().top_k_output;
                    generated_token_ids.reserve_exact(top_k_output);
                    model_scores.reserve_exact(top_k_output);
                    fill_synthetic_generated_outputs(
                        lm.vocab_size(),
                        top_k_output,
                        &mut generated_token_ids,
                        &mut model_scores,
                    );
                    generator_heap_bytes = Some(generator_session.estimated_heap_bytes());
                    generator_heap_limit_bytes = Some(generator_session.heap_limit_bytes());
                    generator_top_k_output = Some(top_k_output);
                    generator_candidate_pool = generator_session.handoff().candidate_pool;
                    generator_visible_candidates =
                        Some(generator_session.handoff().visible_candidates);
                    for index in 0..iterations {
                        let context = contexts[index % contexts.len()];
                        generator_session.prepare_coreml_context(&lm, black_box(context))?;
                        let generated = generator_session.accept_i32_outputs(
                            &lm,
                            black_box(&generated_token_ids),
                            black_box(&model_scores),
                        )?;
                        candidate_total += black_box(generated.len());
                    }
                }
                BenchMode::GeneratorMergeSession => {
                    let manifest_path = generator_manifest.ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "--generator-manifest is required when --mode generator-merge-session",
                        )
                    })?;
                    let manifest = read_generator_manifest(&manifest_path)?;
                    let mut generator_session =
                        AutosuggestGeneratorSession::from_manifest_for_lm(&manifest, &lm)?;
                    let contexts = context
                        .iter()
                        .map(|text| autosuggest_context_from_text(&lm, text))
                        .collect::<Result<Vec<_>, _>>()?;
                    let top_k_output = generator_session.handoff().top_k_output;
                    generated_token_ids.reserve_exact(top_k_output);
                    model_scores.reserve_exact(top_k_output);
                    candidate_ids.reserve_exact(top_k.max(1));
                    fill_synthetic_generated_outputs(
                        lm.vocab_size(),
                        top_k_output,
                        &mut generated_token_ids,
                        &mut model_scores,
                    );
                    generator_heap_bytes = Some(generator_session.estimated_heap_bytes());
                    generator_heap_limit_bytes = Some(generator_session.heap_limit_bytes());
                    generator_top_k_output = Some(top_k_output);
                    generator_candidate_pool = generator_session.handoff().candidate_pool;
                    generator_visible_candidates =
                        Some(generator_session.handoff().visible_candidates);
                    for index in 0..iterations {
                        let context = contexts[index % contexts.len()];
                        generator_session.prepare_coreml_context(&lm, black_box(context))?;
                        lm.suggest_ids_for_context_into(
                            black_box(context),
                            AutosuggestOptions {
                                max_candidates: top_k,
                            },
                            &mut candidate_ids,
                        )?;
                        generator_session.accept_i32_outputs(
                            &lm,
                            black_box(&generated_token_ids),
                            black_box(&model_scores),
                        )?;
                        let merged = generator_session.merge_static_candidates(
                            &candidate_ids,
                            AutosuggestGeneratorMergeOptions {
                                max_candidates: top_k,
                                locked_static_prefix: 1,
                            },
                        );
                        candidate_total += black_box(merged.len());
                    }
                }
                BenchMode::GeneratorScoredUnionSession => {
                    let manifest_path = generator_manifest.ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "--generator-manifest is required when --mode generator-scored-union-session",
                        )
                    })?;
                    let manifest = read_generator_manifest(&manifest_path)?;
                    let mut generator_session =
                        AutosuggestGeneratorSession::from_manifest_for_lm(&manifest, &lm)?;
                    let contexts = context
                        .iter()
                        .map(|text| autosuggest_context_from_text(&lm, text))
                        .collect::<Result<Vec<_>, _>>()?;
                    let top_k_output = generator_session.handoff().top_k_output;
                    let candidate_pool = generator_session
                        .handoff()
                        .candidate_pool
                        .unwrap_or_else(|| top_k.max(1));
                    generated_token_ids.reserve_exact(top_k_output);
                    model_scores.reserve_exact(top_k_output);
                    candidate_ids.reserve_exact(candidate_pool);
                    static_model_scores.reserve_exact(candidate_pool);
                    fill_synthetic_generated_outputs(
                        lm.vocab_size(),
                        top_k_output,
                        &mut generated_token_ids,
                        &mut model_scores,
                    );
                    generator_heap_bytes = Some(generator_session.estimated_heap_bytes());
                    generator_heap_limit_bytes = Some(generator_session.heap_limit_bytes());
                    generator_top_k_output = Some(top_k_output);
                    generator_candidate_pool = Some(candidate_pool);
                    generator_visible_candidates =
                        Some(generator_session.handoff().visible_candidates);
                    for index in 0..iterations {
                        let context = contexts[index % contexts.len()];
                        generator_session.prepare_coreml_context(&lm, black_box(context))?;
                        lm.suggest_ids_for_context_into(
                            black_box(context),
                            AutosuggestOptions {
                                max_candidates: candidate_pool,
                            },
                            &mut candidate_ids,
                        )?;
                        fill_fixed_synthetic_model_scores(
                            &candidate_ids,
                            candidate_pool,
                            &mut static_model_scores,
                        );
                        generator_session.accept_i32_outputs(
                            &lm,
                            black_box(&generated_token_ids),
                            black_box(&model_scores),
                        )?;
                        let scored = generator_session.scored_union_static_candidates(
                            &candidate_ids,
                            black_box(&static_model_scores),
                        )?;
                        candidate_total += black_box(scored.len());
                    }
                }
                BenchMode::GeneratorScoredUnionSessionPersonal => {
                    let manifest_path = generator_manifest.ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "--generator-manifest is required when --mode generator-scored-union-session-personal",
                        )
                    })?;
                    let manifest = read_generator_manifest(&manifest_path)?;
                    let mut generator_session =
                        AutosuggestGeneratorSession::from_manifest_for_lm(&manifest, &lm)?;
                    let mut sessions = context
                        .iter()
                        .map(|text| autosuggest_session_from_text(&lm, text, top_k))
                        .collect::<Result<Vec<_>, _>>()?;
                    let top_k_output = generator_session.handoff().top_k_output;
                    let candidate_pool = generator_session
                        .handoff()
                        .candidate_pool
                        .unwrap_or_else(|| top_k.max(1));
                    generated_token_ids.reserve_exact(top_k_output);
                    model_scores.reserve_exact(top_k_output);
                    static_model_scores.reserve_exact(candidate_pool);
                    fill_synthetic_generated_outputs(
                        lm.vocab_size(),
                        top_k_output,
                        &mut generated_token_ids,
                        &mut model_scores,
                    );
                    for session in &mut sessions {
                        generator_session.prepare_coreml_inputs_for_autosuggest_session(session)?;
                    }
                    let stats = personal_bench_stats(&sessions);
                    personal_heap_bytes = Some(stats.heap_bytes);
                    personal_heap_limit_bytes = Some(stats.heap_limit_bytes);
                    personal_session_heap_bytes = Some(stats.session_heap_bytes);
                    personal_session_heap_limit_bytes = Some(stats.session_heap_limit_bytes);
                    personal_snapshot_bytes = Some(stats.snapshot_bytes);
                    personal_snapshot_limit_bytes = Some(stats.snapshot_limit_bytes);
                    personal_session_snapshot_bytes = Some(stats.session_snapshot_bytes);
                    personal_session_snapshot_limit_bytes =
                        Some(stats.session_snapshot_limit_bytes);
                    generator_heap_bytes = Some(generator_session.estimated_heap_bytes());
                    generator_heap_limit_bytes = Some(generator_session.heap_limit_bytes());
                    generator_top_k_output = Some(top_k_output);
                    generator_candidate_pool = Some(candidate_pool);
                    generator_visible_candidates =
                        Some(generator_session.handoff().visible_candidates);
                    let session_count = sessions.len();
                    for index in 0..iterations {
                        let session = &mut sessions[index % session_count];
                        generator_session.prepare_coreml_inputs_for_autosuggest_session(session)?;
                        fill_fixed_synthetic_model_scores(
                            generator_session.static_candidates(),
                            candidate_pool,
                            &mut static_model_scores,
                        );
                        generator_session.accept_i32_outputs(
                            &lm,
                            black_box(&generated_token_ids),
                            black_box(&model_scores),
                        )?;
                        let scored = generator_session
                            .scored_union_with_static_scores(black_box(&static_model_scores))?;
                        candidate_total += black_box(scored.len());
                    }
                }
            }
            let elapsed = started.elapsed();
            let model_info = lm.model_info();
            print_json(
                &BenchReport {
                    artifact_bytes: model_info.artifact_bytes,
                    vocab_size: model_info.vocab_size,
                    contexts: context.len(),
                    mode,
                    iterations,
                    top_k,
                    load_ms: load_elapsed.as_secs_f64() * 1000.0,
                    total_ms: elapsed.as_secs_f64() * 1000.0,
                    mean_us: elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64,
                    candidates_per_query: candidate_total as f64 / iterations as f64,
                    personal_heap_bytes,
                    personal_heap_limit_bytes,
                    personal_session_heap_bytes,
                    personal_session_heap_limit_bytes,
                    personal_snapshot_bytes,
                    personal_snapshot_limit_bytes,
                    personal_session_snapshot_bytes,
                    personal_session_snapshot_limit_bytes,
                    scorer_heap_bytes,
                    scorer_heap_limit_bytes,
                    scorer_candidate_pool,
                    scorer_visible_candidates,
                    generator_heap_bytes,
                    generator_heap_limit_bytes,
                    generator_top_k_output,
                    generator_candidate_pool,
                    generator_visible_candidates,
                },
                pretty,
            )?;
        }
    }

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn read_autosuggest_lm(
    input: &PathBuf,
) -> Result<AutosuggestLm<AutosuggestMapData>, Box<dyn std::error::Error>> {
    Ok(AutosuggestLm::from_path(input)?)
}

#[cfg(target_arch = "wasm32")]
fn read_autosuggest_lm(
    input: &PathBuf,
) -> Result<AutosuggestLm<AutosuggestMapData>, Box<dyn std::error::Error>> {
    Ok(AutosuggestLm::from_bytes(fs::read(input)?)?)
}

fn read_scorer_manifest(
    input: &PathBuf,
) -> Result<AutosuggestScorerManifest, Box<dyn std::error::Error>> {
    Ok(AutosuggestScorerManifest::from_json_str(
        &fs::read_to_string(input)?,
    )?)
}

fn read_generator_manifest(
    input: &PathBuf,
) -> Result<AutosuggestGeneratorManifest, Box<dyn std::error::Error>> {
    Ok(AutosuggestGeneratorManifest::from_json_str(
        &fs::read_to_string(input)?,
    )?)
}

fn artifact_fingerprint_hex(fingerprint: u64) -> String {
    format!("{fingerprint:016x}")
}

fn autosuggest_context_from_text<D: AsRef<[u8]>>(
    lm: &AutosuggestLm<D>,
    text: &str,
) -> Result<AutosuggestContext, Box<dyn std::error::Error>> {
    let mut context = AutosuggestContext::new();
    lm.push_context_text(&mut context, text)?;
    Ok(context)
}

fn autosuggest_session_from_text<'lm, D: AsRef<[u8]>>(
    lm: &'lm AutosuggestLm<D>,
    text: &str,
    top_k: usize,
) -> Result<AutosuggestSession<'lm, D>, Box<dyn std::error::Error>> {
    let options = AutosuggestOptions {
        max_candidates: top_k,
    };
    let mut session =
        AutosuggestSession::with_personal_config(lm, PersonalAutosuggestConfig::default(), options);
    let token_ids = autosuggest_token_ids_from_text(lm, text)?;

    if let Some(first_id) = token_ids.first().copied() {
        for _ in 0..PersonalAutosuggestConfig::default().min_count {
            session.clear_context();
            for token_id in token_ids.iter().copied().chain(std::iter::once(first_id)) {
                session.commit_token_id(Some(token_id), false)?;
            }
        }
    }

    session.clear_context();
    for token_id in token_ids {
        session.commit_token_id(Some(token_id), false)?;
    }
    Ok(session)
}

fn personal_bench_stats<D: AsRef<[u8]>>(
    sessions: &[AutosuggestSession<'_, D>],
) -> PersonalBenchStats {
    let mut stats = PersonalBenchStats::default();
    for session in sessions {
        let heap_bytes = session.estimated_heap_bytes();
        let heap_limit_bytes = session.heap_limit_bytes();
        let snapshot_bytes = session.personal_snapshot_len();
        let snapshot_limit_bytes = session.personal_snapshot_limit_bytes();

        stats.heap_bytes = stats.heap_bytes.saturating_add(heap_bytes);
        stats.heap_limit_bytes = stats.heap_limit_bytes.saturating_add(heap_limit_bytes);
        stats.snapshot_bytes = stats.snapshot_bytes.saturating_add(snapshot_bytes);
        stats.snapshot_limit_bytes = stats
            .snapshot_limit_bytes
            .saturating_add(snapshot_limit_bytes);
        stats.session_heap_bytes = stats.session_heap_bytes.max(heap_bytes);
        stats.session_heap_limit_bytes = stats.session_heap_limit_bytes.max(heap_limit_bytes);
        stats.session_snapshot_bytes = stats.session_snapshot_bytes.max(snapshot_bytes);
        stats.session_snapshot_limit_bytes =
            stats.session_snapshot_limit_bytes.max(snapshot_limit_bytes);
    }
    stats
}

fn autosuggest_token_ids_from_text<D: AsRef<[u8]>>(
    lm: &AutosuggestLm<D>,
    text: &str,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut ids = Vec::new();
    for token in text.split_whitespace() {
        if let Some(token_id) = lm.token_id(token)? {
            ids.push(token_id);
        }
    }
    Ok(ids)
}

#[derive(Debug, Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum BenchMode {
    Text,
    Context,
    RerankInput,
    CoremlInput,
    RerankApply,
    ScorerSession,
    ScorerSessionPersonal,
    GeneratorSession,
    GeneratorMergeSession,
    GeneratorScoredUnionSession,
    GeneratorScoredUnionSessionPersonal,
    Session,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    artifact_bytes: usize,
    vocab_size: usize,
    contexts: usize,
    mode: BenchMode,
    iterations: usize,
    top_k: usize,
    load_ms: f64,
    total_ms: f64,
    mean_us: f64,
    candidates_per_query: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    personal_heap_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    personal_heap_limit_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    personal_session_heap_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    personal_session_heap_limit_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    personal_snapshot_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    personal_snapshot_limit_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    personal_session_snapshot_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    personal_session_snapshot_limit_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scorer_heap_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scorer_heap_limit_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scorer_candidate_pool: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scorer_visible_candidates: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generator_heap_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generator_heap_limit_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generator_top_k_output: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generator_candidate_pool: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generator_visible_candidates: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
struct PersonalBenchStats {
    heap_bytes: usize,
    heap_limit_bytes: usize,
    session_heap_bytes: usize,
    session_heap_limit_bytes: usize,
    snapshot_bytes: usize,
    snapshot_limit_bytes: usize,
    session_snapshot_bytes: usize,
    session_snapshot_limit_bytes: usize,
}

#[derive(Debug, Serialize)]
struct RerankInputReport {
    context: String,
    context_ids: Vec<u32>,
    candidate_ids: Vec<u32>,
    context_token_count: usize,
    matched_context_token_count: usize,
    scorer_context_token_count: usize,
    candidate_count: usize,
    candidates: Vec<RerankCandidateReport>,
}

#[derive(Debug, Serialize)]
struct RerankCandidateReport {
    text: String,
    token_id: u32,
    source: &'static str,
    count: u32,
    score: i32,
}

#[derive(Debug, Serialize)]
struct RerankApplyReport {
    context: String,
    context_ids: Vec<u32>,
    candidate_ids: Vec<u32>,
    context_token_count: usize,
    matched_context_token_count: usize,
    scorer_context_token_count: usize,
    input_candidate_count: usize,
    candidates: Vec<RerankAppliedCandidateReport>,
}

#[derive(Debug, Serialize)]
struct RerankAppliedCandidateReport {
    text: String,
    token_id: u32,
    source: &'static str,
    count: u32,
    score: i32,
    original_rank: usize,
    model_score: f32,
    rerank_score: f32,
}

fn print_result(
    result: AutosuggestResult<'_>,
    pretty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    print_json(&result, pretty)
}

fn print_rerank_input<D: AsRef<[u8]>>(
    lm: &AutosuggestLm<D>,
    context: &str,
    metadata: AutosuggestRerankInputMetadata,
    context_ids: Vec<u32>,
    candidate_ids: Vec<u32>,
    candidates: Vec<AutosuggestCandidateId>,
    pretty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let candidates = candidates
        .into_iter()
        .map(|candidate| {
            let materialized = lm.materialize_candidate(candidate)?;
            Ok(RerankCandidateReport {
                text: materialized.text.to_string(),
                token_id: candidate.token_id,
                source: candidate.source.as_str(),
                count: candidate.count,
                score: candidate.score,
            })
        })
        .collect::<Result<Vec<_>, obadh_engine::AutosuggestArtifactError>>()?;

    print_json(
        &RerankInputReport {
            context: context.to_string(),
            context_ids,
            candidate_ids,
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            scorer_context_token_count: metadata.scorer_context_token_count,
            candidate_count: metadata.candidate_count,
            candidates,
        },
        pretty,
    )
}

fn print_rerank_apply<D: AsRef<[u8]>>(
    lm: &AutosuggestLm<D>,
    context: &str,
    metadata: AutosuggestRerankInputMetadata,
    context_ids: Vec<u32>,
    candidate_ids: Vec<u32>,
    input_candidate_count: usize,
    candidates: Vec<AutosuggestScoredCandidateId>,
    pretty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let candidates = candidates
        .into_iter()
        .map(|candidate| {
            let materialized = lm.materialize_candidate(candidate.candidate_id())?;
            Ok(RerankAppliedCandidateReport {
                text: materialized.text.to_string(),
                token_id: materialized.token_id,
                source: materialized.source.as_str(),
                count: materialized.count,
                score: materialized.score,
                original_rank: candidate.original_rank,
                model_score: candidate.model_score,
                rerank_score: candidate.rerank_score,
            })
        })
        .collect::<Result<Vec<_>, obadh_engine::AutosuggestArtifactError>>()?;

    print_json(
        &RerankApplyReport {
            context: context.to_string(),
            context_ids,
            candidate_ids,
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            scorer_context_token_count: metadata.scorer_context_token_count,
            input_candidate_count,
            candidates,
        },
        pretty,
    )
}

fn print_json<T: Serialize>(value: &T, pretty: bool) -> Result<(), Box<dyn std::error::Error>> {
    if pretty {
        println!("{}", serde_json::to_string_pretty(value)?);
    } else {
        println!("{}", serde_json::to_string(value)?);
    }
    Ok(())
}

fn parse_score_list(raw: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut scores = Vec::new();
    for part in raw.split([',', ' ', '\n', '\t']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        scores.push(part.parse::<f32>()?);
    }
    Ok(scores)
}

fn fill_synthetic_model_scores(candidates: &[AutosuggestCandidateId], output: &mut Vec<f32>) {
    output.clear();
    output.extend(candidates.iter().map(|candidate| {
        let token_signal = candidate.token_id.wrapping_mul(2_654_435_761) >> 20;
        token_signal as f32 * 0.000_001
    }));
}

fn fill_fixed_synthetic_model_scores(
    candidates: &[AutosuggestCandidateId],
    candidate_pool: usize,
    output: &mut Vec<f32>,
) {
    output.clear();
    output.extend(candidates.iter().take(candidate_pool).map(|candidate| {
        let token_signal = candidate.token_id.wrapping_mul(2_654_435_761) >> 20;
        token_signal as f32 * 0.000_001
    }));
    output.resize(candidate_pool, f32::NEG_INFINITY);
}

fn fill_synthetic_generated_outputs(
    vocab_size: usize,
    top_k_output: usize,
    token_ids: &mut Vec<i32>,
    scores: &mut Vec<f32>,
) {
    token_ids.clear();
    scores.clear();
    let word_vocab = vocab_size.saturating_sub(3).max(1);
    for index in 0..top_k_output {
        let token_id = 3 + (index % word_vocab);
        token_ids.push(i32::try_from(token_id).unwrap_or(i32::MAX));
        scores.push((top_k_output - index) as f32);
    }
}
