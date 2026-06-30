#[cfg(target_arch = "wasm32")]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use obadh_engine::{
    AutosuggestContext, AutosuggestLm, AutosuggestOptions, AutosuggestResult, AutosuggestSession,
    PersonalAutosuggestConfig,
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
    Bench {
        #[arg(long)]
        model: PathBuf,
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
    vocab_size: usize,
    unigram_count: usize,
    bigram_rows: usize,
    trigram_rows: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.command {
        Command::Inspect { input, pretty } => {
            let lm = read_autosuggest_lm(&input)?;
            print_json(
                &InspectReport {
                    artifact: "obadh-autosuggest-ngram",
                    version: 1,
                    bytes: lm.artifact_bytes(),
                    vocab_size: lm.vocab_size(),
                    unigram_count: lm.unigram_count(),
                    bigram_rows: lm.bigram_row_count(),
                    trigram_rows: lm.trigram_row_count(),
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
        Command::Bench {
            model,
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
            let mut personal_heap_bytes = None;
            let mut personal_snapshot_bytes = None;
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
                BenchMode::Session => {
                    let mut sessions = context
                        .iter()
                        .map(|text| autosuggest_session_from_text(&lm, text, top_k))
                        .collect::<Result<Vec<_>, _>>()?;
                    personal_heap_bytes = Some(
                        sessions
                            .iter()
                            .map(|session| session.estimated_heap_bytes())
                            .sum(),
                    );
                    personal_snapshot_bytes = Some(
                        sessions
                            .iter()
                            .map(|session| session.personal_snapshot_len())
                            .sum(),
                    );
                    let session_count = sessions.len();
                    for index in 0..iterations {
                        let session = &mut sessions[index % session_count];
                        session.suggest()?;
                        candidate_total += black_box(session.candidates().len());
                    }
                }
            }
            let elapsed = started.elapsed();
            print_json(
                &BenchReport {
                    artifact_bytes: lm.artifact_bytes(),
                    vocab_size: lm.vocab_size(),
                    contexts: context.len(),
                    mode,
                    iterations,
                    top_k,
                    load_ms: load_elapsed.as_secs_f64() * 1000.0,
                    total_ms: elapsed.as_secs_f64() * 1000.0,
                    mean_us: elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64,
                    candidates_per_query: candidate_total as f64 / iterations as f64,
                    personal_heap_bytes,
                    personal_snapshot_bytes,
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
    let file = File::open(input)?;
    // The mapping is read-only and the returned mmap owns the OS mapping.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    Ok(AutosuggestLm::from_bytes(mmap)?)
}

#[cfg(target_arch = "wasm32")]
fn read_autosuggest_lm(
    input: &PathBuf,
) -> Result<AutosuggestLm<AutosuggestMapData>, Box<dyn std::error::Error>> {
    Ok(AutosuggestLm::from_bytes(fs::read(input)?)?)
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
    personal_snapshot_bytes: Option<usize>,
}

fn print_result(
    result: AutosuggestResult<'_>,
    pretty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    print_json(&result, pretty)
}

fn print_json<T: Serialize>(value: &T, pretty: bool) -> Result<(), Box<dyn std::error::Error>> {
    if pretty {
        println!("{}", serde_json::to_string_pretty(value)?);
    } else {
        println!("{}", serde_json::to_string(value)?);
    }
    Ok(())
}
