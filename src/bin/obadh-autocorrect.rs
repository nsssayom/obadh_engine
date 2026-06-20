use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use obadh_engine::{
    weighted_edit_distance, AutocorrectConfig, AutocorrectEngine, CharCandidateReranker,
    CharReplacementPolicy, CorrectionCandidate, CorrectionRequest, CorrectionSource, Lexicon,
    LexiconEntry, MlpReranker, ObadhEngine, ScoredCorrectionCandidate, AUTOCORRECT_FEATURE_DIM,
};
use serde::Serialize;

#[path = "obadh_autocorrect/corpus.rs"]
mod corpus;

use corpus::{
    expand_corpus_inputs, is_bangla_lexicon_word, is_clean_roman_word_input, normalize_bangla_text,
    read_corpus_text, BanglaTokenIter, CorpusSourceStats,
};

#[derive(Debug, Parser)]
#[command(
    name = "obadh-autocorrect",
    about = "Build and evaluate compact Obadh autocorrect lexicon artifacts"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    ExtractLexicon {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1)]
        min_frequency: u32,
        #[arg(long)]
        max_entries: Option<usize>,
    },
    AuditLexicon {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = false)]
        allow_non_bangla: bool,
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
    AuditPairs {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, value_enum, default_value_t = EvalInputKind::Bangla)]
        input_kind: EvalInputKind,
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
    MergeLexicon {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1)]
        min_frequency: u32,
        #[arg(long)]
        max_entries: Option<usize>,
        #[arg(long, default_value_t = false)]
        allow_non_bangla: bool,
        #[arg(long, default_value_t = false)]
        drop_invalid: bool,
    },
    BuildLexicon {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        allow_non_bangla: bool,
    },
    InspectLexicon {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
    Eval {
        #[arg(long)]
        lexicon: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        reranker: Option<PathBuf>,
        #[arg(long)]
        char_reranker: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        char_autoreplace: bool,
        #[arg(long)]
        char_autoreplace_min_score: Option<f32>,
        #[arg(long)]
        char_autoreplace_min_margin: Option<f32>,
        #[arg(long, value_enum, default_value_t = EvalInputKind::Bangla)]
        input_kind: EvalInputKind,
        #[arg(long)]
        max_candidates: Option<usize>,
        #[arg(long)]
        max_edit_cost: Option<u16>,
        #[arg(long)]
        max_skeleton_candidates: Option<usize>,
        #[arg(long)]
        max_skeleton_edit_cost: Option<u16>,
        #[arg(long, default_value_t = false)]
        search_known_input: bool,
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
    ExportCandidates {
        #[arg(long)]
        lexicon: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        char_reranker: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = EvalInputKind::Bangla)]
        input_kind: EvalInputKind,
        #[arg(long)]
        max_candidates: Option<usize>,
        #[arg(long)]
        max_edit_cost: Option<u16>,
        #[arg(long)]
        max_skeleton_candidates: Option<usize>,
        #[arg(long)]
        max_skeleton_edit_cost: Option<u16>,
        #[arg(long, default_value_t = false)]
        search_known_input: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EvalInputKind {
    Bangla,
    Roman,
}

impl EvalInputKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bangla => "bangla",
            Self::Roman => "roman",
        }
    }
}

#[derive(Debug, Serialize)]
struct BuildReport {
    inputs: Vec<String>,
    output: String,
    input_rows: usize,
    duplicate_rows: usize,
    total_frequency: u64,
    max_frequency: u32,
    entries: usize,
    trie_nodes: usize,
    trie_edges: usize,
    skeleton_keys: usize,
    skeleton_delete_keys: usize,
    artifact_bytes: usize,
}

#[derive(Debug, Serialize)]
struct ExtractReport {
    inputs: Vec<String>,
    output: String,
    text_inputs: usize,
    html_inputs: usize,
    epub_inputs: usize,
    epub_spine_items: usize,
    epub_fallback_inputs: usize,
    epub_fallback_items: usize,
    corpus_bytes: u64,
    token_count: usize,
    unique_words: usize,
    emitted_entries: usize,
    min_frequency: u32,
    max_entries: Option<usize>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    input: String,
    rows: usize,
    accepted_rows: usize,
    unique_words: usize,
    duplicate_rows: usize,
    empty_rows: usize,
    malformed_rows: usize,
    non_bangla_rows: usize,
    invalid_frequency_rows: usize,
    total_frequency: u64,
    max_frequency: u32,
}

#[derive(Debug, Serialize)]
struct MergeReport {
    inputs: Vec<String>,
    output: String,
    rows: usize,
    accepted_rows: usize,
    duplicate_rows: usize,
    dropped_rows: usize,
    malformed_rows: usize,
    empty_rows: usize,
    non_bangla_rows: usize,
    invalid_frequency_rows: usize,
    unique_words: usize,
    emitted_entries: usize,
    total_frequency: u64,
    min_frequency: u32,
    max_entries: Option<usize>,
}

#[derive(Debug)]
struct LexiconInput {
    entries: Vec<LexiconEntry>,
    rows: usize,
    duplicate_rows: usize,
    total_frequency: u64,
    max_frequency: u32,
}

#[derive(Debug, Serialize)]
struct AuditPairsReport {
    input: String,
    input_kind: &'static str,
    rows: usize,
    accepted_rows: usize,
    unique_pairs: usize,
    duplicate_rows: usize,
    malformed_rows: usize,
    empty_source_rows: usize,
    empty_expected_rows: usize,
    non_bangla_source_rows: usize,
    non_roman_source_rows: usize,
    non_bangla_expected_rows: usize,
    identity_rows: usize,
    baseline_exact_rows: usize,
    baseline_total_edit_cost: u64,
    baseline_max_edit_cost: u16,
}

impl AuditPairsReport {
    fn accepted_rate(&self) -> f64 {
        ratio(self.accepted_rows, self.rows)
    }

    fn baseline_exact_rate(&self) -> f64 {
        ratio(self.baseline_exact_rows, self.accepted_rows)
    }

    fn baseline_mean_edit_cost(&self) -> f64 {
        if self.accepted_rows == 0 {
            0.0
        } else {
            self.baseline_total_edit_cost as f64 / self.accepted_rows as f64
        }
    }
}

#[derive(Debug, Serialize)]
struct AuditPairsReportJson<'a> {
    #[serde(flatten)]
    report: &'a AuditPairsReport,
    accepted_rate: f64,
    baseline_exact_rate: f64,
    baseline_mean_edit_cost: f64,
}

#[derive(Debug, Serialize)]
struct InspectReport {
    input: String,
    entries: usize,
    trie_nodes: usize,
    trie_edges: usize,
    skeleton_keys: usize,
    skeleton_delete_keys: usize,
    artifact_bytes: u64,
}

#[derive(Debug, Serialize)]
struct EvalReport {
    input: String,
    lexicon: String,
    input_kind: &'static str,
    total: usize,
    target_in_lexicon: usize,
    exact_baseline: usize,
    final_output_correct: usize,
    replacement_correct: usize,
    top_candidate_correct: usize,
    target_in_candidates: usize,
    suggestion_recall: usize,
    auto_replacements: usize,
    incorrect_replacements: usize,
    reciprocal_rank_sum: f64,
}

#[derive(Debug, Serialize)]
struct ExportCandidatesReport {
    input: String,
    output: String,
    lexicon: String,
    input_kind: &'static str,
    rows: usize,
    candidate_rows: usize,
    target_in_lexicon_rows: usize,
    target_present_rows: usize,
    baseline_exact_rows: usize,
    auto_replacements: usize,
}

#[derive(Debug, Serialize)]
struct CandidateExportRecord {
    source: String,
    expected: String,
    input_kind: &'static str,
    baseline: String,
    obadh_output: Option<String>,
    replacement: Option<String>,
    target_rank: Option<usize>,
    target_in_lexicon: bool,
    baseline_exact: bool,
    candidates: Vec<CandidateExportCandidate>,
}

#[derive(Debug, Serialize)]
struct CandidateExportCandidate {
    text: String,
    source: &'static str,
    edit_cost: u16,
    frequency: u32,
    score: i32,
    model_score: Option<f32>,
    features: [i16; AUTOCORRECT_FEATURE_DIM],
    label: bool,
}

impl EvalReport {
    fn baseline_accuracy(&self) -> f64 {
        ratio(self.exact_baseline, self.total)
    }

    fn final_output_accuracy(&self) -> f64 {
        ratio(self.final_output_correct, self.total)
    }

    fn replacement_accuracy(&self) -> f64 {
        ratio(self.replacement_correct, self.auto_replacements)
    }

    fn top_candidate_accuracy(&self) -> f64 {
        ratio(self.top_candidate_correct, self.total)
    }

    fn suggestion_recall(&self) -> f64 {
        ratio(self.suggestion_recall, self.total)
    }

    fn target_lexicon_coverage(&self) -> f64 {
        ratio(self.target_in_lexicon, self.total)
    }

    fn candidate_recall_given_target_in_lexicon(&self) -> f64 {
        ratio(self.target_in_candidates, self.target_in_lexicon)
    }

    fn mean_reciprocal_rank(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.reciprocal_rank_sum / self.total as f64
        }
    }
}

#[derive(Debug, Serialize)]
struct EvalReportJson<'a> {
    #[serde(flatten)]
    report: &'a EvalReport,
    baseline_accuracy: f64,
    final_output_accuracy: f64,
    replacement_accuracy: f64,
    top_candidate_accuracy: f64,
    suggestion_recall_rate: f64,
    target_lexicon_coverage: f64,
    candidate_recall_given_target_in_lexicon: f64,
    mean_reciprocal_rank: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Command::ExtractLexicon {
            input,
            output,
            min_frequency,
            max_entries,
        } => {
            let report = extract_lexicon(&input, &output, min_frequency, max_entries)?;
            print_json(&report, true)?;
        }
        Command::AuditLexicon {
            input,
            allow_non_bangla,
            pretty,
        } => {
            let report = audit_lexicon_tsv(&input, allow_non_bangla)?;
            print_json(&report, pretty)?;
        }
        Command::AuditPairs {
            input,
            input_kind,
            pretty,
        } => {
            let report = audit_pairs(&input, input_kind)?;
            print_json(
                &AuditPairsReportJson {
                    report: &report,
                    accepted_rate: report.accepted_rate(),
                    baseline_exact_rate: report.baseline_exact_rate(),
                    baseline_mean_edit_cost: report.baseline_mean_edit_cost(),
                },
                pretty,
            )?;
        }
        Command::MergeLexicon {
            input,
            output,
            min_frequency,
            max_entries,
            allow_non_bangla,
            drop_invalid,
        } => {
            let report = merge_lexicon_tsvs(
                &input,
                &output,
                min_frequency,
                max_entries,
                allow_non_bangla,
                drop_invalid,
            )?;
            print_json(&report, true)?;
        }
        Command::BuildLexicon {
            input,
            output,
            allow_non_bangla,
        } => {
            let lexicon_input = read_lexicon_tsvs(&input, allow_non_bangla)?;
            let entries = lexicon_input.entries;
            let lexicon = Lexicon::new(entries);
            let bytes = lexicon.to_compact_bytes()?;
            let stats = lexicon.stats();
            fs::write(&output, &bytes)?;
            print_json(
                &BuildReport {
                    inputs: input
                        .iter()
                        .map(|input| input.display().to_string())
                        .collect(),
                    output: output.display().to_string(),
                    input_rows: lexicon_input.rows,
                    duplicate_rows: lexicon_input.duplicate_rows,
                    total_frequency: lexicon_input.total_frequency,
                    max_frequency: lexicon_input.max_frequency,
                    entries: stats.entries,
                    trie_nodes: stats.trie_nodes,
                    trie_edges: stats.trie_edges,
                    skeleton_keys: stats.skeleton_keys,
                    skeleton_delete_keys: stats.skeleton_delete_keys,
                    artifact_bytes: bytes.len(),
                },
                true,
            )?;
        }
        Command::InspectLexicon { input, pretty } => {
            let lexicon = read_compact_lexicon(&input)?;
            let stats = lexicon.stats();
            print_json(
                &InspectReport {
                    input: input.display().to_string(),
                    entries: stats.entries,
                    trie_nodes: stats.trie_nodes,
                    trie_edges: stats.trie_edges,
                    skeleton_keys: stats.skeleton_keys,
                    skeleton_delete_keys: stats.skeleton_delete_keys,
                    artifact_bytes: fs::metadata(&input)?.len(),
                },
                pretty,
            )?;
        }
        Command::Eval {
            lexicon,
            input,
            reranker,
            char_reranker,
            char_autoreplace,
            char_autoreplace_min_score,
            char_autoreplace_min_margin,
            input_kind,
            max_candidates,
            max_edit_cost,
            max_skeleton_candidates,
            max_skeleton_edit_cost,
            search_known_input,
            pretty,
        } => {
            let lexicon_model = read_compact_lexicon(&lexicon)?;
            let reranker_model = read_reranker(reranker.as_ref())?;
            let char_reranker_model = read_char_reranker(char_reranker.as_ref())?;
            let char_replacement_policy = resolve_char_replacement_policy(
                char_reranker_model.as_ref(),
                char_autoreplace,
                char_autoreplace_min_score,
                char_autoreplace_min_margin,
            )?;
            let config = autocorrect_config(
                max_candidates,
                max_edit_cost,
                max_skeleton_candidates,
                max_skeleton_edit_cost,
                search_known_input,
            );
            let report = evaluate(
                &input,
                &lexicon,
                input_kind,
                lexicon_model,
                config,
                reranker_model.as_ref(),
                char_reranker_model.as_ref(),
                char_replacement_policy,
            )?;
            print_json(
                &EvalReportJson {
                    report: &report,
                    baseline_accuracy: report.baseline_accuracy(),
                    final_output_accuracy: report.final_output_accuracy(),
                    replacement_accuracy: report.replacement_accuracy(),
                    top_candidate_accuracy: report.top_candidate_accuracy(),
                    suggestion_recall_rate: report.suggestion_recall(),
                    target_lexicon_coverage: report.target_lexicon_coverage(),
                    candidate_recall_given_target_in_lexicon: report
                        .candidate_recall_given_target_in_lexicon(),
                    mean_reciprocal_rank: report.mean_reciprocal_rank(),
                },
                pretty,
            )?;
        }
        Command::ExportCandidates {
            lexicon,
            input,
            output,
            char_reranker,
            input_kind,
            max_candidates,
            max_edit_cost,
            max_skeleton_candidates,
            max_skeleton_edit_cost,
            search_known_input,
        } => {
            let lexicon_model = read_compact_lexicon(&lexicon)?;
            let char_reranker_model = read_char_reranker(char_reranker.as_ref())?;
            let config = autocorrect_config(
                max_candidates,
                max_edit_cost,
                max_skeleton_candidates,
                max_skeleton_edit_cost,
                search_known_input,
            );
            let report = export_candidates(
                &input,
                &output,
                &lexicon,
                input_kind,
                lexicon_model,
                config,
                char_reranker_model.as_ref(),
            )?;
            print_json(&report, true)?;
        }
    }

    Ok(())
}

fn autocorrect_config(
    max_candidates: Option<usize>,
    max_edit_cost: Option<u16>,
    max_skeleton_candidates: Option<usize>,
    max_skeleton_edit_cost: Option<u16>,
    search_known_input: bool,
) -> AutocorrectConfig {
    let mut config = AutocorrectConfig::default();
    if let Some(max_candidates) = max_candidates {
        config.max_candidates = max_candidates;
    }
    if let Some(max_edit_cost) = max_edit_cost {
        config.max_edit_cost = max_edit_cost;
        config.roman_input_max_edit_cost = max_edit_cost;
    }
    if let Some(max_skeleton_candidates) = max_skeleton_candidates {
        config.max_skeleton_candidates = max_skeleton_candidates;
    }
    if let Some(max_skeleton_edit_cost) = max_skeleton_edit_cost {
        config.max_skeleton_edit_cost = max_skeleton_edit_cost;
    }
    config.search_known_input = search_known_input;
    config
}

fn extract_lexicon(
    inputs: &[PathBuf],
    output: &PathBuf,
    min_frequency: u32,
    max_entries: Option<usize>,
) -> Result<ExtractReport, Box<dyn std::error::Error>> {
    let inputs = expand_corpus_inputs(inputs)?;
    let mut frequencies = BTreeMap::<String, u32>::new();
    let mut corpus_bytes = 0_u64;
    let mut token_count = 0_usize;
    let mut source_stats = CorpusSourceStats::default();

    for input in &inputs {
        let corpus = read_corpus_text(input)?;
        corpus_bytes = corpus_bytes.saturating_add(corpus.source_bytes);
        source_stats.add(corpus.stats);
        for token in BanglaTokenIter::new(&corpus.text) {
            token_count += 1;
            let token = normalize_bangla_text(token);
            frequencies
                .entry(token)
                .and_modify(|frequency| *frequency = frequency.saturating_add(1))
                .or_insert(1);
        }
    }

    let unique_words = frequencies.len();
    let mut entries = frequencies
        .into_iter()
        .filter(|(_, frequency)| *frequency >= min_frequency)
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    if let Some(max_entries) = max_entries {
        entries.truncate(max_entries);
    }

    let mut tsv = String::new();
    for (word, frequency) in &entries {
        tsv.push_str(word);
        tsv.push('\t');
        tsv.push_str(&frequency.to_string());
        tsv.push('\n');
    }
    fs::write(output, tsv)?;

    Ok(ExtractReport {
        inputs: inputs
            .iter()
            .map(|input| input.display().to_string())
            .collect(),
        output: output.display().to_string(),
        text_inputs: source_stats.text_inputs,
        html_inputs: source_stats.html_inputs,
        epub_inputs: source_stats.epub_inputs,
        epub_spine_items: source_stats.epub_spine_items,
        epub_fallback_inputs: source_stats.epub_fallback_inputs,
        epub_fallback_items: source_stats.epub_fallback_items,
        corpus_bytes,
        token_count,
        unique_words,
        emitted_entries: entries.len(),
        min_frequency,
        max_entries,
    })
}

fn merge_lexicon_tsvs(
    inputs: &[PathBuf],
    output: &PathBuf,
    min_frequency: u32,
    max_entries: Option<usize>,
    allow_non_bangla: bool,
    drop_invalid: bool,
) -> Result<MergeReport, Box<dyn std::error::Error>> {
    let mut words = BTreeMap::<String, u32>::new();
    let mut report = MergeReport {
        inputs: inputs
            .iter()
            .map(|input| input.display().to_string())
            .collect(),
        output: output.display().to_string(),
        rows: 0,
        accepted_rows: 0,
        duplicate_rows: 0,
        dropped_rows: 0,
        malformed_rows: 0,
        empty_rows: 0,
        non_bangla_rows: 0,
        invalid_frequency_rows: 0,
        unique_words: 0,
        emitted_entries: 0,
        total_frequency: 0,
        min_frequency,
        max_entries,
    };

    for input in inputs {
        let content = fs::read_to_string(input)?;
        for (line_index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            report.rows += 1;

            let mut columns = line.split('\t');
            let word = normalize_bangla_text(columns.next().unwrap_or_default().trim());
            let frequency_text = columns
                .next()
                .map(str::trim)
                .filter(|value| !value.is_empty());
            if word.is_empty() {
                report.empty_rows += 1;
                handle_merge_drop(&mut report, input, line_index, "empty word", drop_invalid)?;
                continue;
            }
            if columns.next().is_some() {
                report.malformed_rows += 1;
                handle_merge_drop(
                    &mut report,
                    input,
                    line_index,
                    "too many columns",
                    drop_invalid,
                )?;
                continue;
            }

            let frequency = match frequency_text {
                Some(value) => match value.parse::<u32>() {
                    Ok(frequency) => frequency,
                    Err(error) => {
                        report.invalid_frequency_rows += 1;
                        handle_merge_drop(
                            &mut report,
                            input,
                            line_index,
                            &format!("invalid frequency: {error}"),
                            drop_invalid,
                        )?;
                        continue;
                    }
                },
                None => 1,
            };

            if !allow_non_bangla && !is_bangla_lexicon_word(&word) {
                report.non_bangla_rows += 1;
                handle_merge_drop(
                    &mut report,
                    input,
                    line_index,
                    &format!("non-Bangla lexicon word: {word}"),
                    drop_invalid,
                )?;
                continue;
            }

            report.accepted_rows += 1;
            if let Some(existing) = words.get_mut(word.as_str()) {
                report.duplicate_rows += 1;
                *existing = existing.saturating_add(frequency);
            } else {
                words.insert(word, frequency);
            }
        }
    }

    report.unique_words = words.len();
    report.total_frequency = words.values().fold(0_u64, |total, frequency| {
        total.saturating_add(*frequency as u64)
    });

    let mut entries = words
        .into_iter()
        .filter(|(_, frequency)| *frequency >= min_frequency)
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    if let Some(max_entries) = max_entries {
        entries.truncate(max_entries);
    }
    report.emitted_entries = entries.len();

    let mut tsv = String::new();
    for (word, frequency) in entries {
        tsv.push_str(&word);
        tsv.push('\t');
        tsv.push_str(&frequency.to_string());
        tsv.push('\n');
    }
    fs::write(output, tsv)?;

    Ok(report)
}

fn handle_merge_drop(
    report: &mut MergeReport,
    input: &PathBuf,
    line_index: usize,
    reason: &str,
    drop_invalid: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if drop_invalid {
        report.dropped_rows += 1;
        Ok(())
    } else {
        Err(format!("{}:{}: {reason}", input.display(), line_index + 1).into())
    }
}

fn audit_lexicon_tsv(
    input: &PathBuf,
    allow_non_bangla: bool,
) -> Result<AuditReport, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(input)?;
    let mut report = AuditReport {
        input: input.display().to_string(),
        rows: 0,
        accepted_rows: 0,
        unique_words: 0,
        duplicate_rows: 0,
        empty_rows: 0,
        malformed_rows: 0,
        non_bangla_rows: 0,
        invalid_frequency_rows: 0,
        total_frequency: 0,
        max_frequency: 0,
    };
    let mut seen = BTreeMap::<String, ()>::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        report.rows += 1;

        let mut columns = line.split('\t');
        let word = normalize_bangla_text(columns.next().unwrap_or_default().trim());
        let frequency_text = columns
            .next()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if word.is_empty() {
            report.empty_rows += usize::from(word.is_empty());
            continue;
        }
        if columns.next().is_some() {
            report.malformed_rows += 1;
            continue;
        }

        let frequency = match frequency_text {
            Some(value) => match value.parse::<u32>() {
                Ok(frequency) => frequency,
                Err(_) => {
                    report.invalid_frequency_rows += 1;
                    continue;
                }
            },
            None => 1,
        };

        if !allow_non_bangla && !is_bangla_lexicon_word(&word) {
            report.non_bangla_rows += 1;
            continue;
        }
        if seen.insert(word, ()).is_some() {
            report.duplicate_rows += 1;
        }

        report.accepted_rows += 1;
        report.total_frequency = report.total_frequency.saturating_add(frequency as u64);
        report.max_frequency = report.max_frequency.max(frequency);
    }

    report.unique_words = seen.len();
    Ok(report)
}

fn read_lexicon_tsvs(
    inputs: &[PathBuf],
    allow_non_bangla: bool,
) -> Result<LexiconInput, Box<dyn std::error::Error>> {
    let mut words = BTreeMap::<String, u32>::new();
    let mut rows = 0_usize;
    let mut duplicate_rows = 0_usize;

    for input in inputs {
        for entry in read_lexicon_tsv(input, allow_non_bangla)? {
            rows += 1;
            if let Some(frequency) = words.get_mut(&entry.word) {
                duplicate_rows += 1;
                *frequency = frequency.saturating_add(entry.frequency);
            } else {
                words.insert(entry.word, entry.frequency);
            }
        }
    }

    let total_frequency = words.values().fold(0_u64, |total, frequency| {
        total.saturating_add(*frequency as u64)
    });
    let max_frequency = words.values().copied().max().unwrap_or(0);
    let entries = words
        .into_iter()
        .map(|(word, frequency)| LexiconEntry::new(word, frequency))
        .collect();

    Ok(LexiconInput {
        entries,
        rows,
        duplicate_rows,
        total_frequency,
        max_frequency,
    })
}

fn read_lexicon_tsv(
    input: &PathBuf,
    allow_non_bangla: bool,
) -> Result<Vec<LexiconEntry>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(input)?;
    let mut entries = Vec::new();

    for (line_index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut columns = line.split('\t');
        let word = normalize_bangla_text(columns.next().unwrap_or_default().trim());
        let frequency = match columns
            .next()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            Some(value) => value.parse::<u32>().map_err(|error| {
                format!(
                    "invalid frequency at {}:{}: {error}",
                    input.display(),
                    line_index + 1
                )
            })?,
            None => 1,
        };

        if columns.next().is_some() {
            return Err(
                format!("too many columns at {}:{}", input.display(), line_index + 1).into(),
            );
        }
        if word.is_empty() {
            return Err(format!("empty word at {}:{}", input.display(), line_index + 1).into());
        }
        if !allow_non_bangla && !is_bangla_lexicon_word(&word) {
            return Err(format!(
                "non-Bangla lexicon word at {}:{}: {word}",
                input.display(),
                line_index + 1
            )
            .into());
        }

        entries.push(LexiconEntry::new(word, frequency));
    }

    Ok(entries)
}

fn read_compact_lexicon(input: &PathBuf) -> Result<Lexicon, Box<dyn std::error::Error>> {
    let bytes = fs::read(input)?;
    Ok(Lexicon::from_compact_bytes(&bytes)?)
}

fn read_reranker(
    input: Option<&PathBuf>,
) -> Result<Option<MlpReranker>, Box<dyn std::error::Error>> {
    input
        .map(|input| {
            let bytes = fs::read(input)?;
            Ok(MlpReranker::from_json_bytes(&bytes)?)
        })
        .transpose()
}

fn read_char_reranker(
    input: Option<&PathBuf>,
) -> Result<Option<CharCandidateReranker>, Box<dyn std::error::Error>> {
    input
        .map(|input| {
            let bytes = fs::read(input)?;
            Ok(CharCandidateReranker::from_json_bytes(&bytes)?)
        })
        .transpose()
}

fn resolve_char_replacement_policy(
    char_reranker: Option<&CharCandidateReranker>,
    enabled: bool,
    min_score: Option<f32>,
    min_margin: Option<f32>,
) -> Result<Option<CharReplacementPolicy>, Box<dyn std::error::Error>> {
    if !enabled {
        return Ok(None);
    }

    let base = char_reranker
        .and_then(CharCandidateReranker::replacement_policy)
        .unwrap_or_else(CharReplacementPolicy::high_precision);
    let policy = CharReplacementPolicy {
        min_score: min_score.unwrap_or(base.min_score),
        min_margin: min_margin.unwrap_or(base.min_margin),
    };
    if !policy.is_valid() {
        return Err("invalid char autoreplace policy".into());
    }

    Ok(Some(policy))
}

fn evaluate(
    input: &PathBuf,
    lexicon: &PathBuf,
    input_kind: EvalInputKind,
    lexicon_model: Lexicon,
    config: AutocorrectConfig,
    reranker: Option<&MlpReranker>,
    char_reranker: Option<&CharCandidateReranker>,
    char_replacement_policy: Option<CharReplacementPolicy>,
) -> Result<EvalReport, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(input)?;
    let engine = AutocorrectEngine::with_config(lexicon_model, config);
    let obadh = ObadhEngine::new();
    let mut report = EvalReport {
        input: input.display().to_string(),
        lexicon: lexicon.display().to_string(),
        input_kind: input_kind.as_str(),
        total: 0,
        target_in_lexicon: 0,
        exact_baseline: 0,
        final_output_correct: 0,
        replacement_correct: 0,
        top_candidate_correct: 0,
        target_in_candidates: 0,
        suggestion_recall: 0,
        auto_replacements: 0,
        incorrect_replacements: 0,
        reciprocal_rank_sum: 0.0,
    };

    for (line_index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (source, expected) = parse_pair(input, line_index, line)?;
        let source = match input_kind {
            EvalInputKind::Bangla => normalize_bangla_text(source),
            EvalInputKind::Roman => source.to_string(),
        };
        let expected = normalize_bangla_text(expected);
        let request = match input_kind {
            EvalInputKind::Bangla => CorrectionRequest::new(&source),
            EvalInputKind::Roman => obadh.autocorrect_request(&source),
        };
        let baseline = request.current.clone();
        let mut decision = engine.decide(request);
        if let Some(reranker) = reranker {
            reranker.sort_candidates(&mut decision.candidates);
        }
        if let Some(char_reranker) = char_reranker {
            let ranked = char_reranker.rank_candidates(&source, &decision.candidates);
            if let Some(policy) = char_replacement_policy {
                decision.replacement =
                    char_reranker.replacement_from_ranked_candidates(&ranked, policy);
            }
            decision.candidates = ranked
                .into_iter()
                .map(|scored| scored.candidate)
                .collect::<Vec<_>>();
        }
        let final_output = decision
            .replacement
            .as_ref()
            .map(|candidate| candidate.text.as_str())
            .unwrap_or(decision.input.as_str());
        let target_in_lexicon = engine.lexicon().contains(&expected);

        report.total += 1;
        report.target_in_lexicon += usize::from(target_in_lexicon);
        report.exact_baseline += usize::from(baseline == expected);
        report.final_output_correct += usize::from(final_output == expected);
        report.replacement_correct += usize::from(
            decision
                .replacement
                .as_ref()
                .is_some_and(|candidate| candidate.text == expected),
        );
        report.top_candidate_correct += usize::from(
            decision
                .candidates
                .first()
                .is_some_and(|candidate| candidate.text == expected),
        );
        let candidate_rank = decision
            .candidates
            .iter()
            .position(|candidate| candidate.text == expected)
            .map(|index| index + 1);
        report.target_in_candidates += usize::from(candidate_rank.is_some());
        report.suggestion_recall += usize::from(baseline == expected || candidate_rank.is_some());
        report.reciprocal_rank_sum += if baseline == expected {
            1.0
        } else {
            candidate_rank.map_or(0.0, |rank| 1.0 / rank as f64)
        };
        if let Some(replacement) = &decision.replacement {
            report.auto_replacements += 1;
            report.incorrect_replacements += usize::from(replacement.text != expected);
        }
    }

    Ok(report)
}

fn export_candidates(
    input: &PathBuf,
    output: &PathBuf,
    lexicon: &PathBuf,
    input_kind: EvalInputKind,
    lexicon_model: Lexicon,
    config: AutocorrectConfig,
    char_reranker: Option<&CharCandidateReranker>,
) -> Result<ExportCandidatesReport, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(input)?;
    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);
    let engine = AutocorrectEngine::with_config(lexicon_model, config);
    let obadh = ObadhEngine::new();
    let mut report = ExportCandidatesReport {
        input: input.display().to_string(),
        output: output.display().to_string(),
        lexicon: lexicon.display().to_string(),
        input_kind: input_kind.as_str(),
        rows: 0,
        candidate_rows: 0,
        target_present_rows: 0,
        target_in_lexicon_rows: 0,
        baseline_exact_rows: 0,
        auto_replacements: 0,
    };

    for (line_index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (source, expected) = parse_pair(input, line_index, line)?;
        let source = match input_kind {
            EvalInputKind::Bangla => normalize_bangla_text(source),
            EvalInputKind::Roman => source.to_string(),
        };
        let expected = normalize_bangla_text(expected);
        let request = match input_kind {
            EvalInputKind::Bangla => CorrectionRequest::new(&source),
            EvalInputKind::Roman => obadh.autocorrect_request(&source),
        };
        let baseline = request.current.clone();
        let obadh_output = request.obadh_output.clone();
        let mut decision = engine.decide(request);
        let ranked_candidates =
            char_reranker.map(|reranker| reranker.rank_candidates(&source, &decision.candidates));
        if let Some(ranked_candidates) = &ranked_candidates {
            decision.candidates = ranked_candidates
                .iter()
                .map(|scored| scored.candidate.clone())
                .collect();
        }
        let target_in_lexicon = engine.lexicon().contains(&expected);
        let target_rank = decision
            .candidates
            .iter()
            .position(|candidate| candidate.text == expected)
            .map(|index| index + 1);
        let replacement = decision
            .replacement
            .as_ref()
            .map(|candidate| candidate.text.clone());
        let candidates = decision
            .candidates
            .iter()
            .map(|candidate| export_candidate(candidate, &expected))
            .collect::<Vec<_>>();
        let candidates = if let Some(ranked_candidates) = &ranked_candidates {
            ranked_candidates
                .iter()
                .map(|candidate| export_scored_candidate(candidate, &expected))
                .collect::<Vec<_>>()
        } else {
            candidates
        };
        let baseline_exact = baseline == expected;

        report.rows += 1;
        report.candidate_rows += candidates.len();
        report.target_in_lexicon_rows += usize::from(target_in_lexicon);
        report.target_present_rows += usize::from(target_rank.is_some());
        report.baseline_exact_rows += usize::from(baseline_exact);
        report.auto_replacements += usize::from(replacement.is_some());

        serde_json::to_writer(
            &mut writer,
            &CandidateExportRecord {
                source,
                expected,
                input_kind: input_kind.as_str(),
                baseline,
                obadh_output,
                replacement,
                target_rank,
                target_in_lexicon,
                baseline_exact,
                candidates,
            },
        )?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;

    Ok(report)
}

fn export_candidate(candidate: &CorrectionCandidate, expected: &str) -> CandidateExportCandidate {
    CandidateExportCandidate {
        text: candidate.text.clone(),
        source: correction_source_name(candidate.source),
        edit_cost: candidate.edit_cost.0,
        frequency: candidate.frequency,
        score: candidate.score,
        model_score: None,
        features: candidate.features.as_i16_array(),
        label: candidate.text == expected,
    }
}

fn export_scored_candidate(
    scored: &ScoredCorrectionCandidate,
    expected: &str,
) -> CandidateExportCandidate {
    let mut candidate = export_candidate(&scored.candidate, expected);
    candidate.model_score = Some(scored.model_score);
    candidate
}

fn correction_source_name(source: CorrectionSource) -> &'static str {
    match source {
        CorrectionSource::NoChange => "no_change",
        CorrectionSource::LexiconEdit => "lexicon_edit",
        CorrectionSource::PhoneticSkeleton => "phonetic_skeleton",
        CorrectionSource::RomanEdit => "roman_edit",
    }
}

fn audit_pairs(
    input: &PathBuf,
    input_kind: EvalInputKind,
) -> Result<AuditPairsReport, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(input)?;
    let obadh = ObadhEngine::new();
    let mut report = AuditPairsReport {
        input: input.display().to_string(),
        input_kind: input_kind.as_str(),
        rows: 0,
        accepted_rows: 0,
        unique_pairs: 0,
        duplicate_rows: 0,
        malformed_rows: 0,
        empty_source_rows: 0,
        empty_expected_rows: 0,
        non_bangla_source_rows: 0,
        non_roman_source_rows: 0,
        non_bangla_expected_rows: 0,
        identity_rows: 0,
        baseline_exact_rows: 0,
        baseline_total_edit_cost: 0,
        baseline_max_edit_cost: 0,
    };
    let mut seen = BTreeMap::<(String, String), ()>::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        report.rows += 1;

        let mut columns = line.split('\t');
        let source = columns.next().unwrap_or_default().trim();
        let expected = columns.next().unwrap_or_default().trim();
        if columns.next().is_some() {
            report.malformed_rows += 1;
            continue;
        }
        if source.is_empty() {
            report.empty_source_rows += 1;
        }
        if expected.is_empty() {
            report.empty_expected_rows += 1;
        }
        if source.is_empty() || expected.is_empty() {
            continue;
        }

        let expected = normalize_bangla_text(expected);
        if !is_bangla_lexicon_word(&expected) {
            report.non_bangla_expected_rows += 1;
            continue;
        }

        let (source, baseline) = match input_kind {
            EvalInputKind::Bangla => {
                let source = normalize_bangla_text(source);
                if !is_bangla_lexicon_word(&source) {
                    report.non_bangla_source_rows += 1;
                    continue;
                }
                let baseline = source.clone();
                (source, baseline)
            }
            EvalInputKind::Roman => {
                if !is_clean_roman_word_input(source) {
                    report.non_roman_source_rows += 1;
                    continue;
                }
                (source.to_string(), obadh.transliterate(source))
            }
        };

        report.identity_rows += usize::from(source == expected);
        report.baseline_exact_rows += usize::from(baseline == expected);
        let edit_cost = weighted_edit_distance(&baseline, &expected).0;
        report.baseline_total_edit_cost = report
            .baseline_total_edit_cost
            .saturating_add(edit_cost as u64);
        report.baseline_max_edit_cost = report.baseline_max_edit_cost.max(edit_cost);

        if seen.insert((source, expected), ()).is_some() {
            report.duplicate_rows += 1;
        }
        report.accepted_rows += 1;
    }

    report.unique_pairs = seen.len();
    Ok(report)
}

fn parse_pair<'a>(
    input: &PathBuf,
    line_index: usize,
    line: &'a str,
) -> Result<(&'a str, &'a str), Box<dyn std::error::Error>> {
    let mut columns = line.split('\t');
    let source = columns.next().unwrap_or_default().trim();
    let expected = columns.next().unwrap_or_default().trim();
    if source.is_empty() || expected.is_empty() || columns.next().is_some() {
        return Err(format!(
            "expected two tab-separated columns at {}:{}",
            input.display(),
            line_index + 1
        )
        .into());
    }
    Ok((source, expected))
}

fn print_json<T: Serialize>(value: &T, pretty: bool) -> Result<(), Box<dyn std::error::Error>> {
    if pretty {
        println!("{}", serde_json::to_string_pretty(value)?);
    } else {
        println!("{}", serde_json::to_string(value)?);
    }
    Ok(())
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}
