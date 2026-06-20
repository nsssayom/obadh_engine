use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;

use obadh_engine::{
    FstCandidate, FstLexicon, FstSuggestOptions, ObadhEngine, DEFAULT_FST_MAX_DISTANCE,
    DEFAULT_FST_PREFIX_CANDIDATES,
};
use serde::Serialize;

use super::{ensure_parent_dir, read_lexicon_tsvs};

#[cfg(not(target_arch = "wasm32"))]
type FstMapData = memmap2::Mmap;

#[cfg(target_arch = "wasm32")]
type FstMapData = Vec<u8>;

#[derive(Debug, Serialize)]
pub struct FstBuildReport {
    inputs: Vec<String>,
    output: String,
    input_rows: usize,
    duplicate_rows: usize,
    total_frequency: u64,
    max_frequency: u32,
    entries: usize,
    artifact_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct FstInspectReport {
    input: String,
    entries: usize,
    artifact_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct FstSuggestReport {
    input: String,
    obadh_output: String,
    exact_frequency: Option<u64>,
    max_distance: u32,
    max_edit_cost: Option<u16>,
    candidate_count: usize,
    returned_candidates: usize,
    truncated: bool,
    candidates: Vec<FstSuggestCandidate>,
}

#[derive(Debug, Serialize)]
struct FstSuggestCandidate {
    text: String,
    source: &'static str,
    edit_cost: u16,
    frequency: u64,
    score: i64,
}

pub fn default_max_distance() -> u32 {
    DEFAULT_FST_MAX_DISTANCE
}

pub fn default_prefix_candidates() -> usize {
    DEFAULT_FST_PREFIX_CANDIDATES
}

pub fn build_fst_lexicon_artifact(
    inputs: &[PathBuf],
    output: &PathBuf,
    allow_non_bangla: bool,
) -> Result<FstBuildReport, Box<dyn std::error::Error>> {
    ensure_parent_dir(output)?;

    let lexicon_input = read_lexicon_tsvs(inputs, allow_non_bangla)?;
    let file = File::create(output)?;
    let writer = BufWriter::new(file);
    let mut builder = fst::MapBuilder::new(writer)?;
    for entry in &lexicon_input.entries {
        builder.insert(entry.word.as_bytes(), entry.frequency as u64)?;
    }
    builder.finish()?;

    Ok(FstBuildReport {
        inputs: inputs
            .iter()
            .map(|input| input.display().to_string())
            .collect(),
        output: output.display().to_string(),
        input_rows: lexicon_input.rows,
        duplicate_rows: lexicon_input.duplicate_rows,
        total_frequency: lexicon_input.total_frequency,
        max_frequency: lexicon_input.max_frequency,
        entries: lexicon_input.entries.len(),
        artifact_bytes: fs::metadata(output)?.len(),
    })
}

pub fn inspect_fst_lexicon(
    input: &PathBuf,
) -> Result<FstInspectReport, Box<dyn std::error::Error>> {
    let map = read_fst_lexicon(input)?;
    Ok(FstInspectReport {
        input: input.display().to_string(),
        entries: map.len(),
        artifact_bytes: fs::metadata(input)?.len(),
    })
}

pub fn suggest_fst(
    input: &str,
    lexicon_model: &FstLexicon<FstMapData>,
    max_distance: u32,
    max_edit_cost: Option<u16>,
    max_candidates: usize,
    max_prefix_candidates: usize,
    response_candidates: usize,
) -> Result<FstSuggestReport, Box<dyn std::error::Error>> {
    let obadh = ObadhEngine::new();
    let obadh_output = obadh.transliterate(input);
    let result = lexicon_model.suggest(
        &obadh_output,
        FstSuggestOptions {
            max_distance,
            max_edit_cost,
            max_candidates,
            max_prefix_candidates,
            response_candidates,
        },
    )?;
    let candidates = result
        .candidates
        .into_iter()
        .map(fst_suggest_candidate)
        .collect::<Vec<_>>();

    Ok(FstSuggestReport {
        input: input.to_string(),
        obadh_output,
        exact_frequency: result.exact_frequency,
        max_distance: result.max_distance,
        max_edit_cost: result.max_edit_cost,
        candidate_count: result.candidate_count,
        returned_candidates: candidates.len(),
        truncated: result.truncated,
        candidates,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_fst_lexicon(
    input: &PathBuf,
) -> Result<FstLexicon<FstMapData>, Box<dyn std::error::Error>> {
    let file = File::open(input)?;
    // The mapping is read-only and the file handle is kept valid for the map
    // creation call. The returned mmap owns the OS mapping afterwards.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    Ok(FstLexicon::from_map(fst::Map::new(mmap)?))
}

#[cfg(target_arch = "wasm32")]
pub fn read_fst_lexicon(
    input: &PathBuf,
) -> Result<FstLexicon<FstMapData>, Box<dyn std::error::Error>> {
    Ok(FstLexicon::from_bytes(fs::read(input)?)?)
}

fn fst_suggest_candidate(candidate: FstCandidate) -> FstSuggestCandidate {
    FstSuggestCandidate {
        text: candidate.text,
        source: candidate.source.as_str(),
        edit_cost: candidate.edit_cost,
        frequency: candidate.frequency,
        score: candidate.score,
    }
}
