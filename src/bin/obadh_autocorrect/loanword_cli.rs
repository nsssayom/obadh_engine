use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use obadh_engine::{is_loanword_key, LoanwordEntry, LoanwordLexicon};
use serde::Serialize;

use super::{ensure_parent_dir, is_bangla_lexicon_word, normalize_bangla_text};

const DEFAULT_LOANWORD_FREQUENCY: u32 = 16;

#[derive(Debug, Serialize)]
pub struct LoanwordBuildReport {
    input: String,
    output: String,
    rows: usize,
    accepted_rows: usize,
    duplicate_rows: usize,
    malformed_rows: usize,
    empty_rows: usize,
    non_bangla_rows: usize,
    invalid_english_rows: usize,
    unique_english_keys: usize,
    entries: usize,
    frequency: u32,
    artifact_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct LoanwordBanglaExportReport {
    input: String,
    output: String,
    rows: usize,
    accepted_rows: usize,
    duplicate_rows: usize,
    malformed_rows: usize,
    empty_rows: usize,
    non_bangla_rows: usize,
    invalid_english_rows: usize,
    unique_words: usize,
    frequency: u32,
}

#[derive(Debug, Serialize)]
pub struct LoanwordInspectReport {
    input: String,
    entries: usize,
    english_keys: usize,
    artifact_bytes: u64,
}

#[derive(Debug)]
struct LoanwordPairs {
    entries: Vec<LoanwordEntry>,
    rows: usize,
    accepted_rows: usize,
    duplicate_rows: usize,
    malformed_rows: usize,
    empty_rows: usize,
    non_bangla_rows: usize,
    invalid_english_rows: usize,
    unique_english_keys: usize,
}

pub fn default_frequency() -> u32 {
    DEFAULT_LOANWORD_FREQUENCY
}

pub fn build_loanword_lexicon_artifact(
    input: &PathBuf,
    output: &PathBuf,
    frequency: u32,
) -> Result<LoanwordBuildReport, Box<dyn std::error::Error>> {
    ensure_parent_dir(output)?;
    let pairs = read_loanword_pairs(input, frequency)?;
    let lexicon = LoanwordLexicon::from_entries(pairs.entries.clone())?;
    let entries = lexicon.len();
    fs::write(output, obadh_engine::build_loanword_bytes(pairs.entries)?)?;

    Ok(LoanwordBuildReport {
        input: input.display().to_string(),
        output: output.display().to_string(),
        rows: pairs.rows,
        accepted_rows: pairs.accepted_rows,
        duplicate_rows: pairs.duplicate_rows,
        malformed_rows: pairs.malformed_rows,
        empty_rows: pairs.empty_rows,
        non_bangla_rows: pairs.non_bangla_rows,
        invalid_english_rows: pairs.invalid_english_rows,
        unique_english_keys: pairs.unique_english_keys,
        entries,
        frequency,
        artifact_bytes: fs::metadata(output)?.len(),
    })
}

pub fn export_loanword_bangla_lexicon(
    input: &PathBuf,
    output: &PathBuf,
    frequency: u32,
) -> Result<LoanwordBanglaExportReport, Box<dyn std::error::Error>> {
    ensure_parent_dir(output)?;
    let pairs = read_loanword_pairs(input, frequency)?;
    let mut words = BTreeMap::<String, u32>::new();
    for entry in &pairs.entries {
        words
            .entry(entry.bangla.clone())
            .and_modify(|existing| *existing = (*existing).max(entry.frequency))
            .or_insert(entry.frequency);
    }

    let mut tsv = String::new();
    for (word, frequency) in &words {
        tsv.push_str(word);
        tsv.push('\t');
        tsv.push_str(&frequency.to_string());
        tsv.push('\n');
    }
    fs::write(output, tsv)?;

    Ok(LoanwordBanglaExportReport {
        input: input.display().to_string(),
        output: output.display().to_string(),
        rows: pairs.rows,
        accepted_rows: pairs.accepted_rows,
        duplicate_rows: pairs.duplicate_rows,
        malformed_rows: pairs.malformed_rows,
        empty_rows: pairs.empty_rows,
        non_bangla_rows: pairs.non_bangla_rows,
        invalid_english_rows: pairs.invalid_english_rows,
        unique_words: words.len(),
        frequency,
    })
}

pub fn inspect_loanword_lexicon(
    input: &PathBuf,
) -> Result<LoanwordInspectReport, Box<dyn std::error::Error>> {
    let lexicon = read_loanword_lexicon(input)?;
    Ok(LoanwordInspectReport {
        input: input.display().to_string(),
        entries: lexicon.len(),
        english_keys: lexicon.english_key_count(),
        artifact_bytes: fs::metadata(input)?.len(),
    })
}

pub fn read_loanword_lexicon(
    input: &PathBuf,
) -> Result<LoanwordLexicon<Vec<u8>>, Box<dyn std::error::Error>> {
    Ok(LoanwordLexicon::from_bytes(fs::read(input)?)?)
}

fn read_loanword_pairs(
    input: &PathBuf,
    frequency: u32,
) -> Result<LoanwordPairs, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(input)?;
    let mut entries = Vec::new();
    let mut seen_pairs = BTreeSet::<(String, String)>::new();
    let mut english_keys = BTreeSet::<String>::new();
    let mut report = LoanwordPairs {
        entries: Vec::new(),
        rows: 0,
        accepted_rows: 0,
        duplicate_rows: 0,
        malformed_rows: 0,
        empty_rows: 0,
        non_bangla_rows: 0,
        invalid_english_rows: 0,
        unique_english_keys: 0,
    };

    for (line_index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line_index == 0 && line == "bangla\tenglish" {
            continue;
        }
        report.rows += 1;

        let mut columns = line.split('\t');
        let bangla = normalize_bangla_text(columns.next().unwrap_or_default().trim());
        let english = columns.next().unwrap_or_default().trim().to_string();
        if columns.next().is_some() {
            report.malformed_rows += 1;
            continue;
        }
        if bangla.is_empty() || english.is_empty() {
            report.empty_rows += 1;
            continue;
        }
        if !is_bangla_lexicon_word(&bangla) {
            report.non_bangla_rows += 1;
            continue;
        }
        if !is_loanword_key(&english) {
            report.invalid_english_rows += 1;
            continue;
        }
        if !seen_pairs.insert((bangla.clone(), english.clone())) {
            report.duplicate_rows += 1;
            continue;
        }

        english_keys.insert(english.clone());
        report.accepted_rows += 1;
        entries.push(LoanwordEntry::new(english, bangla, frequency));
    }

    report.unique_english_keys = english_keys.len();
    report.entries = entries;
    Ok(report)
}
