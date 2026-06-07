use std::collections::BTreeSet;

use obadh_engine::definitions::{
    consonant_categories, consonant_value, diacritic_rules, diacritic_value, is_vowel, vowel_value,
};
use obadh_engine::ObadhEngine;

const CONSONANT_RULES_DOC: &str = include_str!("../data/rules/consonants.md");
const README_DOC: &str = include_str!("../README.md");
const KNOWN_ISSUES_DOC: &str = include_str!("../KNOWN_ISSUES.md");
const CONJUNCT_RULES_DOC: &str = include_str!("../data/rules/conjunct.wiki");
const VOWEL_RULES_DOC: &str = include_str!("../data/rules/vowels.md");

#[test]
fn documented_basic_vowel_table_matches_runtime_rules() {
    for row in basic_vowel_table_rows(VOWEL_RULES_DOC) {
        for roman in roman_rule_keys(row.roman_input) {
            let vowel =
                vowel_value(roman).unwrap_or_else(|| panic!("documented vowel {roman:?} missing"));

            assert_eq!(
                vowel.independent, row.independent,
                "documented independent vowel for {roman:?} should match runtime"
            );
            assert_eq!(
                vowel.dependent,
                documented_dependent_vowel(row.dependent),
                "documented dependent vowel for {roman:?} should match runtime"
            );
        }
    }
}

#[test]
fn documented_lowercase_oi_ou_policy_matches_runtime_behavior() {
    assert!(!is_vowel("oi"));
    assert!(!is_vowel("ou"));

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("boi bou koi kou"), "বই বউ কই কউ");
    assert_eq!(engine.transliterate("bOI bOU kOI kOU"), "বৈ বৌ কৈ কৌ");
}

#[test]
fn deliberate_input_contract_documents_every_diacritic_rule() {
    let documented_signals = deliberate_input_contract_signal_cells(README_DOC);

    for &(roman, expected) in diacritic_rules() {
        assert_eq!(
            diacritic_value(roman),
            Some(expected),
            "diacritic {roman:?} should be directly renderable"
        );
        assert!(
            documented_signals
                .iter()
                .any(|signal| signal_cell_mentions(signal, roman)),
            "runtime diacritic signal {roman:?} is missing from README deliberate input contract"
        );
    }
}

#[test]
fn documented_consonant_table_matches_runtime_rules() {
    let mut documented_keys = BTreeSet::new();

    for row in basic_consonant_table_rows(CONSONANT_RULES_DOC) {
        for roman in roman_rule_keys(row.roman_input) {
            assert!(
                documented_keys.insert(roman),
                "documented consonant key {roman:?} should appear only once"
            );
            assert_eq!(
                consonant_value(roman),
                Some(row.bengali_output),
                "documented consonant output for {roman:?} should match runtime"
            );
        }
    }

    for category in consonant_categories() {
        for &(roman, expected) in category {
            assert_eq!(
                consonant_value(roman),
                Some(expected),
                "consonant {roman:?} should be directly renderable"
            );
            assert!(
                documented_keys.contains(roman),
                "runtime consonant signal {roman:?} is missing from data/rules/consonants.md"
            );
        }
    }
}

#[test]
fn deliberate_input_contract_documents_non_conjunct_ra_ya_zwnj_source_note() {
    let zwnj_ra_ya = "র\u{200C}\u{09CD}য";

    assert!(
        CONJUNCT_RULES_DOC.contains(zwnj_ra_ya),
        "source conjunct notes should retain the non-conjunct ra-ya ZWNJ example"
    );
    assert!(
        README_DOC.contains("`rZy`") && README_DOC.contains(zwnj_ra_ya),
        "README deliberate input contract should document the non-conjunct ra-ya ZWNJ signal"
    );
    assert!(
        !KNOWN_ISSUES_DOC.contains("Non-Conjunct Ra-Ya Form"),
        "KNOWN_ISSUES.md should not list the implemented non-conjunct ra-ya signal as open work"
    );
}

struct VowelTableRow<'a> {
    roman_input: &'a str,
    independent: &'a str,
    dependent: &'a str,
}

struct ConsonantTableRow<'a> {
    roman_input: &'a str,
    bengali_output: &'a str,
}

fn basic_consonant_table_rows(markdown: &str) -> impl Iterator<Item = ConsonantTableRow<'_>> {
    markdown
        .lines()
        .skip_while(|line| !line.starts_with("| Roman Input | Bengali Output |"))
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .filter_map(parse_consonant_table_row)
}

fn parse_consonant_table_row(line: &str) -> Option<ConsonantTableRow<'_>> {
    let mut columns = line.trim_matches('|').split('|').map(str::trim);

    Some(ConsonantTableRow {
        roman_input: columns.next()?,
        bengali_output: columns.next()?,
    })
}

fn basic_vowel_table_rows(markdown: &str) -> impl Iterator<Item = VowelTableRow<'_>> {
    markdown
        .lines()
        .skip_while(|line| !line.starts_with("| Roman Input | Independent Vowel |"))
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .filter_map(parse_vowel_table_row)
}

fn parse_vowel_table_row(line: &str) -> Option<VowelTableRow<'_>> {
    let mut columns = line.trim_matches('|').split('|').map(str::trim);

    Some(VowelTableRow {
        roman_input: columns.next()?,
        independent: columns.next()?,
        dependent: columns.next()?,
    })
}

fn roman_rule_keys(input: &str) -> impl Iterator<Item = &str> {
    input.split('/').map(str::trim)
}

fn documented_dependent_vowel(value: &str) -> Option<&str> {
    if value.starts_with('-') {
        None
    } else {
        Some(value)
    }
}

fn deliberate_input_contract_signal_cells(markdown: &str) -> Vec<&str> {
    markdown
        .lines()
        .skip_while(|line| !line.starts_with("| Roman Signal | Bengali Rule Intent |"))
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .filter_map(|line| line.trim_matches('|').split('|').next().map(str::trim))
        .collect()
}

fn signal_cell_mentions(signal: &str, roman: &str) -> bool {
    signal.contains(&format!("`{roman}`")) || signal.contains(&format!("<code>{roman}</code>"))
}
