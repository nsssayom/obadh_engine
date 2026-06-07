use obadh_engine::definitions::{is_vowel, vowel_value};
use obadh_engine::ObadhEngine;

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

struct VowelTableRow<'a> {
    roman_input: &'a str,
    independent: &'a str,
    dependent: &'a str,
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
