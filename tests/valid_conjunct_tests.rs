use obadh_engine::{
    definitions::{conjuncts, consonants_static},
    ObadhEngine, PhoneticUnitType, Tokenizer,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// Tests for valid 2-component conjuncts
#[test]
fn test_two_component_conjuncts() {
    let tokenizer = Tokenizer::new();

    // Test for valid 2-component conjuncts
    let test_cases = vec![
        ("kk", 1),  // Valid 2-component conjunct
        ("st", 1),  // Valid 2-component conjunct
        ("NGj", 1), // Correct form for ঞ্জ (palatalized nya + ja)
        ("ky", 1),  // Valid 2-component conjunct
        ("kw", 1),  // Valid ba-phola conjunct
        ("bw", 1),  // Valid ba-phola conjunct over regular ba
        ("zy", 1),  // Valid ya-phola conjunct over regular ya
        ("Sw", 1),  // Valid ba-phola conjunct
    ];

    for (input, expected_units) in test_cases {
        let units = tokenizer.tokenize_word(input);

        assert_eq!(
            units.len(),
            expected_units,
            "Expected {} unit(s) for '{}', got {}",
            expected_units,
            input,
            units.len()
        );

        // There should be at least one conjunct in each result
        let has_conjunct = units.iter().any(|u| {
            matches!(
                u.unit_type,
                PhoneticUnitType::Conjunct
                    | PhoneticUnitType::ConjunctWithVowel
                    | PhoneticUnitType::ConjunctWithTerminator
            )
        });

        assert!(
            has_conjunct,
            "No conjunct found in tokenization of '{}'",
            input
        );
    }
}

/// Tests for conjuncts with vowels
#[test]
fn test_conjuncts_with_vowel() {
    let tokenizer = Tokenizer::new();

    // Conjuncts with various vowels
    let test_cases = [
        ("kkA", PhoneticUnitType::ConjunctWithVowel),
        ("ktU", PhoneticUnitType::ConjunctWithVowel),
        ("nti", PhoneticUnitType::ConjunctWithVowel),
        ("kto", PhoneticUnitType::ConjunctWithTerminator), // With terminator vowel
        ("dwa", PhoneticUnitType::ConjunctWithVowel),
        ("Shkwa", PhoneticUnitType::ConjunctWithVowel),
        ("rrwya", PhoneticUnitType::ConjunctWithVowel),
        ("k,,shwa", PhoneticUnitType::ConjunctWithVowel),
        ("r,,rwa", PhoneticUnitType::ConjunctWithVowel),
        ("kShya", PhoneticUnitType::ConjunctWithVowel),
        ("mwa", PhoneticUnitType::ConjunctWithVowel),
        ("mwra", PhoneticUnitType::ConjunctWithVowel),
        ("bwa", PhoneticUnitType::ConjunctWithVowel),
        ("zya", PhoneticUnitType::ConjunctWithVowel),
    ];

    for (input, expected_type) in &test_cases {
        let units = tokenizer.tokenize_word(input);
        // Verify we got at least one unit of the expected type
        assert!(
            units.iter().any(|unit| unit.unit_type == *expected_type),
            "Expected at least one {:?} for '{}'",
            expected_type,
            input
        );
    }
}

#[test]
fn test_source_conjunct_csv_keys_are_implemented() {
    let definitions = conjuncts();
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    for row in source_conjunct_rows(&csv) {
        let key = row.key();
        let actual = definitions.create_conjunct(&key);
        assert!(
            actual.is_some(),
            "Missing conjunct rule for CSV key '{key}' on row {}",
            row.line_number
        );

        let actual = actual.unwrap();
        assert!(
            actual == row.conjunct || allowed_csv_value_conflict(&key, actual, row.conjunct),
            "Mismatched conjunct rule for CSV key '{key}' on row {}: expected '{}', got '{actual}'",
            row.line_number,
            row.conjunct
        );
    }
}

fn allowed_csv_value_conflict(key: &str, actual: &str, expected: &str) -> bool {
    key == "rrt" && actual == "র্ত" && expected == "র্ৎ"
}

#[test]
fn test_source_conjunct_csv_rows_are_structurally_valid() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };
    let consonants = consonants_static();

    for row in source_conjunct_rows(&csv) {
        assert!(
            (2..=4).contains(&row.component_count),
            "Unexpected component count {} on row {}",
            row.component_count,
            row.line_number
        );
        assert_eq!(
            row.bn_components.len(),
            row.component_count,
            "Bengali component count mismatch on row {}",
            row.line_number
        );
        assert_eq!(
            row.roman_components.len(),
            row.component_count,
            "Roman component count mismatch on row {}",
            row.line_number
        );
        assert!(
            !row.example.is_empty(),
            "Missing Bengali example on row {}",
            row.line_number
        );

        for roman in &row.roman_components {
            assert!(
                roman.is_ascii() && !roman.contains(",,"),
                "Invalid Roman component '{roman}' on row {}",
                row.line_number
            );
            assert!(
                consonants.contains_key(roman) || is_conjunct_source_special_component(roman),
                "Unknown Roman component '{roman}' on row {}",
                row.line_number
            );
        }

        for field in std::iter::once(row.conjunct)
            .chain(row.bn_components.iter().copied())
            .chain(std::iter::once(row.example))
        {
            assert!(
                field.chars().all(is_bengali_source_char),
                "Unexpected non-Bengali source character in '{field}' on row {}",
                row.line_number
            );
        }
    }
}

#[test]
fn test_source_conjunct_csv_roman_components_match_bengali_components() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };
    let consonants = consonants_static();

    for row in source_conjunct_rows(&csv) {
        for (roman, bengali) in row.roman_components.iter().zip(&row.bn_components) {
            let expected = source_component_bengali(&row, roman, consonants).unwrap_or_else(|| {
                panic!(
                    "Unknown Roman component '{roman}' on row {}",
                    row.line_number
                )
            });

            assert_eq!(
                *bengali, expected,
                "Roman component '{roman}' maps to '{expected}', not '{}' on row {}",
                bengali, row.line_number
            );
        }
    }
}

#[test]
fn test_source_conjunct_csv_examples_contain_declared_forms() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    for row in source_conjunct_rows(&csv) {
        let normalized_conjunct = row.conjunct.replace('\u{200c}', "");
        let normalized_example = row.example.replace('\u{200c}', "");
        assert!(
            normalized_example.contains(&normalized_conjunct),
            "Example '{}' does not contain declared conjunct '{}' on row {}",
            row.example,
            row.conjunct,
            row.line_number
        );
    }
}

#[test]
fn test_source_conjunct_csv_outputs_match_components_or_declared_exceptions() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    for row in source_conjunct_rows(&csv) {
        let expected = row.bn_components.join("্");
        let without_zwnj = row.conjunct.replace('\u{200c}', "");

        assert!(
            row.conjunct == expected
                || without_zwnj == expected
                || is_contextual_khanda_ta_row(&row),
            "Conjunct output '{}' does not match components '{}' on row {}",
            row.conjunct,
            expected,
            row.line_number
        );
    }
}

#[test]
fn test_source_conjunct_csv_duplicate_keys_are_intentional() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    let mut by_key: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for row in source_conjunct_rows(&csv) {
        by_key
            .entry(row.key())
            .or_default()
            .push(row.conjunct.to_string());
    }

    let duplicates: Vec<_> = by_key
        .into_iter()
        .filter(|(_, values)| values.len() > 1)
        .collect();

    assert_eq!(
        duplicates,
        vec![("rrt".to_string(), vec!["র্ত".to_string(), "র্ৎ".to_string()])]
    );
}

#[test]
fn test_source_conjunct_wiki_inventory_is_represented_in_csv() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };
    let Some(wiki) = source_conjunct_wiki() else {
        return;
    };

    let csv_conjuncts: BTreeSet<_> = source_conjunct_rows(&csv)
        .map(|row| row.conjunct.to_string())
        .collect();

    for row in source_conjunct_wiki_rows(&wiki) {
        assert!(
            csv_conjuncts.contains(row.conjunct),
            "Wiki conjunct '{}' on line {} is missing from data/conjuncts.csv",
            row.conjunct,
            row.line_number
        );
    }
}

#[test]
fn test_source_data_conjunct_aliases_render() {
    let engine = ObadhEngine::new();

    assert_eq!(
        engine.transliterate("kShy kShya mw mwa mwr mwra bw bwa zy zY zya"),
        "ক্ষ্য ক্ষ্যা ম্ব ম্বা ম্ব্র ম্ব্রা ব্ব ব্বা য্য য্য য্যা"
    );
}

struct SourceConjunctRow<'a> {
    line_number: usize,
    conjunct: &'a str,
    component_count: usize,
    bn_components: Vec<&'a str>,
    roman_components: Vec<&'a str>,
    example: &'a str,
}

struct SourceConjunctWikiRow<'a> {
    line_number: usize,
    conjunct: &'a str,
}

impl SourceConjunctRow<'_> {
    fn key(&self) -> String {
        self.roman_components.concat()
    }
}

fn source_conjunct_wiki() -> Option<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rules/conjunct.wiki");
    match fs::read_to_string(&path) {
        Ok(wiki) => Some(wiki),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => panic!("Failed to read {}: {error}", path.display()),
    }
}

fn source_conjunct_csv() -> Option<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/conjuncts.csv");
    match fs::read_to_string(&path) {
        Ok(csv) => Some(csv),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => panic!("Failed to read {}: {error}", path.display()),
    }
}

fn source_conjunct_wiki_rows(wiki: &str) -> impl Iterator<Item = SourceConjunctWikiRow<'_>> {
    wiki.lines()
        .enumerate()
        .filter_map(|(index, line)| parse_source_conjunct_wiki_row(index + 1, line))
}

fn parse_source_conjunct_wiki_row(
    line_number: usize,
    line: &str,
) -> Option<SourceConjunctWikiRow<'_>> {
    let declaration = line.strip_prefix("# ")?;
    let (conjunct, _) = declaration.split_once('=')?;

    Some(SourceConjunctWikiRow {
        line_number,
        conjunct: conjunct.trim(),
    })
}

fn source_conjunct_rows(csv: &str) -> impl Iterator<Item = SourceConjunctRow<'_>> {
    csv.lines()
        .enumerate()
        .skip(1)
        .map(|(index, line)| parse_source_conjunct_row(index + 1, line))
}

fn parse_source_conjunct_row(line_number: usize, line: &str) -> SourceConjunctRow<'_> {
    let columns: Vec<&str> = line.split(',').collect();
    assert_eq!(
        columns.len(),
        11,
        "Malformed conjunct CSV row {line_number}: expected 11 columns, got {}",
        columns.len()
    );

    let component_count = columns[1]
        .parse::<usize>()
        .unwrap_or_else(|error| panic!("Invalid component count on row {line_number}: {error}"));

    SourceConjunctRow {
        line_number,
        conjunct: columns[0],
        component_count,
        bn_components: non_empty_components(&columns[2..6]),
        roman_components: non_empty_components(&columns[6..10]),
        example: columns[10],
    }
}

fn non_empty_components<'a>(components: &[&'a str]) -> Vec<&'a str> {
    components
        .iter()
        .copied()
        .filter(|component| !component.is_empty())
        .collect()
}

fn is_bengali_source_char(c: char) -> bool {
    matches!(
        c,
        '\u{0980}'..='\u{09ff}' | '\u{0964}' | '\u{0965}' | '\u{200c}' | '\u{200d}'
    )
}

fn is_contextual_khanda_ta_row(row: &SourceConjunctRow<'_>) -> bool {
    if !matches!(row.key().as_str(), "tk" | "tkh" | "tp" | "tl" | "ts") {
        return false;
    }

    let Some((first, tail)) = row.bn_components.split_first() else {
        return false;
    };
    if *first != "ত" {
        return false;
    }

    row.conjunct == format!("ৎ{}", tail.join("্"))
}

fn is_conjunct_source_special_component(component: &str) -> bool {
    matches!(component, "rr" | "w")
}

fn source_component_bengali<'a>(
    row: &SourceConjunctRow<'_>,
    roman: &str,
    consonants: &'a std::collections::HashMap<&'static str, &'static str>,
) -> Option<&'a str> {
    match (row.key().as_str(), row.conjunct, roman) {
        ("rrt", "র্ৎ", "t") => Some("ৎ"),
        (_, _, "rr") => Some("র"),
        (_, _, "w") => Some("ব"),
        (_, _, "y" | "Y") => Some("য"),
        _ => consonants.get(roman).copied(),
    }
}

/// Tests for invalid conjuncts (should remain as separate consonants)
#[test]
fn test_invalid_conjuncts() {
    let tokenizer = Tokenizer::new();

    // These combinations are not valid Bengali conjuncts
    let test_cases = [
        "kf",  // ক+ফ is not a standard conjunct
        "pv",  // প+ভ is not a standard conjunct
        "qw",  // q is not even a Bengali consonant
        "hk",  // হ+ক is not a standard conjunct
        "Rw",  // w is not blindly accepted after every consonant
        "kfw", // only declared full clusters absorb the ba-phola marker
    ];

    for &input in &test_cases {
        let units = tokenizer.tokenize_word(input);
        // Verify none of the units is a conjunct
        assert!(
            !units.iter().any(|u| {
                matches!(
                    u.unit_type,
                    PhoneticUnitType::Conjunct
                        | PhoneticUnitType::ConjunctWithVowel
                        | PhoneticUnitType::ConjunctWithTerminator
                )
            }),
            "Expected no conjuncts for '{}'",
            input
        );
    }
}

/// Tests for special cases: reph (রেফ), jo-phola (য-ফলা), and bo-phola (ব-ফলা)
#[test]
fn test_special_consonant_forms() {
    let tokenizer = Tokenizer::new();

    // Test special cases - groups by expected types
    let reph_cases = [
        ("rrk", PhoneticUnitType::RephOverConsonant), // Reph over consonant
        ("rrkA", PhoneticUnitType::RephOverConsonantWithVowel), // Reph with vowel
        ("rrko", PhoneticUnitType::RephOverConsonantWithTerminator), // Reph with terminator
    ];

    let phola_cases = [
        "ky",  // ক্য - ya-phola
        "ty",  // ত্য - ya-phola
        "dhy", // ধ্য - ya-phola
    ];

    // Test reph cases
    for (input, expected_type) in &reph_cases {
        let units = tokenizer.tokenize_word(input);
        // Verify we got the expected reph form
        assert!(
            units.iter().any(|unit| unit.unit_type == *expected_type),
            "Expected at least one {:?} for '{}'",
            expected_type,
            input
        );
    }

    // Test phola cases
    for &input in &phola_cases {
        let units = tokenizer.tokenize_word(input);
        // Verify we got at least one conjunct for phola forms
        assert!(
            units
                .iter()
                .any(|unit| unit.unit_type == PhoneticUnitType::Conjunct),
            "Expected at least one conjunct for '{}'",
            input
        );
    }
}
