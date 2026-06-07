//! Definitions for the Obadh Engine
//!
//! This module contains all character definitions and mappings used in the transliteration process,
//! organized by linguistic categories.

use std::collections::HashMap;

// Declare modules
pub mod conjuncts;
pub mod consonants;
pub mod diacritics;
pub mod numerals;
pub mod symbols;
pub mod vowels;

// Re-export commonly used functions
pub use conjuncts::conjuncts;
pub use consonants::{
    consonant_categories, consonant_system, consonant_value, consonants, consonants_static,
    is_consonant,
};
pub use diacritics::{diacritic_rules, diacritic_value, diacritics, diacritics_static};
pub use numerals::numerals;
pub use symbols::{symbol_rules, symbol_value, symbols, symbols_static};
pub use vowels::{is_vowel, vowel_rules, vowel_value, vowels, vowels_static};

pub(crate) fn insert_unique<V>(
    map: &mut HashMap<&'static str, V>,
    table_name: &str,
    key: &'static str,
    value: V,
) {
    assert!(
        !map.contains_key(key),
        "duplicate {table_name} rule key: {key}"
    );
    map.insert(key, value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_definition_lookups_match_ordered_rule_tables() {
        for category in consonant_categories() {
            for &(roman, bengali) in category {
                assert_eq!(consonant_value(roman), Some(bengali));
            }
        }

        for &(roman, expected) in vowel_rules() {
            let actual = vowel_value(roman).expect("vowel table key should have direct lookup");
            assert_eq!(actual.independent, expected.independent);
            assert_eq!(actual.dependent, expected.dependent);
        }

        for &(roman, bengali) in diacritic_rules() {
            assert_eq!(diacritic_value(roman), Some(bengali));
        }

        for &(roman, bengali) in symbol_rules() {
            assert_eq!(symbol_value(roman), Some(bengali));
        }
    }

    #[test]
    fn direct_definition_lookups_reject_unknown_rule_keys() {
        assert_eq!(consonant_value("qq"), None);
        assert_eq!(vowel_value("ei").map(|vowel| vowel.independent), None);
        assert_eq!(diacritic_value("~~"), None);
        assert_eq!(symbol_value(","), None);
    }
}
