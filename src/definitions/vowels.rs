//! Definitions for Bengali vowels
//!
//! This file contains the mappings for Bengali vowels in both their
//! independent forms and dependent forms (vowel signs/kars).

use super::insert_unique;
use std::collections::HashMap;
use std::sync::OnceLock;

/// A complete Bengali vowel with both independent and dependent forms
#[derive(Clone, Copy)]
pub struct BengaliVowel {
    /// Independent form (used at word beginning or standalone)
    pub independent: &'static str,
    /// Dependent form (used after consonants as modifiers/kars)
    pub dependent: Option<&'static str>,
}

impl BengaliVowel {
    /// Create a new Bengali vowel with both forms
    pub const fn new(independent: &'static str, dependent: Option<&'static str>) -> Self {
        Self {
            independent,
            dependent,
        }
    }
}

/// A Roman input rule and its complete Bengali vowel output.
pub type VowelRule = (&'static str, BengaliVowel);

/// Maximum Roman byte length of any vowel rule key.
///
/// Vowel keys are ASCII rule signals, so this also bounds the suffix scan used
/// when splitting already-tokenized consonant-vowel units.
pub(crate) const MAX_VOWEL_RULE_BYTES: usize = 3;

const VOWEL_RULES: &[VowelRule] = &[
    // Inherent vowel (no visible kar when used with consonants)
    ("o", BengaliVowel::new("অ", None)),
    // The remaining vowels have both independent and dependent forms
    ("A", BengaliVowel::new("আ", Some("া"))),
    ("AY", BengaliVowel::new("অ্যা", Some("্যা"))),
    ("ai", BengaliVowel::new("আই", Some("াই"))),
    ("au", BengaliVowel::new("আউ", Some("াউ"))),
    ("ae", BengaliVowel::new("আএ", Some("াএ"))),
    ("ao", BengaliVowel::new("আও", Some("াও"))),
    ("ia", BengaliVowel::new("ইয়া", Some("িয়া"))),
    ("io", BengaliVowel::new("ইও", Some("িও"))),
    ("eo", BengaliVowel::new("এও", Some("েও"))),
    ("a", BengaliVowel::new("আ", Some("া"))),
    ("aY", BengaliVowel::new("অ্যা", Some("্যা"))),
    ("i", BengaliVowel::new("ই", Some("ি"))),
    ("I", BengaliVowel::new("ঈ", Some("ী"))),
    ("u", BengaliVowel::new("উ", Some("ু"))),
    ("U", BengaliVowel::new("ঊ", Some("ূ"))),
    ("e", BengaliVowel::new("এ", Some("ে"))),
    ("E", BengaliVowel::new("এ", Some("ে"))),
    ("OI", BengaliVowel::new("ঐ", Some("ৈ"))),
    ("O", BengaliVowel::new("ও", Some("ো"))),
    ("OU", BengaliVowel::new("ঔ", Some("ৌ"))),
    ("rri", BengaliVowel::new("ঋ", Some("ৃ"))),
];

fn build_vowels() -> HashMap<&'static str, BengaliVowel> {
    let mut map = HashMap::with_capacity(VOWEL_RULES.len());

    for &(roman, vowel) in VOWEL_RULES {
        insert_unique(&mut map, "vowel", roman, vowel);
    }

    map
}

/// Returns the ordered static vowel rule table.
pub const fn vowel_rules() -> &'static [VowelRule] {
    VOWEL_RULES
}

/// Look up a Bengali vowel from a Roman rule key without constructing or hashing a map.
pub fn vowel_value(roman: &str) -> Option<BengaliVowel> {
    match roman {
        "o" => Some(BengaliVowel::new("অ", None)),
        "A" => Some(BengaliVowel::new("আ", Some("া"))),
        "AY" => Some(BengaliVowel::new("অ্যা", Some("্যা"))),
        "ai" => Some(BengaliVowel::new("আই", Some("াই"))),
        "au" => Some(BengaliVowel::new("আউ", Some("াউ"))),
        "ae" => Some(BengaliVowel::new("আএ", Some("াএ"))),
        "ao" => Some(BengaliVowel::new("আও", Some("াও"))),
        "ia" => Some(BengaliVowel::new("ইয়া", Some("িয়া"))),
        "io" => Some(BengaliVowel::new("ইও", Some("িও"))),
        "eo" => Some(BengaliVowel::new("এও", Some("েও"))),
        "a" => Some(BengaliVowel::new("আ", Some("া"))),
        "aY" => Some(BengaliVowel::new("অ্যা", Some("্যা"))),
        "i" => Some(BengaliVowel::new("ই", Some("ি"))),
        "I" => Some(BengaliVowel::new("ঈ", Some("ী"))),
        "u" => Some(BengaliVowel::new("উ", Some("ু"))),
        "U" => Some(BengaliVowel::new("ঊ", Some("ূ"))),
        "e" | "E" => Some(BengaliVowel::new("এ", Some("ে"))),
        "OI" => Some(BengaliVowel::new("ঐ", Some("ৈ"))),
        "O" => Some(BengaliVowel::new("ও", Some("ো"))),
        "OU" => Some(BengaliVowel::new("ঔ", Some("ৌ"))),
        "rri" => Some(BengaliVowel::new("ঋ", Some("ৃ"))),
        _ => None,
    }
}

/// Return whether a Roman rule key is a known vowel.
pub fn is_vowel(roman: &str) -> bool {
    vowel_value(roman).is_some()
}

/// Returns a shared map of Bengali vowels with their independent and dependent forms.
pub fn vowels_static() -> &'static HashMap<&'static str, BengaliVowel> {
    static INSTANCE: OnceLock<HashMap<&'static str, BengaliVowel>> = OnceLock::new();
    INSTANCE.get_or_init(build_vowels)
}

/// Returns a map of Bengali vowels with their independent and dependent forms
pub fn vowels() -> HashMap<&'static str, BengaliVowel> {
    vowels_static().clone()
}

/// Returns only the independent vowels for convenience
pub fn independent_vowels() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();

    for (key, value) in vowels_static().iter() {
        map.insert(*key, value.independent);
    }

    map
}

/// Returns only the vowel modifiers (kars) for convenience
pub fn vowel_modifiers() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();

    for (key, value) in vowels_static().iter() {
        if let Some(dependent) = value.dependent {
            map.insert(*key, dependent);
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_vowel_rule_bytes_matches_rule_table() {
        assert!(
            VOWEL_RULES.iter().all(|(roman, _)| roman.is_ascii()),
            "vowel rule keys are byte-scanned ASCII signals"
        );
        assert_eq!(
            MAX_VOWEL_RULE_BYTES,
            VOWEL_RULES
                .iter()
                .map(|(roman, _)| roman.len())
                .max()
                .expect("vowel rule table should not be empty")
        );
    }
}
