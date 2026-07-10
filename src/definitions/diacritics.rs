//! Definitions for Bengali diacritics
//!
//! This file contains mappings for Bengali diacritics like
//! hasant, bisarga, chandrabindu, etc.

use super::insert_unique;
use std::collections::HashMap;
use std::sync::OnceLock;

/// A Roman input rule and its Bengali diacritic output.
pub type DiacriticRule = (&'static str, &'static str);

pub(crate) const HASANT: &str = "্";
pub(crate) const CHANDRABINDU: &str = "ঁ";
pub(crate) const BISARGA: &str = "ঃ";
pub(crate) const KHANDA_TA: &str = "ৎ";
pub(crate) const ANUSVAR: &str = "ং";

const DIACRITIC_RULES: &[DiacriticRule] = &[
    // Hasant - explicit user signal that suppresses the inherent vowel
    // and drives deterministic conjunct formation when placed between consonants.
    (",,", HASANT),
    ("^", CHANDRABINDU), // Chandrabindu
    (":", BISARGA),      // Bisarga
    ("t``", KHANDA_TA),  // Khanda Ta
    ("T``", KHANDA_TA),  // Khanda Ta alias
    ("ng", ANUSVAR),     // Anusvar
    ("M", ANUSVAR),      // Explicit anusvar alias; useful before g/gh without invoking ngg/nggh
];

fn build_diacritics() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::with_capacity(DIACRITIC_RULES.len());

    for &(roman, bengali) in DIACRITIC_RULES {
        insert_unique(&mut map, "diacritic", roman, bengali);
    }

    map
}

/// Returns the ordered static diacritic rule table.
pub const fn diacritic_rules() -> &'static [DiacriticRule] {
    DIACRITIC_RULES
}

/// Look up a Bengali diacritic from a Roman rule key without constructing or hashing a map.
pub fn diacritic_value(roman: &str) -> Option<&'static str> {
    match roman {
        ",," => Some(HASANT),
        "^" => Some(CHANDRABINDU),
        ":" => Some(BISARGA),
        "t``" | "T``" => Some(KHANDA_TA),
        "ng" | "M" => Some(ANUSVAR),
        _ => None,
    }
}

/// Returns a shared map of Bengali diacritics.
pub fn diacritics_static() -> &'static HashMap<&'static str, &'static str> {
    static INSTANCE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    INSTANCE.get_or_init(build_diacritics)
}

/// Returns a map of Bengali diacritics
pub fn diacritics() -> HashMap<&'static str, &'static str> {
    diacritics_static().clone()
}
