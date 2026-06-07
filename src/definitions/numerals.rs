//! Definitions for Bengali numerals
//!
//! This file contains mappings for Bengali numerals (০-৯).

use super::insert_unique;
use std::collections::HashMap;
use std::sync::OnceLock;

/// A Roman input digit and its Bengali numeral output.
pub type NumeralRule = (&'static str, &'static str);

const NUMERAL_RULES: &[NumeralRule] = &[
    ("0", "\u{09e6}"),
    ("1", "\u{09e7}"),
    ("2", "\u{09e8}"),
    ("3", "\u{09e9}"),
    ("4", "\u{09ea}"),
    ("5", "\u{09eb}"),
    ("6", "\u{09ec}"),
    ("7", "\u{09ed}"),
    ("8", "\u{09ee}"),
    ("9", "\u{09ef}"),
];

/// Return the Bengali numeral for an ASCII decimal digit.
pub fn bengali_digit(digit: char) -> Option<&'static str> {
    match digit {
        '0' => Some("\u{09e6}"),
        '1' => Some("\u{09e7}"),
        '2' => Some("\u{09e8}"),
        '3' => Some("\u{09e9}"),
        '4' => Some("\u{09ea}"),
        '5' => Some("\u{09eb}"),
        '6' => Some("\u{09ec}"),
        '7' => Some("\u{09ed}"),
        '8' => Some("\u{09ee}"),
        '9' => Some("\u{09ef}"),
        _ => None,
    }
}

fn build_numerals() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::with_capacity(NUMERAL_RULES.len());

    for &(roman, bengali) in NUMERAL_RULES {
        insert_unique(&mut map, "numeral", roman, bengali);
    }

    map
}

/// Returns the ordered static numeral rule table.
pub const fn numeral_rules() -> &'static [NumeralRule] {
    NUMERAL_RULES
}

/// Returns a shared map of Latin numerals to Bengali numerals.
pub fn numerals_static() -> &'static HashMap<&'static str, &'static str> {
    static INSTANCE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    INSTANCE.get_or_init(build_numerals)
}

/// Returns a map of Latin numerals to Bengali numerals.
pub fn numerals() -> HashMap<&'static str, &'static str> {
    numerals_static().clone()
}
